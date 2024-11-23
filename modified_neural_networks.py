import numpy as np
from matplotlib import pyplot as plt
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch

from utils import (
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
)


def load_data(base_path="./data"):
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = torch.relu(self.fc1(z))
        raw_output = torch.clamp(self.fc2(h), min=-10, max=10)  # avoid out of range raw output
        x_hat = torch.sigmoid(raw_output)
        return x_hat


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100, hidden_dim=60):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(num_question, k, hidden_dim)
        self.decoder = Decoder(k, num_question, hidden_dim)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, inputs):
        mu, logvar = self.encoder(inputs)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar


def vae_loss(reconstructed, original, mu, log_var):
    bce = F.binary_cross_entropy(reconstructed, original, reduction="sum")
    kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return bce + kl_div


def train(model, lr, train_data, zero_train_data, valid_data, num_epoch):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    train_losses = []
    validation_accs = []

    for epoch in range(0, num_epoch):
        train_loss = 0.0

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            reconstructed, mu, log_var = model(inputs)

            # Mask the target to exclude missing entries
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[nan_mask] = reconstructed[nan_mask]

            loss = vae_loss(reconstructed, target, mu, log_var)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        train_losses.append(train_loss)
        valid_acc, _, _ = evaluate(model, zero_train_data, valid_data)
        validation_accs.append(valid_acc)

        print(f"Epoch: {epoch}, Train Loss: {train_loss}, Validation Accuracy: {valid_acc}")

    plot_metrics(train_losses, validation_accs)


def evaluate(model, train_data, valid_data):
    """Evaluate the valid_data on the current model, including sensitivity and specificity.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list, question_id: list, is_correct: list}
    :return: accuracy, sensitivity, specificity
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output, _, _ = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.50
        actual = valid_data["is_correct"][i]

        # Evaluate sensitivity and specificity
        if guess == actual:
            correct += 1
            if guess:
                true_positive += 1
            else:
                true_negative += 1
        else:
            if guess:
                false_positive += 1
            else:
                false_negative += 1
        total += 1

    # Accuracy
    accuracy = correct / float(total)

    # Sensitivity
    sensitivity = true_positive / float(true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0

    # Specificity
    specificity = true_negative / float(true_negative + false_positive) if (true_negative + false_positive) > 0 else 0.0

    return accuracy, sensitivity, specificity


def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()
    lr = 0.0015
    num_epoch = 50
    k = 100
    print(f"\nTraining AutoEncoder with k={k}")
    model = AutoEncoder(num_question=train_matrix.shape[1], k=k)
    train(model, lr, train_matrix, zero_train_matrix, valid_data, num_epoch)
    validation_acc, validation_sensitivity, validation_specificity = evaluate(model, zero_train_matrix, valid_data)
    print(f"For validation set on k = {k}, validation accuracy = {validation_acc}, validation sensitivity = "
          f"{validation_sensitivity}, validation specificity = {validation_specificity}.")

    test_model = AutoEncoder(num_question=train_matrix.shape[1], k=k)
    train(test_model, lr, train_matrix, zero_train_matrix, valid_data, num_epoch)
    test_acc, test_sensitivity, test_specificity = evaluate(test_model, zero_train_matrix, test_data)
    print(f"For test set on k = {k}, test accuracy = {test_acc}, test sensitivity = {test_sensitivity}"
          f", test specificity = {test_specificity}.")


def plot_metrics(train_losses, validation_accs):
    epochs = range(1, len(train_losses) + 1)

    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig("training_loss.png")
    plt.show()

    # Plot validation accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, validation_accs, label="Validation Accuracy", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig("validation_accuracy.png")
    plt.show()


if __name__ == "__main__":
    main()
