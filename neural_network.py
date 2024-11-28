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
    """Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
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


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        """Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k)
        self.h = nn.Linear(k, num_question)

    def get_weight_norm(self):
        """Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # TODO:                                                             #
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.                              #
        #####################################################################
        # out = inputs
        encoded = torch.sigmoid(self.g(inputs))
        decoded = torch.sigmoid(self.h(encoded))
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return decoded


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch):
    """Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """
    # TODO: Add a regularizer to the cost function.

    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    train_losses = []
    validation_accs = []

    for epoch in range(0, num_epoch):
        train_loss = 0.0

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[nan_mask] = output[nan_mask]

            # loss = torch.sum((output - target) ** 2.0)
            loss = torch.sum((output - target) ** 2.0) + 0.5 * lamb * model.get_weight_norm()
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        train_losses.append(train_loss)
        valid_acc = evaluate(model, zero_train_data, valid_data)
        validation_accs.append(valid_acc)

        valid_acc = evaluate(model, zero_train_data, valid_data)
        print(
            "Epoch: {} \tTraining Cost: {:.6f}\t " "Valid Acc: {}".format(
                epoch, train_loss, valid_acc
            )
        )

    # Q3 d) Report and plot the training and validation objectives
    # epochs = range(1, num_epoch + 1)
    # plt.show()
    # plt.plot(epochs, train_losses, label="Training Loss", color="blue")
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.title("Epoch vs. Training Loss")
    # plt.legend()
    # plt.savefig("epoch_vs_training_loss.png")
    # plt.show()
    #
    # plt.plot(epochs, validation_accs, label="Validation Accuracy", color="green")
    # plt.xlabel("Epoch")
    # plt.ylabel("Accuracy")
    # plt.title("Epoch vs. Validation Accuracy")
    # plt.legend()
    # plt.savefig("epoch_vs_valid_acc.png")
    # plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def evaluate(model, train_data, valid_data):
    """Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    #####################################################################
    # TODO:                                                             #
    # Try out 5 different k and select the best k using the             #
    # validation set.                                                   #
    #####################################################################

    # Set optimization hyperparameters.
    lr = 0.01
    num_epoch = 30
    lamb = 0.001

    # Q3 c) - commented out since we've found the best k and parameters to use.
    # Set model hyperparameters.
    # k_values = [10, 50, 100, 200, 500]
    # best_k = None
    # best_valid_acc = 0
    # results = {}
    #
    # for k in k_values:
    #     model = AutoEncoder(num_question=train_matrix.shape[1], k=k)
    #     train(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epoch)
    #     valid_acc = evaluate(model, zero_train_matrix, valid_data)
    #     results[k] = valid_acc
    #
    #     if valid_acc > best_valid_acc:
    #         best_k = k
    #         best_valid_acc = valid_acc
    #
    #     print(f"k = {k}, validation accuarcy = {valid_acc}\n")
    #
    # print(f"The best k: {best_k} with validation accuracy: {best_valid_acc}")

    # this is the model for the best k
    best_k = 100
    model = AutoEncoder(num_question=train_matrix.shape[1], k=best_k)
    train(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epoch)

    # Q3 d)
    # For test set
    test_model = AutoEncoder(num_question=train_matrix.shape[1], k=best_k)
    train(test_model, lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epoch)
    test_acc = evaluate(test_model, zero_train_matrix, test_data)
    print(f"For test set on k = {best_k}, test accuracy = {test_acc}.")

    # Q3 e) tuning regularization term - commented out since we've found the best lambda value to use.
    # best_lamb = None
    # best_valid_acc = 0
    # results = {}
    #
    # lamb_list = [0.001, 0.01, 0.1, 1]
    # for lamb in lamb_list:
    #     model = AutoEncoder(num_question=train_matrix.shape[1], k=best_k)
    #     train(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epoch)
    #     valid_acc = evaluate(model, zero_train_matrix, valid_data)
    #     results[lamb] = valid_acc
    #
    #     if valid_acc > best_valid_acc:
    #         best_lamb = lamb
    #         best_valid_acc = valid_acc
    #
    #     print(f"lambda = {lamb}, validation accuarcy = {valid_acc}\n")
    #
    # print(f"The best lambda: {best_lamb} with validation accuracy: {best_valid_acc}")

    # this is the model for the best lambda
    best_lamb = 0.001
    model = AutoEncoder(num_question=train_matrix.shape[1], k=best_k)
    train(model, lr, best_lamb, train_matrix, zero_train_matrix, valid_data, num_epoch)
    valid_acc = evaluate(model, zero_train_matrix, valid_data)
    print(f"For validation set on lambda = {best_lamb}, validation accuarcy = {valid_acc}\n")

    # getting the testing accuraccies:
    test_model = AutoEncoder(num_question=train_matrix.shape[1], k=best_k)
    train(test_model, lr, best_lamb, train_matrix, zero_train_matrix, valid_data, num_epoch)
    test_acc = evaluate(test_model, zero_train_matrix, test_data)
    print(f"For test set on lambda = {best_lamb}, test accuracy = {test_acc}.")

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
