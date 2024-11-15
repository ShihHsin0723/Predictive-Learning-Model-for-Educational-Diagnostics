from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
)
import numpy as np
from matplotlib import pyplot as plt


def sigmoid(x):
    """Apply sigmoid function."""
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0.0
    for i in range(len(data["user_id"])):
        user_id = data["user_id"][i]
        question_id = data["question_id"][i]
        c_i_j = data["is_correct"][i]

        z = theta[user_id] - beta[question_id]
        sigmoid_func = sigmoid(z)

        log_lklihood += (c_i_j * np.log(sigmoid_func)) + ((1 - c_i_j) * np.log(1 - sigmoid_func))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    update_theta = np.zeros(theta.shape[0])
    update_beta = np.zeros(beta.shape[0])

    for i in range(len(data["is_correct"])):
        user_id = data["user_id"][i]
        question_id = data["question_id"][i]
        c_i_j = data["is_correct"][i]

        update_theta[user_id] += c_i_j - sigmoid(theta[user_id] - beta[question_id])

        update_beta[question_id] += sigmoid(theta[user_id] - beta[question_id]) - c_i_j

    theta += lr * update_theta
    beta += lr * update_beta
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    num_users = max(data["user_id"]) + 1
    num_questions = max(data["question_id"]) + 1

    theta = np.zeros(num_users)
    beta = np.zeros(num_questions)
    # theta = np.full(542, 0.5)
    # beta = np.zeros(1774)

    train_lld_lst = []
    val_lld_lst = []
    val_acc_lst = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        train_lld_lst.append(neg_lld)

        val_neg_lld = neg_log_likelihood(data=val_data, theta=theta, beta=beta)
        val_lld_lst.append(val_neg_lld)

        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)

        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst, train_lld_lst, val_lld_lst


def evaluate(data, theta, beta):
    """Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) / len(data["is_correct"])


def main():
    train_data = load_train_csv("./data")
    # You may optionally use the sparse matrix.
    # sparse_matrix = load_train_sparse("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    learning_rate = 0.01
    num_iter = 100

    theta, beta, val_acc, train_lld, val_lld = irt(train_data, val_data, learning_rate, num_iter)
    val_accuracy = evaluate(data=val_data, theta=theta, beta=beta)
    test_accuracy = evaluate(data=test_data, theta=theta, beta=beta)

    print(f"Hyperparameters used were: \n learning_rate: {learning_rate} \n number of iterations: {num_iter}")

    print(f"Validation accuracy: {val_accuracy}")
    print(f"Test accuracy: {test_accuracy}")

    # Plots
    plt.figure(figsize=(12, 6))
    plt.plot(train_lld, label='Negative Log-Likelihood Train Set')
    plt.plot(val_lld, label='Negative Log-Likelihood Validation Set')
    plt.title('Training and Validation Costs')
    plt.xlabel('Iterations')
    plt.ylabel('Negative Log Likelihood)')
    plt.legend()
    plt.savefig("training_val_cost.png")
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    theta_range = np.linspace(-5, 5, 500)
    question_indices = [10, 100, 400]

    prob_correct_j1 = sigmoid(theta_range - beta[question_indices[0]])
    prob_correct_j2 = sigmoid(theta_range - beta[question_indices[1]])
    prob_correct_j3 = sigmoid(theta_range - beta[question_indices[2]])

    plt.figure(figsize=(12, 6))
    plt.plot(theta_range, prob_correct_j1, label=f"Question {question_indices[0]}")
    plt.plot(theta_range, prob_correct_j2, label=f"Question {question_indices[1]}")
    plt.plot(theta_range, prob_correct_j3, label=f"Question {question_indices[2]}")
    plt.xlabel("Theta")
    plt.ylabel(f"Probability of Correct Response")
    plt.title("Probability of Correct Response as a Function of User Ability")
    plt.legend()
    plt.savefig("2d.png")
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
