import numpy as np
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
from utils import (
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
    sparse_matrix_evaluate,
)


def knn_impute_by_user(matrix, valid_data, k):
    """Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    # print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix.T).T
    acc = sparse_matrix_evaluate(valid_data, mat)
    # print("Validation Accuracy: {}".format(acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse("./data").toarray()
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.
    k_values = [1, 6, 11, 16, 21, 26]

    # BY USER
    print("BY USER:")
    val_accuracies = []
    optimal_k = k_values[0]
    highest_accuracy = 0

    # Find validation accuracy and the optimal k
    for k in k_values:
        accuracy = knn_impute_by_user(sparse_matrix, val_data, k)
        val_accuracies.append(accuracy)
        print(f"Valiation Accuracy with K = {k}: {accuracy}")

        if accuracy >= highest_accuracy:
            optimal_k = k
            highest_accuracy = accuracy

    print(f"Optimal k (k*) is {optimal_k} with validation accuracy {highest_accuracy}")

    # Plot the validation accuracies for each k value
    plt.plot(k_values, val_accuracies, marker='o')
    plt.title("Validation Accuracy for Different k Values")
    plt.xlabel("k (Number of Neighbors)")
    plt.ylabel("Validation Accuracy")
    plt.grid(True)

    plt.savefig("KNN_validation_accuracy_plot_by_user.png", format="png")
    plt.show()

    # Report the final test accuracy using the k*
    test_accuracy = knn_impute_by_user(sparse_matrix, test_data, optimal_k)
    print(f"Final Test Accuracy with k = {optimal_k}: {test_accuracy}")

    # BY ITEM
    print("BY ITEM:")
    val_accuracies = []
    optimal_k = k_values[0]
    highest_accuracy = 0

    # Find validation accuracy and the optimal k
    for k in k_values:
        accuracy = knn_impute_by_item(sparse_matrix, val_data, k)
        val_accuracies.append(accuracy)
        print(f"Valiation Accuracy with K = {k}: {accuracy}")

        if accuracy >= highest_accuracy:
            optimal_k = k
            highest_accuracy = accuracy

    print(f"Optimal k (k*) is {optimal_k} with validation accuracy {highest_accuracy}")

    # Plot the validation accuracies for each k value
    plt.plot(k_values, val_accuracies, marker='o')
    plt.title("Validation Accuracy for Different k Values")
    plt.xlabel("k (Number of Neighbors)")
    plt.ylabel("Validation Accuracy")
    plt.grid(True)

    plt.savefig("KNN_validation_accuracy_plot_by_item.png", format="png")
    plt.show()

    # Report the final test accuracy using the k*
    test_accuracy = knn_impute_by_item(sparse_matrix, test_data, optimal_k)
    print(f"Final Test Accuracy with k = {optimal_k}: {test_accuracy}")
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
