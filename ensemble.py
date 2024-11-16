from utils import load_train_csv, load_valid_csv, load_public_test_csv
import numpy as np
import item_response


def bootstrap_sample(data):
    indices = np.random.randint(len(data['user_id']), size=len(data['user_id']))
    return {
        'user_id': list(np.array(data['user_id'])[indices]),
        'question_id': list(np.array(data['question_id'])[indices]),
        'is_correct': list(np.array(data['is_correct'])[indices])
    }


def aggregate_predictions(predictions, num_models):
    aggregated = np.sum(predictions, axis=0)
    return (aggregated / num_models > 0.5).astype(int)


def ensemble_irt(train_data, val_data, test_data, lr, iterations, num_models):
    val_predictions = []
    test_predictions = []

    for _ in range(num_models):
        bootstrapped_data = bootstrap_sample(train_data)

        theta, beta, _, _, _ = item_response.irt(bootstrapped_data, val_data, lr, iterations)

        val_pred = [
            item_response.sigmoid(theta[u] - beta[q]) >= 0.5
            for u, q in zip(val_data['user_id'], val_data['question_id'])
        ]
        val_predictions.append(val_pred)

        test_pred = [
            item_response.sigmoid(theta[u] - beta[q]) >= 0.5
            for u, q in zip(test_data['user_id'], test_data['question_id'])
        ]
        test_predictions.append(test_pred)

    val_final_pred = aggregate_predictions(val_predictions, num_models)
    test_final_pred = aggregate_predictions(test_predictions, num_models)

    val_accuracy = np.mean(val_data['is_correct'] == val_final_pred)
    test_accuracy = np.mean(test_data['is_correct'] == test_final_pred)

    return val_accuracy, test_accuracy


def main():
    train_data = load_train_csv("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    learning_rate = 0.01
    num_iterations = 100
    num_models = 3

    val_acc, test_acc = ensemble_irt(train_data, val_data, test_data, learning_rate, num_iterations, num_models)

    print(f"Validation Accuracy: {val_acc}")
    print(f"Test Accuracy: {test_acc}")


if __name__ == "__main__":
    main()
