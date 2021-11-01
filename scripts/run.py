import numpy as np
from proj1_helpers import *
import implementations as imp
from pre_processing import *

TRAIN_PATH = '../resources/train.csv'
TEST_PATH = '../resources/test.csv'
OUTPUT_PATH = '../results.csv'


def main():
    print("Loading data.")

    (tx_1, y_1, ids_1), (tx_2, y_2, ids_2), (tx_3, y_3,
                                             ids_3), indexes, parameters = process_data(TRAIN_PATH)
    (xtest_1, ids_1), (xtest_2, ids_2), (xtest_3,
                                         ids_3) = process_test(TEST_PATH, indexes, parameters)

    print("Expansion on data.")

    x_train_1 = expand_with_pairwise_products(tx_1, 10)
    x_train_2 = expand_with_pairwise_products(tx_2, 12)
    x_train_3 = expand_with_pairwise_products(tx_3, 12)

    x_test_1 = expand_with_pairwise_products(xtest_1, 10)
    x_test_2 = expand_with_pairwise_products(xtest_2, 12)
    x_test_3 = expand_with_pairwise_products(xtest_3, 12)

    print("Model tuning with ridge regression.")

    w1, l1 = imp.ridge_regression(y_1, x_train_1, 0.0001)
    w2, l2 = imp.ridge_regression(y_2, x_train_2, 0.0004124626382901352)
    w3, l3 = imp.ridge_regression(y_3, x_train_3, 0.0008376776400682916)

    print("Prediction of the test.")

    y_pred1 = predict_labels(w1, x_test_1)
    y_pred2 = predict_labels(w2, x_test_2)
    y_pred3 = predict_labels(w3, x_test_3)

    y = np.concatenate((y_pred1, y_pred2))
    y = np.concatenate((y, y_pred3))

    ids = np.concatenate((ids_1, ids_2))
    ids = np.concatenate((ids, ids_3))

    print("Creating the csv submission file.")

    create_csv_submission(ids, y, OUTPUT_PATH)


if __name__ == '__main__':
    main()
