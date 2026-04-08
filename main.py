from pathlib import Path
from src.regression.linear_regression import LinRegression
from src.regression.neural_network import RegressionNeuralNet
from src.classification.decision_tree import DecisionTree
from src.data_preprocessing import *
from configs.config import *


def lin_reg():
    X, y = load_and_preprocess_regression_data(DATA_PATH)

    X_train, X_test, y_train, y_test = split_data(X, y)

    reg = LinRegression()

    rmse = reg.train_model(X_train, y_train)

    coefficents = reg.get_coefficients()

    intercept = reg.get_intercept()

    cv_rmse = reg.get_cv_mean_rmse(X_train, y_train)

    test_preds = reg.make_predictions(X_test)

    test_rmse = reg.evaluate_model(X_test, y_test)

    comp_df, ssr = reg.calculate_ssr(y_test, test_preds)

    return f"""
        RMSE: {rmse}, 
        Cross-Validation RMSE: {cv_rmse},
        Test RMSE: {test_rmse},
        Residual Dataframe: \n {comp_df},
        Sum of Squared Residuals: {ssr}"""


def decision_tree():
    X, y = load_and_preprocess_classification_data(
        DATA_PATH)

    X_train, X_test, y_train, y_test = split_data(X, y)

    decision_tree = DecisionTree(max_depth=6)

    accuracy, precision, recall, f1 = decision_tree.train_model(
        X_train, y_train)

    test_accuracy, test_precision, test_recall, test_f1 = decision_tree.evaluate_model(
        X_test, y_test)

    decision_tree.plot()

    return f"""
        Training Data: 
        -----------------------------------
        Accuracy: {accuracy}, 
        Precision: {precision},
        Recall: {recall},
        F1-Score: {f1},
        Test Data: 
        -----------------------------------
        Accuracy: {test_accuracy}, 
        Precision: {test_precision},
        Recall: {test_recall},
        F1-Score: {test_f1},"""


def main():
    # print(lin_reg())
    # print(decision_tree())
    X, y = load_and_preprocess_regression_data(DATA_PATH)

    X_train, x_test, y_train, y_test = split_data(X, y)

    X_train, x_test, y_train, y_test = convert_to_tensors(
        X_train, x_test, y_train, y_test)

    nn = RegressionNeuralNet(input_size=14)

    preds = nn(X_train)

    # device = nn.find_device()

    print(preds)


if __name__ == "__main__":
    main()
