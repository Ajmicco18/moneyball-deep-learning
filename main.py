from pathlib import Path
from src.regression.linear_regression import LinRegression
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

    return rmse, cv_rmse, test_rmse, comp_df, ssr


def main():

    X, y = load_and_preprocess_classification_data(
        DATA_PATH)

    X_train, X_test, y_train, y_test = split_data(X, y)

    decision_tree = DecisionTree(max_depth=6)

    accuracy = decision_tree.train_model(X_train, y_train)

    test_accuracy = decision_tree.evaluate_model(X_test, y_test)

    decision_tree.plot()

    return accuracy, test_accuracy


if __name__ == "__main__":
    print(main())
