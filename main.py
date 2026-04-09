from pathlib import Path
import torch
import torch.nn as nn
import torchmetrics
from src.regression.linear_regression import LinRegression
from src.regression.neural_network import RegressionNeuralNet
from src.classification.decision_tree import DecisionTree
from src.classification.neural_network import ClassificationNeuralNet
from src.data_preprocessing import *
from src.trainer import *
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


def regression_nn():
    X, y = load_and_preprocess_regression_data(DATA_PATH)

    X_train, X_test, y_train, y_test = split_data(X, y)

    X_train, X_test, y_train, y_test = convert_to_tensors(
        X_train, X_test, y_train, y_test)

    train_loader = create_data_loaders(X_train, y_train)

    val_loader = create_data_loaders(X_test, y_test)

    torch.manual_seed(42)

    model = RegressionNeuralNet(input_size=14)

    learning_rate = 0.01

    mse = nn.MSELoss()

    rmse = torchmetrics.MeanSquaredError(squared=False)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate)

    history = train_model(
        model, mse, optimizer, train_loader, val_loader, 50, rmse)

    model.plot_losses(history["train_losses"])

    model.plot_learning_curve(
        50, history["train_metrics"], history["validation_metrics"])

    return history


def decision_tree():
    X, y = load_and_preprocess_classification_data(DATA_PATH)

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


def classification_nn():

    return


def main():
    # print(lin_reg())
    # print(decision_tree())
    # print(regression_nn())

    X, y = load_and_preprocess_classification_data(DATA_PATH)

    X_train, X_test, y_train, y_test = split_data(X, y)

    X_train, X_test, y_train, y_test = convert_to_tensors(
        X_train, X_test, y_train, y_test)

    train_loader = create_data_loaders(X_train, y_train)

    val_loader = create_data_loaders(X_test, y_test)

    torch.manual_seed(42)

    model = ClassificationNeuralNet(input_size=12, num_classes=1)

    learning_rate = 0.01

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    xentropy = nn.BCEWithLogitsLoss()

    f1 = torchmetrics.F1Score(task="binary")
    precision = torchmetrics.Precision(task="binary")
    recall = torchmetrics.Recall(task="binary")
    confusion = torchmetrics.ConfusionMatrix(task="binary")
    accuracy = torchmetrics.Accuracy(task="binary")

    history = train_model(model, xentropy, optimizer,
                          train_loader, val_loader, 50, f1)

    model.plot_losses(history["train_losses"])

    model.plot_learning_curve(
        50, history["train_metrics"], history["validation_metrics"])

    return history


if __name__ == "__main__":
    main()
