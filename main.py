import torch
import torch.nn as nn
import torchmetrics
from src.regression.linear_regression import LinRegression
from src.regression.neural_network import RegressionNeuralNet
from src.regression.wide_deep_nn import WideAndDeepNN
from src.classification.decision_tree import DecisionTree
from src.classification.neural_network import ClassificationNeuralNet
from src.classification.multi_input_nn import MultiInputNeuralNet
from src.data_preprocessing import *
from src.trainer import *
from configs.config import *

# NOTE: INTEND TO FURTHER MODULARIZE THIS PROJECT SO I DO NOT HAVE TO CALL EACH FUNCTION IN THE WAY BELOW
# ****FUNCTIONS TO CALL REGRESSION MODEL TRAINING****

# Linear Regression Model


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

# Regression Neural Net Model


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

    model.plot_neural_net(model)

    avg_train_rmse = np.mean(history["train_metrics"])
    avg_val_rmse = np.mean(history["validation_metrics"])

    return f"""
    Average Training RMSE: {avg_train_rmse:.4f}
    Average Validation RMSE: {avg_val_rmse:.4f}
    """

# Wide and Deep Regression Neural Net Model


def regression_wide_and_deep_nn():

    X, y = load_and_preprocess_regression_data(DATA_PATH)

    X_train, X_test, y_train, y_test = split_data(X, y)

    X_train, X_test, y_train, y_test = convert_to_tensors(
        X_train, X_test, y_train, y_test)

    train_loader = create_data_loaders(X_train, y_train)

    val_loader = create_data_loaders(X_test, y_test)

    torch.manual_seed(42)

    model = WideAndDeepNN(input_size=14)

    learning_rate = 0.02

    mse = nn.MSELoss()

    rmse = torchmetrics.MeanSquaredError(squared=False)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate)

    history = train_model(model, mse, optimizer,
                          train_loader, val_loader, 50, rmse)

    model.plot_losses(history["train_losses"])

    model.plot_learning_curve(
        50, history["train_metrics"], history["validation_metrics"])

    model.plot_neural_net(model)

    avg_train_rmse = np.mean(history["train_metrics"])
    avg_val_rmse = np.mean(history["validation_metrics"])

    return f"""
    Average Training RMSE: {avg_train_rmse:.4f}
    Average Validation RMSE: {avg_val_rmse:.4f}
    """

# ****FUNCTIONS TO CALL CLASSIFICATION MODEL TRAINING****

# Decision Tree Classifier Model


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

# Classification Neural Net Model


def classification_nn():
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

    model.plot_neural_net(model)

    avg_train_f1 = np.mean(history["train_metrics"])
    avg_val_f1 = np.mean(history["validation_metrics"])

    return f"""
    Average Training F1-Score: {avg_train_f1:.4f}
    Average Validation F1-Score: {avg_val_f1:.4f}
    """

# Multi-Input Classification Neural Net Model


def classification_multi_input_nn():
    X, y = load_and_preprocess_classification_data(DATA_PATH)

    X_train, X_test, y_train, y_test = split_data(X, y)

    X_train, X_test, y_train, y_test = convert_to_tensors(
        X_train, X_test, y_train, y_test)

    train_loader = create_wide_deep_data_loaders(
        X_train[:, [0, 4, 5, 6, 7, 10, 11]], X_train[:, [1, 2, 3, 8, 9]], y_train)

    val_loader = create_wide_deep_data_loaders(
        X_test[:, [0, 4, 5, 6, 7, 10, 11]], X_test[:, [1, 2, 3, 8, 9]], y_test)

    torch.manual_seed(42)

    model = MultiInputNeuralNet(7)

    learning_rate = 0.01

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    xentropy = nn.BCEWithLogitsLoss()

    f1 = torchmetrics.F1Score(task="binary")
    precision = torchmetrics.Precision(task="binary")
    recall = torchmetrics.Recall(task="binary")
    confusion = torchmetrics.ConfusionMatrix(task="binary")
    accuracy = torchmetrics.Accuracy(task="binary")

    history = train_multi_input_model(model, xentropy, optimizer,
                                      train_loader, val_loader, 50, f1)

    model.plot_losses(history["train_losses"])

    model.plot_learning_curve(
        50, history["train_metrics"], history["validation_metrics"])

    model.plot_neural_net(model)

    avg_train_f1 = np.mean(history["train_metrics"])
    avg_val_f1 = np.mean(history["validation_metrics"])

    return f"""
    Average Training F1-Score: {avg_train_f1:.4f}
    Average Validation F1-Score: {avg_val_f1:.4f}
    """


def main():
    # print(lin_reg())
    # print(decision_tree())
    # print(regression_nn())
    # print(classification_nn())
    # print(regression_wide_and_deep_nn())
    # print(classification_multi_input_nn())

    return


if __name__ == "__main__":
    main()
