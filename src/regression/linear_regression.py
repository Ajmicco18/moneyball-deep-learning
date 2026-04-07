from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd


class LinRegression():

    def __init__(self):

        self.lr = LinearRegression()

    def train_model(self, X_train, y_train):

        self.lr.fit(X_train, y_train)

        preds = self.lr.predict(X_train)

        lin_rmse = root_mean_squared_error(y_train, preds)

        return lin_rmse

    def evaluate_model(self, X_test, y_test):
        test_preds = self.lr.predict(X_test)

        test_rmse = root_mean_squared_error(y_test, test_preds)

        return (test_rmse)

    def make_predictions(self, X):

        return self.lr.predict(X)

    def get_cv_mean_rmse(self, X_train, y_train):

        cv_rmse = -cross_val_score(self.lr, X_train, y_train,
                                   scoring="neg_root_mean_squared_error", cv=10)

        average_rmse = np.mean(cv_rmse)

        return average_rmse

    def calculate_ssr(self, y_test, test_preds):
        comp_df = pd.DataFrame({
            "Actual": y_test.values,
            "Predicted": test_preds,
            "SSR": (y_test.values - test_preds)
        })

        ssr = np.sum(comp_df["SSR"] ** 2)

        return comp_df, ssr

    def get_coefficients(self):

        coefficients = self.lr.coef_

        return coefficients

    def get_intercept(self):

        intercept = self.lr.intercept_

        return intercept

    def plot(self):
        return
