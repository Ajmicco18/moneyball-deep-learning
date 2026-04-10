from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score
from configs.config import PLOTS_PATH
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class LinRegression():

    def __init__(self):

        self.lr = LinearRegression()

    # ****FUNCTION TO TRAIN MODEL USING TRAINING DATA****
    def train_model(self, X_train, y_train):

        self.lr.fit(X_train, y_train)

        preds = self.lr.predict(X_train)

        lin_rmse = root_mean_squared_error(y_train, preds)

        self.plot(y_train, preds, 'training_fit.png')

        return lin_rmse

    # ****FUNCTION TO EVALUATE MODEL USING TEST DATA****
    def evaluate_model(self, X_test, y_test):
        test_preds = self.lr.predict(X_test)

        test_rmse = root_mean_squared_error(y_test, test_preds)

        self.plot(y_test, test_preds, 'test_fit.png')

        return (test_rmse)

    # ****FUNCTION TO MAKE GENERAL PREDICTIONS****
    def make_predictions(self, X):

        return self.lr.predict(X)

    # ****FUNCTION TO CALCULATE CROSS-VALIDATION RMSE****
    def get_cv_mean_rmse(self, X_train, y_train):

        cv_rmse = -cross_val_score(self.lr, X_train, y_train,
                                   scoring="neg_root_mean_squared_error", cv=10)

        average_rmse = np.mean(cv_rmse)

        return average_rmse

    # ****FUNCTION TO CALCULATE SUM OF SQUARED RESIDUALS****
    def calculate_ssr(self, y_test, test_preds):
        comp_df = pd.DataFrame({
            "Actual": y_test.values,
            "Predicted": test_preds,
            "SSR": (y_test.values - test_preds)
        })

        ssr = np.sum(comp_df["SSR"] ** 2)

        return comp_df, ssr

    # ****FUNCTION TO GET COEFFICIENTS FOR EACH FEATURE****
    def get_coefficients(self):

        coefficients = self.lr.coef_

        return coefficients

    # ****FUNCTION TO GET MODEL'S INTERCEPT****
    def get_intercept(self):

        intercept = self.lr.intercept_

        return intercept

    # ****FUNCTION TO PLOT BEST FIT REGRESSION LINE****
    def plot(self, y, preds, filename):

        filepath = PLOTS_PATH / 'regressions/linear_regression' / filename

        plt.figure()
        plt.scatter(y, preds, color='blue', alpha=0.5)
        plt.plot([y.min(), y.max()], [y.min(), y.max()],
                 color='red', linewidth=2)
        plt.title('Actual vs. Predicted')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')

        plt.savefig(filepath,  bbox_inches="tight", dpi=300)
        plt.close()
