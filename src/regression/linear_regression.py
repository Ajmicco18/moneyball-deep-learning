from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score


class LinRegression():

    def __init__(self, X_train, X_test, y_train, y_test):

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.lr = LinearRegression()

    def fit_model(self):

        self.lr.fit(self.X_train, self.y_train)

    def make_predictions(self):

        preds = self.fit_model().predict(self.X_train)

        print(preds)

        return preds

    def evaluate_model(self):

        preds = self.make_predictions()

        lin_rmse = root_mean_squared_error(self.y_train, preds)

        return (lin_rmse)
