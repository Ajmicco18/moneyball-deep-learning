from sklearn.linear_model import LinearRegression
from data.data_preprocessing import preprocess_data


def lin_reg():

    X_train, X_test, y_train, y_test = preprocess_data()

    lr = LinearRegression()

    lr.fit(X_train, y_train)

    preds = lr.predict(X_train)

    print(preds)
