from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def preprocess_data(df):
    """
    Function that will preprocess the data by scaling it and removing/replace any null values
    """

    X = df.drop(columns=["Team", "League", "W"])
    y = df["W"]

    imputer = SimpleImputer()

    X = imputer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    return
