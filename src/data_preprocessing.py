import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_and_preprocess_data(data_path):
    """
    Function that will preprocess the data by scaling it and removing/replace any null values
    """
    data = pd.read_csv(data_path)

    X = data.drop(columns=["Team", "League", "W"])
    y = data["W"]

    imputer = SimpleImputer()

    X = imputer.fit_transform(X)

    return X, y


def split_data(X, y, train_ratio):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
