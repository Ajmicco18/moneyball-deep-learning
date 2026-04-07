import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.model_selection import train_test_split

num_pipeline = make_pipeline(SimpleImputer(
    strategy="median"), StandardScaler())

cat_pipeline = make_pipeline(SimpleImputer(
    strategy="most_frequent"), OrdinalEncoder())

proprocessing = make_column_transformer(
    (num_pipeline, make_column_selector(dtype_include=np.number)),
    (cat_pipeline, make_column_selector(dtype_include=object)),
)


def load_and_preprocess_regression_data(data_path):
    """
    Function that will preprocess the data by scaling it and removing/replace any null values
    """
    data = pd.read_csv(data_path)

    X = data.drop(columns=["W"])
    y = data["W"]

    X_prepared = proprocessing.fit_transform(X)

    return X_prepared, y


def load_and_preprocess_classification_data(data_path):

    data = pd.read_csv(data_path)

    X = data.drop(columns=["Playoffs", "RankSeason", "RankPlayoffs"])
    y = data["Playoffs"]

    X_prepared = proprocessing.fit_transform(X)

    return X_prepared, y


def split_data(X, y):  # train_ratio parameter

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


def create_data_loaders(X_train, y_train, batch_size=32):
    """Create PyTorch data loaders"""
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)

    return train_loader


"""
def split_data_tensor(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)

    return X_train, X_test, y_train, y_test
"""
