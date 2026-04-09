import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.model_selection import train_test_split

# ****GENERATING DATA PREPROCESSING PIPELINES****
num_pipeline = make_pipeline(SimpleImputer(
    strategy="median"), StandardScaler())

cat_pipeline = make_pipeline(SimpleImputer(
    strategy="most_frequent"), OrdinalEncoder())

proprocessing = make_column_transformer(
    (num_pipeline, make_column_selector(dtype_include=np.number)),
    (cat_pipeline, make_column_selector(dtype_include=object)),
)


def load_and_preprocess_regression_data(data_path):

   # ****FUNCTION THAT LOADS AND PREPROCESSES DATA PRIOR TO USAGE BY REGRESSION MODELS****

    data = pd.read_csv(data_path)

    X = data.drop(columns=["W"])
    y = data["W"]

    X_prepared = proprocessing.fit_transform(X)

    return X_prepared, y


def load_and_preprocess_classification_data(data_path):

    # ****FUNCTION THAT LOADS AND PREPROCESSES DATA PRIOR TO USAGE BY CLASSIFICATION MODELS****

    data = pd.read_csv(data_path)

    X = data.drop(columns=["Playoffs", "RankSeason", "RankPlayoffs"])
    y = data["Playoffs"]

    X_prepared = proprocessing.fit_transform(X)

    return X_prepared, y


def split_data(X, y):

    # ****FUNCTION THAT SPLITS DATA INTO TRAINING AND TEST SETS****

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42)

    return X_train, X_test, y_train, y_test


def convert_to_tensors(X_train, X_test, y_train, y_test):

    # ****FUNCTION THAT CONVERTS DATA SPLITS INTO TENSORS FOR PYTORCH****

    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.FloatTensor(y_train.values).view(-1, 1)
    y_test = torch.FloatTensor(y_test.values).view(-1, 1)

    return X_train, X_test, y_train, y_test


def create_data_loaders(X, y, batch_size=32):

    # ****FUNCTION THAT TURNS FEATURES AND TARGET VARIABLES INTO A DATALOADER****

    # Create data loaders
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size, shuffle=True)

    return loader


"""
def convert_to_classification_tensors(X_train, X_test, y_train, y_test):

    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train.values)
    y_test = torch.LongTensor(y_test.values)

    return X_train, X_test, y_train, y_test"""
