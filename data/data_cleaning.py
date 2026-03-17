import pandas as pd


def load_data(dataset):
    """
    Args:
        dataset (string): A .csv file that contains data from a .csv in the data folder

    Returns:
        DataFrame : A dataframe containing data from the .csv dataset passed in as a parameter
    """
    df = pd.read_csv(dataset)

    return df
