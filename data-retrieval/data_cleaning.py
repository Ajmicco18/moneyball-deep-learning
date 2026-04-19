import pandas as pd
from configs.config import DATA_PATH


def load_data(dataset):
    """
    Args:
        dataset (string): A .csv file that contains data from a .csv in the data folder

    Returns:
        DataFrame : A dataframe containing data from the .csv dataset passed in as a parameter
    """
    df = pd.read_csv(dataset)

    return df


def merge_datasets():

    # loading in the datasets
    batting_df = load_data("2013-2025-batting.csv")
    opp_batting_df = load_data("2013-2025-opp-batting.csv")
    standings_df = load_data("2013-2025-standings.csv")
    playoffs_df = load_data("2013-2025-playoffs.csv")

    # ******CLEANING STANDINGS DATA********
    standings_df = clean_standings(standings_df)

    # ********CLEANING BATTING DATA********
    batting_df = clean_batting(batting_df)

    # *******CLEAING OPP BATTING DATA********
    opp_batting_df = clean_opp_batting(opp_batting_df)

    # ********MERGING DATA FRAMES TO GET COMPLETE DATASET********
    # merging batting_df and opp_batting_df on Team, Year and League columns
    merge_one = pd.merge(batting_df, opp_batting_df, on=[
                         "Team", "Year", "League"], how="inner")

    # merging first merged df and standings_df on Team, Year and League columns
    merge_two = pd.merge(merge_one, standings_df, on=[
        "Team", "Year", "League"], how="inner")

    # finally mergining the second merged df with playoffs_df on Team and Year columns
    merged_full = pd.merge(merge_two, playoffs_df, on=[
                           "Team", "Year"], how="inner")

    # sorting the data by the year and creating a new df from it
    sorted_df = merged_full.sort_values(by=["Year"], ascending=False)

    # reading in the original Moneyball data
    orignal_df = pd.read_csv("baseball.csv")

    # concatenating the orignal data and the data from 2013-2025
    final_df = pd.concat([sorted_df, orignal_df], axis=0)

    # writing the complete dataset to a .csv file
    final_df.to_csv(DATA_PATH, index=False)

    print(final_df)


def clean_standings(df):
    df["G"] = df["W"] + df["L"]

    assign_abbreviation(df, "Tm")

    assign_league(df, "Team")

    standings_columns_to_drop = ['Tm', 'L', 'W-L%', 'GB']
    df.drop(columns=standings_columns_to_drop, inplace=True)

    df.to_csv(
        "./clean-data/cleaned-2013-2025-standings.csv", index=False)

    return df


def clean_batting(df):
    assign_abbreviation(df, "Tm")

    assign_league(df, "Team")

    batting_columns_to_drop = ['Tm', '#Bat', 'BatAge', 'R/G', 'G', 'PA', 'AB', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS',
                               'BB', 'SO', 'OPS', 'OPS+', 'TB', 'GDP', 'HBP', 'SH', 'SF', 'IBB', 'LOB']
    df.drop(columns=batting_columns_to_drop, inplace=True)

    df.rename(columns={'R': 'RS'}, inplace=True)

    df.to_csv(
        "./clean-data/cleaned-2013-2025-batting.csv", index=False)
    return df


def clean_opp_batting(df):
    assign_abbreviation(df, "Tm")

    assign_league(df, "Team")

    opp_columns_to_drop = ['Tm', 'RA/G', 'PAu', 'G', 'PA', 'AB', 'H', '2B', '3B', 'HR', 'SB', 'CS',
                           'BB', 'SO', 'BA', 'OPS', 'BAbip', 'TB', 'GDP', 'HBP', 'SH', 'SF', 'IBB', 'ROE']
    df.drop(columns=opp_columns_to_drop, inplace=True)

    df.rename(columns={'R': 'RA', 'OBP': 'OOBP', 'SLG': 'OSLG'}, inplace=True)

    df.to_csv(
        "./clean-data/cleaned-2013-2025-opp-batting.csv", index=False)

    return df


def assign_abbreviation(df, team_column_name):

    mlb_teams_map = {
        "Arizona Diamondbacks": "ARI",
        "Atlanta Braves": "ATL",
        "Baltimore Orioles": "BAL",
        "Boston Red Sox": "BOS",
        "Chicago Cubs": "CHC",
        "Chicago White Sox": "CHW",
        "Cincinnati Reds": "CIN",
        "Cleveland Guardians": "CLE",
        "Cleveland Indians": "CLE",
        "Colorado Rockies": "COL",
        "Detroit Tigers": "DET",
        "Houston Astros": "HOU",
        "Kansas City Royals": "KCR",
        "Los Angeles Angels": "LAA",
        "Los Angeles Angels of Anaheim": "LAA",
        "Los Angeles Dodgers": "LAD",
        "Miami Marlins": "MIA",
        "Milwaukee Brewers": "MIL",
        "Minnesota Twins": "MIN",
        "New York Mets": "NYM",
        "New York Yankees": "NYY",
        "Athletics": "OAK",
        "Oakland Athletics": "OAK",
        "Philadelphia Phillies": "PHI",
        "Pittsburgh Pirates": "PIT",
        "San Diego Padres": "SDP",
        "San Francisco Giants": "SFG",
        "Seattle Mariners": "SEA",
        "St. Louis Cardinals": "STL",
        "Tampa Bay Rays": "TBR",
        "Texas Rangers": "TEX",
        "Toronto Blue Jays": "TOR",
        "Washington Nationals": "WSN"
    }

    df["Team"] = df[team_column_name].map(mlb_teams_map)

    return df


def assign_league(df, team_column_name):

    mlb_teams_map = {
        "ARI": "NL",
        "ATL": "NL",
        "BAL": "AL",
        "BOS": "AL",
        "CHC": "NL",
        "CHW": "AL",
        "CIN": "NL",
        "CLE": "AL",
        "COL": "NL",
        "DET": "AL",
        "HOU": "AL",
        "KCR": "AL",
        "LAA": "AL",
        "LAD": "NL",
        "MIA": "NL",
        "MIL": "NL",
        "MIN": "AL",
        "NYM": "NL",
        "NYY": "AL",
        "OAK": "AL",
        "PHI": "NL",
        "PIT": "NL",
        "SDP": "NL",
        "SFG": "NL",
        "SEA": "AL",
        "STL": "NL",
        "TBR": "AL",
        "TEX": "AL",
        "TOR": "AL",
        "WSN": "NL"
    }

    df["League"] = df[team_column_name].map(mlb_teams_map)

    return df


if __name__ == "__main__":
    merge_datasets()
