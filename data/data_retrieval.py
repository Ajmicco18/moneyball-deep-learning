import pandas as pd
import time

# Websites to use for data scraping
opp_stats = "https://www.baseball-reference.com/leagues/majors/ -batting-pitching.shtml"

team_stats = "https://www.baseball-reference.com/leagues/majors/ -standard-batting.shtml"

team_standings = "https://www.baseball-reference.com/leagues/majors/ -standings.shtml"

# years that need data scraped for
years = ['2013', '2014', '2015', '2016', '2017', '2018',
         '2019', '2020', '2021', '2022', '2023', '2024', '2025']

# initializing dataframes
opp_batting_df = pd.DataFrame()
batting_df = pd.DataFrame()
standings_df = pd.DataFrame()

# function to scrape data


def scrape_data(url, data_frame, csv_name):
    """
    Args:
        url (string): The url of the page that is being scrapped
        data_frame (DataFrame): The name of the dataframe the data is being added to 
        csv_name (string): The name of the csv file the dataframe will write to
    """

    # iterate through every year in the years list
    for year in years:
        # reading the html of the url we pass in and replacing the space with the year at the current index
        tables = pd.read_html(url.replace(' ', year))

        # finding the first table tag in the html
        df = tables[0]

        # setting a year column to the year at the current index
        df["Year"] = year

        # printing the shape of the data
        print(df.shape)

        # concatenating the overall dataframe w/ the year index's data to ensure all data from 2013-2025 is in one dataframe
        data_frame = pd.concat([data_frame, df], axis=0)

        # setting a 10 second pause between requests
        time.sleep(10)

    # writing the data to a csv file
    data_frame.to_csv(f"{csv_name}.csv", index=False)


def scrape_standings_data(url, data_frame, csv_name):

    for year in years:

        tables = pd.read_html(url.replace(' ', year))

        df = pd.concat([tables[0], tables[1], tables[2],
                       tables[3], tables[4], tables[5]], axis=0)

        df["Year"] = year

        print(df.shape)

        data_frame = pd.concat([data_frame, df], axis=0)

        time.sleep(10)

    data_frame.to_csv(f"{csv_name}.csv", index=False)


if __name__ == "__main__":
    # load_data()
    scrape_data(opp_stats, opp_batting_df, "2013-2025-opp-batting")
    scrape_data(team_stats, batting_df, "2013-2025-batting")
    scrape_standings_data(team_standings, standings_df, "2013-2025-standings")
