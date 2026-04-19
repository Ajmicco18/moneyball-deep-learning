# Moneyball Deep Learning Analysis Project

In this repository is the code I used to investigate how well I could predict the number of wins a Major League Baseball (MLB) team would win and classify whether a team would make the MLB playoffs using statistical team data from 1962-2025. Data includes runs allowed, runs scored, team batting average, and opponenent on base percentage, among others. To predict the wins, I used a linear regression as my baseline regression model and compared its performance with a base, 3-layer neural network and a wide and deep neural network. Then to classify a team as a playoff team or not, I used a decision tree classifier as my baseline model and compared it to a base, 3-layer neural network and a wide and deep neural network. 

I used a modular approach to complete this and separated my code into directories that contained specific elements that I needed to complete the project. Below is an explanation of each directory and the specified purpose of each. 

## configs 

The [configs directory](https://github.com/Ajmicco18/moneyball-deep-learning/tree/main/configs)  contains `config.py` file which contains the file paths to each directory in the project to ensure it is easier to access and save information to a specific directory. 

## data-retrieval 

The [data-retrieval directory](https://github.com/Ajmicco18/moneyball-deep-learning/tree/main/data-retrieval) contains multiple files used in scraping the data from 2013-2025 from multiple tables on the Baseball Reference website. The `data-retrieval.py` file is the script I used to actually scrap all the data I needed and store them in their specific .csv files before I extracted my needed infomation and concatenated all the required data in the `data_cleaning.py` file. The final, cleaned dataset used for the project is saved to the **data** directory to avoid any confusion with the datasets in this directory.  

## data

The [data directory](https://github.com/Ajmicco18/moneyball-deep-learning/tree/main/data) contains the complete, cleaned dataset of team statistics from 1962-2025 used throughout the project. 

## plots

The [plots directory](https://github.com/Ajmicco18/moneyball-deep-learning/tree/main/plots) contains multiple plots illustrating the performance of all the classification and regression models I utilized.

## src

The [src directory](https://github.com/Ajmicco18/moneyball-deep-learning/tree/main/src) contains two subdirectories containing the class definitions of all the models I created. It also contains my data preprocessing functions, as well as the functions used to train my models. 

## `main.py`

[`main.py`](https://github.com/Ajmicco18/moneyball-deep-learning/blob/main/main.py) is the script used to run all the models. It contains functions that create instances of the model classes, define all the hyperparameters for each of the models, train the models, generate the models' plots and returns the training and evaluation metrics from the models. To run each model, you simply uncomment the `print(model_function_call())` of the specific model you want to run. You can uncomment all the models at once, but it is easier to see the results by running the models one at a time. The code below the model functions is used to generate evaluation comparison graphs, which are not required to run and simply used for evaluation and comparison of the models. 