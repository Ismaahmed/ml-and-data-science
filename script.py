import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import  RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# load data
CSV_PATH = "Dataset\clean_house_l5_dataset.csv"
df = pd.read_cvs(CSV_PATH)

x = df.drop(columns=["Price" , "LogPrice"])
y = df["Price"]
