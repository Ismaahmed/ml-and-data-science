import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import  RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# load data
CSV_PATH = "./Dataset/clean_house_l5_dataset.csv"
df = pd.read_csv(CSV_PATH)

x = df.drop(columns=["Price" , "LogPrice"])
y = df["Price"]
x_train, x_test, y_train, y_test =  train_test_split (x,y, test_size=0.2, random_state=42)
lr = LinearRegression()
lr.fit(x_train, y_train)
lr_pred = lr.predict(x_test)
# print(lr_pred[:10])

rf = RandomForestRegressor(n_estimators=100, random_state=42)

rf.fit(x_train, y_train)
rf_pred = rf.predict(x_test)

def print_metrics(name, y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mean = mean_absolute_error(y_true, y_pred)
    mean_squar = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mean_squar)
    print(f"\n{name} performance")
    print(f"  R2  : {r2: .3f}")
    print(f"  mean  : {mean: ,.0f}")
    print(f"  mean_squar  : {mean_squar: ,.0f}")
    print(f"  RMSE  : {rmse: ,.0f}")


print_metrics("linear Regression", y_test, lr_pred)
print_metrics("Random forest Regression", y_test, rf_pred)

i = 0
x_one_df = x_test.iloc[[i]]
y_one_df = y_test.iloc[i]
p_lr_one = float(lr.predict(x_one_df) [i])
p_rf_one = float(rf.predict(x_one_df) [i])

print("single_row sanity chech:")
print(f"Actual price ${y_one_df: ,.0f}")
print(f"Actual price: ${y_one_df: ,.0f}")
print(f"LR PRED: ${p_lr_one: ,.0f}")
print(f"RF PRED: ${p_rf_one: ,.0f}")
