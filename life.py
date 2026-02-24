import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# load data
CSV_PATH = "./Lifestyle_Sleep_clean.csv"
df = pd.read_csv(CSV_PATH)


X = df.drop(columns=["How many hours did you sleep last night?"])
y = df["How many hours did you sleep last night?"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)


rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
# 
# 



def print_matrics(name, y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mean = mean_absolute_error(y_true, y_pred)
    mean_squar = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mean_squar)
    print(f"\n{name} performance")
    print(f" R2 : {r2: .4f}")
    print(f"mean : {mean: .4f}")
    print(f"mean_squar : {mean_squar:.4f}" )
    print(F"RMSE : {rmse: .4f}")


print_matrics("Linear Regression", y_test, lr_pred)
print_matrics("Random forest Regression", y_test, rf_pred)

# print_metrics("linear Regression", y_test, lr_pred)
# print_metrics("Random forest Regression", y_test, rf_pred)



i = 0
x_one_df = X_test.iloc[[i]]
y_one_df = y_test.iloc[i]
p_lr_one = float(lr.predict(x_one_df) [0])
p_rf_one = float(rf.predict(x_one_df) [0])

 
print("Sinle_Row sanity check:")
print(f"actual Hours {y_one_df: } Hours")
print(f"actual Hours : {y_one_df} Hours")
print(f"LR PRED:  {p_lr_one:.2f} Hours")
print(f"RF RED: {p_rf_one: .2f} Hours")



