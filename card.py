import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load data
CSV_PATH = "./car_l3_clean_ready.csv"
df = pd.read_csv(CSV_PATH)

# 2. Features and Target
X = df.drop(columns=['Unnamed: 0', 'Price', 'LogPrice', 'Car_year', 'Is_Rural'])
y = df['LogPrice'] 

# SAXID: X (weyn) halkii ay ka ahayd x (yar)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Models
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# 4. Metrics Function
def print_metrics(name, y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    
    # SAXID: Dib ugu celi qiimaha rasmiga ah (exp) si MAE/RMSE u noqdaan lacag dhab ah
    y_true_actual = np.exp(y_true)
    y_pred_actual = np.exp(y_pred)
    
    mae = mean_absolute_error(y_true_actual, y_pred_actual)
    rmse = np.sqrt(mean_squared_error(y_true_actual, y_pred_actual))
    
    print(f"\n{name} performance")
    print(f"  R2 Score : {r2:.3f}")
    print(f"  MAE      : ${mae:,.2f}")
    print(f"  RMSE     : ${rmse:,.2f}")

print_metrics("Linear Regression", y_test, lr_pred)
print_metrics("Random Forest Regression", y_test, rf_pred)

# 5. Sanity Check
i = 0
x_one_df = X_test.iloc[[i]]
y_one_log = y_test.iloc[i]

# SAXID: Isticmaal [0] markaad saadaalinayso hal saf
p_lr_one_log = lr.predict(x_one_df)[0]
p_rf_one_log = rf.predict(x_one_df)[0]

print("\n--- Single Row Sanity Check (Actual Prices) ---")
print(f"Actual Price:   ${np.exp(y_one_log):,.2f}")
print(f"LR Prediction:  ${np.exp(p_lr_one_log):,.2f}")
print(f"RF Prediction:  ${np.exp(p_rf_one_log):,.2f}")