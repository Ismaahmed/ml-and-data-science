import pandas as pd
CSV_PATH = './Lifestyle_Sleep_Dataset.csv'
df = pd.read_csv(CSV_PATH)

df.columns = df.columns.str.strip()
df = df.drop('Timestamp', axis=1)

# print(df.head(10))
# print(df.shape )


# print(df.info())
# print(df.isnull().sum())

# df["Daily Screen Time (Hours) "] = df["Daily Screen Time (Hours) "].astype(int)
# print(df[' Primary Social Media App '].head(10))


df["Primary Social Media App"] = df["Primary Social Media App"].replace({"youtube": "YouTube"}).str.strip()
# print(df['Primary Social Media App'].value_counts(dropna=False))

df["How many days do you exercise per week?"] = df["How many days do you exercise per week?"].fillna(df["How many days do you exercise per week?"].mode()[0])
df["Daily Screen Time (Hours)"] = df["Daily Screen Time (Hours)"].fillna(df["Daily Screen Time (Hours)"].mode()[0]).astype(int)
df["Age"] = df["Age"].fillna(df["Age"].mode()[0]).astype(int)
before = df.shape
df = df.drop_duplicates()
after = df.shape
# print("Before:", before, "after: ", after)

def iqr_fun(series, k=1.5):
    q1, q3 = series.quantile([0.25,0.75])
    iqr = q3 - q1 
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return lower, upper


low_age, high_age = iqr_fun(df["Age"])
low_daily, high_daily = iqr_fun(df["Daily Screen Time (Hours)"])
low_Excercise, high_Excercise = iqr_fun(df["How many days do you exercise per week?"])

df["Age"] = df["Age"].clip(lower=low_age, upper=high_age).astype(int)
df["Daily Screen Time (Hours)"] = df["Daily Screen Time (Hours)"].clip(lower=low_daily, upper=high_daily)

df["How many days do you exercise per week?"] = df["How many days do you exercise per week?"].clip(lower=low_Excercise, upper=high_Excercise)

# print("after clipping")
# print(df["Daily Screen Time (Hours)"].describe())


# print("IQR of Daily Screen Time (Hours):")
# print("low_daily:", low_daily, "high_daily:", high_daily)
# print("IQR of how How many hours did you sleep last night?:")
# print("low_sleep:", low_sleep, "high_sleep:", high_sleep)


exercise_per_week = {
    "0":0,
    "1-2 days":1.5,
     "3-4 days":3.5,
     "5+ days": 5

}
df["How many days do you exercise per week?"] = df["How many days do you exercise per week?"].map(exercise_per_week)
# print(df["How many days do you exercise per week?"].value_counts())

df = pd.get_dummies(df, columns=["Primary Social Media App"], drop_first=False)
print("One hot encoding Primary Social Media App:")
# print([c for c in df.columns if c.startswith("Primary Social Media App")])
# print(df.head(10))







