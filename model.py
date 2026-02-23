import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix)
RANDOM_STATE = 42 
df = pd.read_csv("mail_l7_dataset.csv")
# print(df.head())

df["Category"] = df["Category"].astype(object)

df.loc[df["Category"].str.lower().str.strip() == "spam", "Category"] = 0
df.loc[df["Category"].str.lower().str.strip() == "ham", "Category"] = 1


# print(df.head())
X = df["Message"].astype(str)
y = df["Category"].astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
tfidf = TfidfVectorizer(min_df=1, stop_words="english", lowercase=True)
X_train_feature = tfidf.fit_transform(X_train)
X_test_feature = tfidf.transform(X_test)
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_feature, y_train)
lr_predict = lr.predict(X_test_feature)

rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
rf.fit(X_train_feature, y_train)
rf_predict = rf.predict(X_test_feature.toarray())


Np = MultinomialNB()
Np.fit(X_train_feature, y_train)
Np_predict = Np.predict(X_test_feature)

# model = MultinomialNB()

# # Train (bar model-ka)
# model.fit(X_train_tfidf, y_train)

# # Saadaali
# y_pred = model.predict(X_test_tfidf)


def print_metrics(name, y_true, y_pred, pos_label=0):
    acc = accuracy_score (y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=pos_label)
    rec = recall_score(y_true, y_pred, pos_label=pos_label)
    f1 = f1_score(y_true, y_pred, pos_label=pos_label)

    print(f"{name} performance: ")
    print(f"accuracy: {acc:.3f} ")
    print(f"precision: {prec:.3f} (positive : spam = 0) ")
    print(f"recall: {rec:.3f}  (positive : spam = 0)")
    print(f"f1-score: {f1:.3f} (positive : spam = 0)")
def print_confmat(name, y_true, y_pred, ):
    cm = confusion_matrix(y_true, y_pred, labels=[1,0])
    cm_df = pd.DataFrame(
        cm,
        index = ["Actual Ham (1)", "Actual Spam (0)"],
        columns = ["pred Ham (1)", "pred Spam (0)"]
    )
    print(F"{name} - Comfusion matrix: \n{cm_df}")

print_metrics("Logistic Regression:", y_test, lr_predict)

print_confmat("Logistic Regression:", y_test, lr_predict)
print_metrics("Random Forest Classifier:", y_test, rf_predict)
print_confmat("Random Forest Classifier:", y_test, rf_predict)
print_metrics("Naive Bayes (MultinomialNB):", y_test, Np_predict)
print_confmat("Naive Bayes (MultinomialNB):", y_test, Np_predict)



test = [
    "Free entry in 2 a weekly competition!",
    "I will meet you at the cafe tomorrow",
    "Congratulations, you won a free ticket"

]
vec = tfidf.transform(test)
print("Logistic Regression:", lr.predict(vec))
print("Random Forest Classifier:", rf.predict(vec.toarray()))
print("Naive Bayes (MultinomialNB):", Np.predict(vec))
    





