#WINE QUALITY PREDICTION

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import svm

BASE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
CSV_PATH = BASE / "WineQT.csv"

if not CSV_PATH.exists():
    hits = list(BASE.rglob("WineQT*.csv"))
    if not hits:
        raise FileNotFoundError(f"Couldn't find WineQT.csv near {BASE}.")
    CSV_PATH = hits[0]  

print("Reading:", CSV_PATH)
dataset = pd.read_csv(CSV_PATH)

#Data Exploration and Some Basic Information
def data_exploration():
    print("The Columns in the dataset are :")
    c = dataset.columns
    for _ in c:
        print(_)
    print("The Shape of the dataset is : ",dataset.shape)
    print("The first few values in the dataset are : ")
    print(dataset.head())

X = dataset[['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide',
                 'total sulfur dioxide','density','pH','sulphates','alcohol']]
y = dataset['quality']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=0)
print("Class distribution (train):")
print(y_train.value_counts(normalize=True).rename({0:"Bad",1:"Good"}).round(3))
print("\nClass distribution (test):")
print(y_test.value_counts(normalize=True).rename({0:"Bad",1:"Good"}).round(3))  

def random_forest():
    rf = RandomForestClassifier(
        n_estimators=300,
        criterion='gini',
        class_weight='balanced',
        n_jobs=-1,
        random_state=0,
        oob_score=True
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]
    print("\nRandom Forest (Binary) : ")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")

random_forest()

def stochastic_gradient_descent():
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    sgd = SGDClassifier( loss="log_loss", penalty="l2", alpha=1e-6, class_weight="balanced", 
                        max_iter=10000, early_stopping=True, validation_fraction=0.1, 
                        n_iter_no_change=10, random_state= 0 ) 
    sgd.fit(X_train_s, y_train) 
    sgd_pred = sgd.predict(X_test_s) 
    sgd_proba = sgd.predict_proba(X_test_s)[:, 1] 
    print("\nSGD (Logistic) Binary : ") 
    print(f"Accuracy: {accuracy_score(y_test, sgd_pred):.3f}")

stochastic_gradient_descent()


def support_vector_machine():
    clf = svm.SVC(kernel='linear') 
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("\nSVC (Linear) : ")
    print("Accuracy: ",accuracy_score(y_test, y_pred))

support_vector_machine()


#Data Analysis Visuals
def heatmap():
    plt.figure(figsize=(10,8))
    corr = dataset.corr()
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap of Wine Features", fontsize=14)
    plt.show()

heatmap()


