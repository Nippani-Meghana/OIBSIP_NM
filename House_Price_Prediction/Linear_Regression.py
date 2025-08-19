#Predicting House Prices With Linear Regression

import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from pathlib import Path

BASE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
CSV_PATH = BASE / "Housing.csv"

if not CSV_PATH.exists():
    # fallback: search nearby
    hits = list(BASE.rglob("Housing*.csv"))
    if not hits:
        raise FileNotFoundError(f"Couldn't find Housing.csv near {BASE}.")
    CSV_PATH = hits[0]  # pick the first match (or choose interactively)

print("Reading:", CSV_PATH)
dataset = pd.read_csv(CSV_PATH)


#Dataset Exploration and Some Basic Information
def data_exploration():
    print("All the the columns in the dataset are :")
    print(dataset.columns)
    print("The Shape of the dataset is :")
    print(dataset.shape)
    print("The first 5 Rows the dataset are :")
    print(dataset.head())

data_exploration()
#Data Preparation 
dataset['mainroad_value'] = dataset['mainroad'].str.lower().map({'yes':1,'no':0})

dataset['guestroom_value'] = dataset['guestroom'].str.lower().map({'yes':1,'no':0})

dataset['basement_value'] = dataset['basement'].str.lower().map({'yes':1,'no':0})

dataset['hotwater_value'] = dataset['hotwaterheating'].str.lower().map({'yes':1,'no':0})

dataset['airconditioning_value']= dataset['airconditioning'].str.lower().map({'yes':1,'no':0})

dataset['prefarea_value'] = dataset['prefarea'].str.lower().map({'yes':1,'no':0})

fs = dataset['furnishingstatus'].str.lower()
dataset['fs_semi']     = (fs == 'semi-furnished').astype(int)
dataset['fs_furnished']= (fs == 'furnished').astype(int)

dataset.drop(['mainroad','guestroom', 'basement','hotwaterheating','airconditioning','prefarea','furnishingstatus'],axis = 1, inplace=True)

print("Modified Dataset is : ")
print(dataset.head())

def predict_house_price_and_visualization():
    y = dataset['price']
    X = dataset.drop(columns=['price'])
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.05, random_state=0)
    reg = LinearRegression()
    reg.fit(X_train,y_train)
    train_score = reg.score(X_train,y_train)
    print("The training score of model is: ", round(train_score*100,4))
    test_score = reg.score(X_test, y_test)
    print("The score of the model on test data is:", round(test_score*100,4))
    y_pred = reg.predict(X_test)
    plt.figure()
    plt.scatter(y_test, y_pred)
    lo = float(min(np.min(y_test), np.min(y_pred)))
    hi = float(max(np.max(y_test), np.max(y_pred)))
    plt.plot([lo, hi], [lo, hi], linewidth=2)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(f"Predicted vs Actual (Test) - R^2 = {test_score:.3f}")
    plt.tight_layout()
    plt.show()

predict_house_price_and_visualization()
