#CLEANING DATA

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
import pandas as pd

DATA_DIR = Path(r"c:\Users\sasha\Desktop\Comp Neuron\Cleaning_Data")

def load_if_exists(filename):
    fp = DATA_DIR / filename
    if fp.exists():
        print(f"Loaded: {fp}")
        return pd.read_csv(fp), str(fp)
    else:
        print(f"Missing (skipping): {fp}")
        return None, str(fp)

airbnb, file_path  = load_if_exists("AB_NYC_2019.csv")
ca_videos, file_path2 = load_if_exists("CAvideos.csv")
gb_videos, file_path3 = load_if_exists("GBvideos.csv")
us_videos, file_path4 = load_if_exists("USvideos.csv")


#1. Data Preview

#Data Preview of Airbnb Dataset
def preview_airbnb():
    print("Preview of Airbnb Dataset")
    print(airbnb.head())
    print("The columns are : ", airbnb.columns)
    print("The shape of the dataset is : ",airbnb.shape)

#Data Preview of Canada youtube Dataset
def preview_ca_videos():
    print("Preview of Canada Youtube Dataset")
    print(ca_videos.head())
    print("The columns are : ", ca_videos.columns)
    print("The shape of the dataset is : ",ca_videos.shape)

#Data Preview of Great Britain youtube Dataset
def preview_gb_videos():
    print("Preview of Great Britain Youtube Dataset")
    print(gb_videos.head())
    print("The columns are : ", gb_videos.columns)
    print("The shape of the dataset is : ",gb_videos.shape)

#Data Preview of USA youtube Dataset
def preview_us_videos():
    print("Preview of USA Youtube Dataset")
    print(us_videos.head())
    print("The columns are : ", us_videos.columns)
    print("The shape of the dataset is : ",us_videos.shape)


#2. Data Auditing

#Data Auditing of Airbnb Dataset
def data_audit_airbnb():
    print("The Number of Missing Values (Total) : ",airbnb.isna().sum().sum())
    print("The Number of Missing Values in a column : ")
    print(airbnb.isna().sum())
    print("The Number of Duplicate Values are : ",airbnb.duplicated().sum())

#Data Auditing of CA Videos Dataset
def data_audit_ca_videos():
    print("The Number of Missing Values (Total) : ",ca_videos.isna().sum().sum())
    print("The Number of Missing Values in a column : ")
    print(ca_videos.isna().sum())
    print("The Number of Duplicate Values are : ",ca_videos.duplicated().sum())

#Data Auditing of GB Videos Dataset
def data_audit_gb_videos():
    print("The Number of Missing Values (Total) : ",gb_videos.isna().sum().sum())
    print("The Number of Missing Values in a column : ")
    print(gb_videos.isna().sum())
    print("The Number of Duplicate Values are : ",gb_videos.duplicated().sum())

#Data Auditing of the USA Videos Dataset
def data_audit_us_videos():
    print("The Number of Missing Values (Total) : ",us_videos.isna().sum().sum())
    print("The Number of Missing Values in a column : ")
    print(us_videos.isna().sum())
    print("The Number of Duplicate Values are : ",us_videos.duplicated().sum())


#3. Missing Data

#Missing Data of Airbnb dataset
def missing_data_airbnb():
    print("The Missing Data from Airbnb Dataset : ")
    print(airbnb.isnull())
    print("The Missing Values are replaced by 0/0.0 ")
    airbnb['number_of_reviews'] = airbnb["number_of_reviews"].fillna(0)
    airbnb['reviews_per_month'] = airbnb["reviews_per_month"].fillna(0.0)
    airbnb['last_review'] = airbnb["last_review"].fillna(pd.NaT)
    print(airbnb.head())
    return airbnb

#Mising Data of CA Videos Dataset
def missing_data_ca_videos():
    print("The Missing Data from Canada youtube Dataset : ")
    print(ca_videos.isnull())
    print("The Missing Values are replaced by Unknown ")
    ca_videos['description'] = ca_videos['description'].fillna('Unknown')
    print(ca_videos.head())
    return ca_videos

#Mising Data of GB Videos Dataset
def missing_data_gb_videos():
    print("The Missing Data from Great Britain youtube Dataset : ")
    print(gb_videos.isnull())
    print("The Missing Values are replaced by Unknown ")
    gb_videos['description'] = gb_videos['description'].fillna('Unknown')
    print(gb_videos.head())
    return gb_videos

#Mising Data of US Videos Dataset
def missing_data_us_videos():
    print("The Missing Data from the USA youtube Dataset : ")
    print(us_videos.isnull())
    print("The Missing Values are replaced by Unknown ")
    us_videos['description'] = us_videos['description'].fillna('Unknown')
    print(us_videos.head())
    return us_videos


#4. Data Standardization and Data Integrity

#Standardization of Airbnb Dataset
def airbnb_standardization():
    airbnb["last_review"] = pd.to_datetime(airbnb["last_review"], errors="coerce")
    airbnb["last_review"] = airbnb["last_review"].dt.strftime("%Y-%m-%d")
    airbnb['availability_365_flag'] = airbnb["availability_365"].between(0,365)
    airbnb['price_flag_non_positive'] = airbnb['price']<=0
    airbnb["number_of_reviews_flag_negative"] = airbnb["number_of_reviews"]<0
    print(airbnb.head())

#Standardization of CA Videos Dataset
def ca_videos_standardization():
    ca_videos['views_flag'] = ca_videos['views']<0
    ca_videos['likes_flag'] = ca_videos['likes']<0
    ca_videos['dislikes_flag'] = ca_videos['dislikes']<0
    ca_videos['comment_count_flag'] = ca_videos['comment_count']<0
    ca_videos['flag_reactions_exceed_views'] = (ca_videos['likes'] + ca_videos['dislikes']) > ca_videos['views']
    print(ca_videos.head())

#Standardization of GB Videos Dataset
def gb_videos_standardization():
    gb_videos['views_flag'] = gb_videos['views']<0
    gb_videos['likes_flag'] = gb_videos['likes']<0
    gb_videos['dislikes_flag'] = gb_videos['dislikes']<0
    gb_videos['comment_count_flag'] = gb_videos['comment_count']<=0
    print(gb_videos.head())

#Standardization of the USA Videos Dataset
def usa_videos_standardization():
    us_videos['views_flag'] = us_videos['views']<0
    us_videos['likes_flag'] = us_videos['likes']<0
    us_videos['dislikes_flag'] = us_videos['dislikes']<0
    us_videos['comment_count_flag'] = us_videos['comment_count']<0
    print(us_videos.head())

#5. Outlier Detection And Handling

def iqr_flags(s, k=1.5):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return pd.Series(False, index=s.index), (np.nan, np.nan, np.nan)
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    low = q1 - k*iqr
    high = q3 + k*iqr
    return None, (q1, q3, high)  

def outlier_handling_airbnb():
    s = pd.to_numeric(airbnb["price"], errors="coerce").dropna()
    _, (q1, q3, high) = iqr_flags(s, k=1.5)

    plt.figure()
    plt.hist(s, bins=50)
    plt.axvline(q1)
    plt.axvline(q3)
    plt.axvline(high)
    plt.title("Airbnb Price — IQR fences (Q1, Q3, High)")
    plt.xlabel("Price"); plt.ylabel("Count")
    plt.show()


#MENU
while True:
    print("Welcome to the Menu : ")
    print("1. Airbnb Dataset")
    print("2. Canada Youtube Dataset")
    print("3. Great Britain Youtube Dataset")
    print("4. USA Youtube Dataset")
    print("5. Exit")
    choice = input("Please Select Your Choice (1-5) : ")
    if choice == '1':
        while True:
            print("Airbnb Menu : ")
            print("1. Data Preview of Airbnb Dataset")
            print("2. Data Auditing of Airbnb Dataset")
            print("3. Missing Data of Airbnb dataset")
            print("4. Standardization of Airbnb Dataset")
            print("5. Outlier Detection And Handling")
            print("6. Return to Main Menu")
            ch = input("Please choose from option (1-6) : ")
            if ch == '1':
                preview_airbnb()
            elif ch == '2':
                data_audit_airbnb()
            elif ch == '3':
                missing_data_airbnb()
            elif ch == '4':
                airbnb_standardization()
            elif ch == '5':
                outlier_handling_airbnb()
            elif ch == '6':
                break
            else:
                print("Invalid Choice!")
    
    elif choice == '2':
        while True:
            print("Canada Youtube Menu : ")
            print("1. Data Preview of Canada Youtube Dataset")
            print("2. Data Auditing of Canada Youtube Dataset")
            print("3. Missing Data of Canada Youtube dataset")
            print("4. Standardization of Canada Youtube Dataset")
            print("5. Return to Main Menu")
            ch = input("Please choose from option (1-5) : ")
            if ch == '1':
                preview_ca_videos()
            elif ch == '2':
                data_audit_ca_videos()
            elif ch == '3':
                missing_data_ca_videos()
            elif ch == '4':
                ca_videos_standardization()
            elif ch == '5':
                break
            else:
                print("Invalid Choice!")

    elif choice == '3':
        while True:
            print("Great Britain Youtube Menu : ")
            print("1. Data Preview of Great Britain Youtube Dataset")
            print("2. Data Auditing of Great Britain Youtube Dataset")
            print("3. Missing Data of Great Britain Youtube dataset")
            print("4. Standardization of Great Britain Youtube Dataset")
            print("5. Return to Main Menu")
            ch = input("Please choose from option (1-5) : ")
            if ch == '1':
                preview_gb_videos()
            elif ch == '2':
                data_audit_gb_videos()
            elif ch == '3':
                missing_data_gb_videos()
            elif ch == '4':
                gb_videos_standardization()
            elif ch == '5':
                break
            else:
                print("Invalid Choice!")

    elif choice == '4':
        while True:
            print("USA Youtube Menu : ")
            print("1. Data Preview of USA Youtube Dataset")
            print("2. Data Auditing of USA Youtube Dataset")
            print("3. Missing Data of USA Youtube dataset")
            print("4. Standardization of USA Youtube Dataset")
            print("5. Return to Main Menu")
            ch = input("Please choose from option (1-5) : ")
            if ch == '1':
                preview_us_videos()
            elif ch == '2':
                data_audit_us_videos()
            elif ch == '3':
                missing_data_us_videos()
            elif ch == '4':
                usa_videos_standardization()
            elif ch == '5':
                break
            else:
                print("Invalid Choice!")

    elif choice == '5':
        print("Program Terminated! Have a Great Day Ahead!")
        break

    else:
        print("Invalid Choice!")

