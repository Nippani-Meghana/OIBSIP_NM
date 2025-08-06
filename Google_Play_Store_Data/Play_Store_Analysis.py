#ANALYZING GOOGLE PLAY STORE DATA

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

#Data loading
current_dir = os.getcwd()
file_path1 = os.path.join(current_dir, 'apps.csv')
apps = pd.read_csv(file_path1)
apps.drop(columns=['Unnamed: 0'], inplace=True)
file_path2 = os.path.join(current_dir, 'user_reviews.csv')
user_reviews = pd.read_csv(file_path2)

#Data Preview
def apps_data_preview():
    print("The preview of Apps dataset is as follows :")
    print(apps.head())
    print("The shape of Apps dataset :")
    print(apps.shape)
    print("The columns in Apps data set :")
    print(apps.columns)
    print("Missing Data :")
    missing_data = pd.isnull(apps['Rating'])
    print(apps[missing_data])


def user_reviews_preview():
    print("The preview of User Reviews dataset is as follows :")
    print(user_reviews.head())
    print("The shape of User Reviews dataset :")
    print(user_reviews.shape)
    print("The columns in Usser Reviews data set :")
    print(user_reviews.columns)
    missing_data = pd.isnull(user_reviews['Sentiment'])
    print(user_reviews[missing_data])
    merged_df = pd.merge(apps, user_reviews, on="App")
    print(merged_df)

#Data Cleaning 
apps['Price'] = apps['Price'].str.replace(',', '').str.replace('$','').astype(float)
apps['Installs'] = apps['Installs'].str.replace(',', '').str.replace('+','', regex = False).astype(float)
apps = apps.dropna(subset = ['Rating'])
user_reviews = user_reviews.dropna(subset=['Sentiment'])


#Category Exploration
def category_exploration():
    category_counts = apps['Category'].value_counts()
    plt.figure(figsize=(10, 8))
    category_counts.plot(kind='bar', color='#E3C6FF')
    plt.xlabel('Category of Apps')
    plt.ylabel('Number of Apps')
    plt.title('Distribution of Apps Across Categories')
    plt.tight_layout()
    plt.show()

#Metric Analysis
def plot_app_ratings():
    category_ratings = apps.groupby('Category')['Rating'].mean().sort_values()
    plt.figure(figsize=(10, 6))
    category_ratings.plot(kind='bar', color='skyblue')
    plt.xlabel("Category of Apps")
    plt.ylabel("Average Rating")
    plt.title("Average Rating in Each Category")
    plt.tight_layout()
    plt.show()

def plot_category_vs_installs():
    category_installs = apps.groupby('Category')['Installs'].mean().sort_values()
    plt.figure(figsize=(10, 6))
    category_installs.plot(kind='bar', color='#FFB7A5')
    plt.xlabel("Category")
    plt.ylabel("Average Number of Installs(1 x 10^7)")
    plt.title("Average Number of Installs per Category")
    plt.tight_layout()
    plt.show()

def plot_price_vs_installs():
    install_share = apps.groupby('Type')['Installs'].sum()
    plt.figure(figsize=(6, 6))
    install_share.plot(kind = 'bar',color = '#C1E1C1')
    plt.xlabel("Type : Free or Paid")
    plt.ylabel("Installs")
    plt.title('Share of Installs: Free vs Paid Apps')
    plt.show()

#Sentiment Analysis
def plot_type_vs_sentiment():
    merged_data = pd.merge(apps, user_reviews, on='App')
    count = merged_data.groupby(['Type','Sentiment']).size().unstack()
    count.plot(kind='bar', figsize=(8, 6), color=['#B5EAD7', '#FFB3BA', '#AEC6CF'])
    plt.xlabel('App Type : Free or Paid')
    plt.ylabel('Sentiment')
    plt.title('App Type vs Sentiment')
    plt.show()

def plot_sentiment_polarity_hist():
    sentiments = ['Positive', 'Neutral', 'Negative']
    colors = ['#B5EAD7', '#FFFFBA', '#FFB3BA']  

    plt.figure(figsize=(14, 6))
    for sentiment, color in zip(sentiments, colors):
        polarity_values = user_reviews[user_reviews['Sentiment'] == sentiment]['Sentiment_Polarity']
        plt.hist(polarity_values, bins=30, label=sentiment, color=color)

    plt.xlabel('Sentiment Polarity')
    plt.ylabel('Number of Reviews')
    plt.title('Distribution of Sentiment Polarity by Sentiment')
    plt.show()


while True: 
    print("Welcome to Google Play Store Data Analytics:")
    print("1. Apps Dataset Preview")
    print("2. User Reviews Dataset Preview")
    print("3. Category Exploration")
    print("4. Metric Analysis")
    print("5. Sentiment Analysis")
    print("6. Exit")
    choice = input('Enter any option(1-6) : ')

    if choice == '1':
        apps_data_preview()
    elif choice == '2':
        user_reviews_preview()
    elif choice == '3':
        category_exploration()
    elif choice == '4':
        while True :
            print("Metric Analysis Menu : ")
            print("1. Plot for Average Rating in Each Category")
            print("2. Plot for Average Number of Installs per Category")
            print("3. Plot for Share of Installs: Free vs Paid Apps")
            print("4. Exit to Main Menu")
            ch = input("Enter your choice please :")
            if ch == '1':
                plot_app_ratings()
            elif ch == '2':
                plot_category_vs_installs()
            elif ch == '3':
                plot_price_vs_installs()
            elif ch == '4':
                print("Thank You! You are now being forwaded to the Main Menu")
                break
            else:
                print("Invalid Choice")
    elif choice == '5':
        while True:
            print("1. Plot for App Type vs Sentiment")
            print("2. Plot for Distribution of Sentiment Polarity by Sentiment")
            print("3. Exit to Main Menu")
            ch = input("Enter your choice please :")
            if ch == '1':
                plot_type_vs_sentiment()
            elif ch == '2':
                plot_sentiment_polarity_hist()
            elif ch == '3':
                print("Thank You! You are now being forwaded to the Main Menu")
                break
            else:
                print("Invalid Choice")
    elif choice == '6':
        print("Program Terminated Successfully. Have a great day!")
        break
    else:
        print("Invalid Choice!")













