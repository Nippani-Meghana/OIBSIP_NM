#EXPLORATORY DATA ANALYSIS (EDA) ON RETAIL SALES DATA

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os
current_dir = os.getcwd()
file_path = os.path.join(current_dir, 'retail_sales_dataset.csv')
df = pd.read_csv(file_path)
file_path2 = os.path.join(current_dir,'menu.csv')
menu  = pd.read_csv(file_path2)


#Dataset Exploration and Some Basic Information
def data_loading_and_cleaning(d,m):
    print("All the columns in the Retails Sales Dataset are :")
    print(df.columns)
    print("The shape of the Retail Sales Dataset are :")
    print(df.shape)
    print("The first 5 Rows of the Retails Sales Dataset are :")
    print(df.head())
    print("All the columns in the Menu Dataset are :")
    print(menu.columns)
    print("The shape of the Menu Dataset are :")
    print(menu.shape)
    print("The first 5 Rows of the Menu Dataset are :")
    print(menu.head())


#Descriptive Statistics of Retails Sales Dataset
def retail_sales_statistics(df):
    columns = ['Characteristics','Mean','Mode','Median','Standard Deviation']
    rows = ['Age','Quantity','Price per Unit','Total Amount']
    table = []
    for _ in rows:
        mean = np.mean(df[_])
        median = np.median(df[_])
        mode = stats.mode(df[_], keepdims=True)[0][0]
        standard_devn = np.std(df[_])
        table.append([_,float(mean),float(median),float(mode),float(standard_devn)])
    final_table = pd.DataFrame(table, columns = columns)
    print(final_table)   


#Descriptive Statistics of Menu Dataset
def menu_statistics(menu):
    columns = ['Characteristics','Mean','Mode','Median','Standard Deviation']
    rows = ['Calories','Calories from Fat','Total Fat','Total Fat (% Daily Value)','Saturated Fat','Saturated Fat (% Daily Value)',
            'Trans Fat','Cholesterol','Cholesterol (% Daily Value)','Sodium','Sodium (% Daily Value)','Carbohydrates','Carbohydrates (% Daily Value)',
            'Dietary Fiber','Dietary Fiber (% Daily Value)','Sugars','Protein','Vitamin A (% Daily Value)','Vitamin C (% Daily Value)','Calcium (% Daily Value)',
            'Iron (% Daily Value)']
    table = []
    for _ in rows:
        mean = np.mean(menu[_])
        median = np.median(menu[_])
        mode = stats.mode(menu[_], keepdims=True)[0][0]
        standard_devn = np.std(menu[_])
        table.append([_,float(mean),float(median),float(mode),float(standard_devn)])
    final_table = pd.DataFrame(table, columns = columns)
    print(final_table)


#Time Series Analysis
def plot_weekly_sales_analysis(df):
    #Sort the Data by Date
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by = 'Date')
    df = df.set_index('Date')
    weekly_sales = df['Total Amount'].resample('W').sum()
    plt.figure(figsize=(10, 6))
    plt.plot(weekly_sales)
    plt.xlabel('Date(Weekly)')
    plt.ylabel('Total Amount')
    plt.title('Weekly Sales Analysis')
    plt.show()


#Customer and Product Analysis
def plot_sales_by_gender(df):
    group = df.groupby(['Gender','Product Category'])['Total Amount'].mean().unstack()
    group.plot(kind = 'bar', figsize=(14, 6), color = ['cyan','deeppink','green'])
    plt.xlabel('Gender')
    plt.ylabel('Average money spent on each Category')
    plt.title('Average Money spent on each Product Category by each Gender')
    plt.show()


def plot_age_by_product_category(df):
    df['Age Group'] = pd.cut(df['Age'], bins=[0, 30, 45, 60, 100], 
                         labels=["15-30", "31-45", "46-60", ">60"])
    group = df.groupby(['Age Group', 'Product Category'])['Total Amount'].mean().unstack()
    group.plot(kind = 'bar', figsize=(14, 6), color = ['cyan','deeppink','green'])
    plt.xlabel('Ages')
    plt.ylabel('Average money spent on each Category')
    plt.title('Average Money spent on each Product Category by each Age Group')
    plt.show()


def print_recommendations():
    recommendations = [
        "1. Target Younger Age Groups for Clothing Sales:\n"
        "   Customers aged 15-30 spend the most on clothing (₹553.07 per transaction).\n"
        "   → Launch seasonal fashion campaigns and student discounts for this segment.\n",

        "2. Promote Electronics to Older Demographics:\n"
        "   Customers above 60 spend the most on electronics (₹501.32).\n"
        "   → Offer senior-friendly tech bundles and loyalty rewards.\n",

        "3. Gender-Based Product Optimization:\n"
        "   Males spend more on Beauty (₹487.13) and Electronics (₹466.10);\n"
        "   Females spend more on Clothing (₹467.10).\n"
        "   → Use gender-targeted promotions (e.g., tech-grooming bundles, fashion previews).\n",

        "4. Leverage December Sales Peaks:\n"
        "   Weekly sales peak in mid-late December (₹16,000+ per week).\n"
        "   → Launch early holiday promotions and increase inventory .\n",

        "5. Improve Menu Nutrition Balance:\n"
        "   Menu shows high calories/sodium and low protein/fiber.\n"
        "   → Promote high-protein, low-sodium options with 'Healthy Choice' tags.\n",

        "6. Build Loyalty and Upsell Programs:\n"
        "   Spending patterns reveal strong segmentation by age and gender.\n"
        "   → Offer personalized discounts and coupons using customer profiles.\n"
    ]
    print("\nRECOMMENDATIONS")
    print("-" * 60)
    for rec in recommendations:
        print(rec)

while True:
    print("\n Retail Sales Data Analysis Menu")
    print("1. Explore Dataset")
    print("2. Retail Descriptive Statistics")
    print("3. Menu Descriptive Statistics")
    print("4. Weekly Sales Analysis")
    print("5. Sales by Gender")
    print("6. Sales by Age Group")
    print("7. Actionable Recommendations based on the EDA")
    print("8. Exit")

    choice = input("Enter your choice (1-8): ")

    if choice == '1':
        data_loading_and_cleaning(df, menu)
    elif choice == '2':
        retail_sales_statistics(df)
    elif choice == '3':
        menu_statistics(menu)
    elif choice == '4':
        plot_weekly_sales_analysis(df)
    elif choice == '5':
        plot_sales_by_gender(df)
    elif choice == '6':
        plot_age_by_product_category(df)
    elif choice == '7':
        print_recommendations()
    elif choice == '8':
        print("Program terminated successfully. Have a great day!")
        break
    else:
        print("Invalid choice. Please enter a number from 1 to 8.")