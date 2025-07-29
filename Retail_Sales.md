# Project: Exploratory Data Analysis on Retail Sales and Nutritional Data

## Code Walkthrough: Weekly Sales Analysis Module

This walkthrough highlights one of the key components of the menu-driven EDA tool built using Python. The focus is on analyzing weekly retail sales trends and deriving actionable business recommendations based on observable patterns.

### Step 1: Launching the Menu-Driven Interface

Upon running the script, the user is presented with a menu interface:

```
 Retail Sales Data Analysis Menu
1. Explore Dataset
2. Retail Descriptive Statistics
3. Menu Descriptive Statistics
4. Weekly Sales Analysis
5. Sales by Gender
6. Sales by Age Group
7. Actionable Recommendations based on the EDA
8. Exit
```

Option 4 initiates the **Weekly Sales Analysis** module.

### Step 2: Weekly Sales Analysis Function Logic

The script performs the following:

* Parses the `Date` column from the retail sales dataset
* Sorts the transactions chronologically
* Resamples the data by week (`W` frequency)
* Calculates the total `Total Amount` spent each week
* Plots a time series line chart using `matplotlib`

```python
# Date preprocessing and resampling
retail_df['Date'] = pd.to_datetime(retail_df['Date'])
retail_df = retail_df.sort_values(by='Date')
retail_df = retail_df.set_index('Date')
weekly_sales = retail_df['Total Amount'].resample('W').sum()

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(weekly_sales)
plt.xlabel('Date (Weekly)')
plt.ylabel('Total Amount')
plt.title('Weekly Sales Analysis')
plt.show()
```

### Step 3: Output – Weekly Sales Trend Graph

A time series plot is generated showing the total sales volume on a weekly basis.

**Example Screenshot:**

![Weekly Sales Analysis Plot](weekly_sales_plot.png)

The x-axis shows week ending dates, and the y-axis shows total revenue per week.

### Step 4: Key Observation

The graph reveals a **sharp spike in sales during mid-to-late December**, suggesting a seasonal peak that aligns with holiday shopping patterns.

### Step 5: Recommendation

Based on this insight, the following recommendation is presented to the user through the recommendation module:

> **"Launch early holiday promotions and scale inventory planning to capitalize on the December sales spike."**

This recommendation was derived directly from the data, in line with the expected objectives of the project.

### Closing Note

This module demonstrates how structured time-based resampling and simple visualizations can help uncover valuable business insights. All logic and interpretation were guided by the project's original analysis framework.
