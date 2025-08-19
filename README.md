# OIBSIP_NM
# Internship Project Portfolio

This repository documents all projects completed during the course of a professional internship focused on applied data analysis and visualization using Python. Each project follows a structured format involving data exploration, statistical analysis, and insight generation.

## Projects Completed

### 1. Exploratory Data Analysis on Retail Sales and Nutritional Data

**Description:**
A menu-driven Python application to perform exploratory data analysis on retail sales and nutritional datasets. The tool computes descriptive statistics, generates behavior-based visualizations, and outputs actionable business recommendations.

**Key Features:**

* Interactive CLI with multiple analysis modules
* Descriptive statistics (mean, median, mode, std deviation)
* Weekly time series trend analysis
* Segmentation by gender and age
* Bar charts and line plots
* Data-driven recommendation engine

**Technologies:** Python, Pandas, NumPy, Matplotlib, SciPy

**Objective:** To practice real-world data analysis workflows, improve reporting skills, and interpret structured insights from raw data.

**Status:** Completed


### 2. Sentiment Analysis on Twitter Data

**Description:**
A Python-based application to perform sentiment analysis on Twitter data using Natural Language Processing (NLP) and machine learning algorithms. The tool processes tweet text, applies TF-IDF vectorization with additional engineered features, trains classification models, and presents results through multiple visualization options.

**Key Features:**

* Data cleaning pipeline to remove links, usernames, and special characters
* Feature engineering including character count and word count
* TF-IDF vectorization with extra features stacked into the model input
* Sentiment classification into Positive, Neutral, and Negative
* Model comparison between Naive Bayes and Linear SVM
* Confusion matrix heatmaps for both models using Matplotlib
* Comparative bar charts for prediction distributions
* Interactive CLI menu for selecting visualization outputs

**Technologies:**
Python, Pandas, NumPy, SciPy, Matplotlib, scikit-learn

**Objective:**
To implement a complete sentiment analysis workflow from raw Twitter data to model training, evaluation, and visualization, while practicing feature engineering, model comparison, and result interpretation.

**Status:**
Completed


## 3. Analyzing Google Play Store Data

**Description:**
A Python-based application to analyze Google Play Store application data, focusing on category distribution, key metrics, and sentiment analysis from user reviews. The tool processes two datasets (`apps.csv` and `user_reviews.csv`), performs data cleaning, integrates related fields, and presents insights through multiple visualization options in an interactive CLI menu.

**Key Features:**

* Data cleaning pipeline to standardize numerical formats in `Price` and `Installs` and handle missing values.
* Integration of app details and review sentiment data via dataset merging.
* Category exploration to visualize the distribution of apps across various categories.
* Metrics analysis including:

  * Average rating per category
  * Average installs per category
  * Comparison of installs between free and paid apps
* Sentiment analysis including:

  * Distribution of review sentiment for free vs paid apps
  * Sentiment polarity histograms by sentiment category
* Interactive CLI menu for accessing dataset previews, category exploration, metrics analysis, and sentiment analysis visualizations.

**Technologies:** Python, Pandas, NumPy, Matplotlib

**Objective:**
To implement a complete exploratory data analysis (EDA) workflow on Google Play Store datasets, integrating app metadata with sentiment data from user reviews to identify category trends, usage patterns, and sentiment distributions.

**Status:** Completed

## 4. Data Cleaning Menu for Airbnb & YouTube Datasets

Description: A menu‑driven Python application to demonstrate the five core data‑cleaning steps on the Airbnb NYC (2019) dataset and YouTube Trending datasets for Canada (CA), Great Britain (GB), and the United States (US). The tool audits data quality, applies policy‑based missing handling, standardizes formats, validates integrity via Boolean flags, and visualizes price outliers (Airbnb) using the IQR method.

## Key Features

* Interactive CLI with dataset‑specific submenus (Airbnb, CA, GB, US)
* Audit module: missing counts (total and per column) and duplicate counts
* Missing‑value handling policies:

  * Airbnb: `number_of_reviews → 0`; conditional `reviews_per_month → 0.0` when `last_review` is missing; preserve `last_review` as NaT
  * YouTube: fill missing `description` with "Unknown"
* Standardization and integrity checks:

  * Airbnb: parse `last_review` (ISO shadow column); flags for `availability_365` outside \[0, 365], non‑positive `price`, negative `number_of_reviews`, optional logical check (reviews present requires `last_review`)
  * YouTube: flags for negative `views/likes/dislikes/comment_count`; cross‑field check that `likes + dislikes ≤ views` (implemented for CA)
* Outlier visualization: histogram of Airbnb `price` with Q1, Q3, and the upper IQR fence (k = 1.5)
* Non‑destructive workflow: values are not altered; issues are flagged (True/False)

**Technologies:** Python, Pandas, NumPy, Matplotlib, scikit‑learn, pathlib/os

**Objective:** Practice real‑world data‑cleaning workflows aligned to the rubric (Data Integrity, Missing Data Handling, Duplicate Removal, Standardization, Outlier Detection), and present a reproducible, validation‑first approach suitable for internship review.

**Status:** Completed

## 5. Predicting House Prices with Linear Regression

**Description:** A Python application that builds a linear regression model to estimate residential property prices from structural and amenity features. The workflow covers data loading, cleaning, categorical encoding (binary and one‑hot), model training/evaluation, and a parity plot to visualize agreement between predictions and actual prices. Dataset used: `Housing.csv` (545 rows × 13 columns) with columns: `price, area, bathrooms, bedrooms, stories, mainroad, guestrooms, basement, hotwaterheating, airconditioning, parking, prefarea, furnishingstatus`.

**Key Features:**

* Robust file loading with fallback search using `pathlib` (works even if the working directory changes).
* Clean preprocessing pipeline: missing-value drop, case/whitespace normalization, **Yes/No → 0/1** encoding for binary fields, and **one‑hot** encoding for `furnishingstatus` (baseline: unfurnished).
* Clear feature set combining numeric and encoded categorical variables for modeling.
* Reproducible **train/test split** and linear regression training.
* **Model evaluation**: R² on train and test sets.
* **Visualization**: Predicted vs. Actual (parity) scatter with a 45° reference line.
* Extensible diagnostics suggested: residuals vs. predictions, residual distribution, and log‑price variant to address heteroscedasticity.

**Technologies:** Python, Pandas, NumPy, Matplotlib, scikit‑learn, pathlib/os

**Objective:** Practice an end‑to‑end applied ML workflow for tabular regression: prepare features correctly (including categorical encodings), train a baseline linear model, validate with appropriate metrics/plots, and communicate findings clearly for stakeholders.

**Status:** Completed

---

## License

This work is intended solely for academic and professional learning purposes. Datasets used are assumed to be publicly available or anonymized for educational analysis.

## Author Note

All projects were completed as part of an internship curriculum designed to develop practical data analysis skills. Each project adheres to assigned objectives and is documented with clarity and reproducibility in mind.

Additional projects completed during the internship will be appended here as they are finalized.
