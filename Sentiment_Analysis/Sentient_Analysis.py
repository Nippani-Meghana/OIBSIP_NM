#SENTIENT ANALYSIS

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy
import re
import sklearn
import os
current_dir = os.getcwd()
file_path = os.path.join(current_dir, 'Twitter_Data.csv')
dataset = pd.read_csv(file_path)
dataset = dataset.dropna(subset=['clean_text', 'category'])

#Data Preview
def data_preview(df):
    print("The Shape of Twitter Data is ",df.shape)
    print("The columns in the dataset are :",df.columns)
    print("The first few data from Twitter Data are:")
    print(df.head())

#data_preview(dataset)

#Data Cleaning
def data_cleaning(tweet):
    if pd.isna(tweet):
        return ' '
    link = r'https:\/\/\S+'
    remove = r'[!@#$%^&*(),.?":{}|<>]'
    username = r'@\w+'
    non_link = re.sub(link,"",tweet)
    new = re.sub(remove,"",non_link)
    modified = re.sub(username,"",new)
    return modified

dataset['clean_text'] = dataset['clean_text'].apply(data_cleaning)

#Feature Engineering
dataset['char_count'] = dataset['clean_text'].str.len()
dataset['word_count'] = dataset['clean_text'].str.split().str.len()


#Sentient Analysis
label_map = {-1: "Negative", 0: "Neutral", 1: "Positive"}
dataset['sentiment_label'] = dataset['category'].map(label_map) 


#NLP
def NLP(df):
    X = df['clean_text']
    Y = df['sentiment_label']
    vectorizer = sklearn.feature_extraction.text.TfidfVectorizer()
    X_Vectorized = vectorizer.fit_transform(X)
    extra_features = df[['char_count', 'word_count']].values
    extra_features_sparse = scipy.sparse.csr_matrix(extra_features)
    X_Vectorized = scipy.sparse.hstack([X_Vectorized,extra_features_sparse])
    return X_Vectorized,Y

X_Vector, y = NLP(dataset)  


#Model Split
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X_Vector, y, test_size=0.20, random_state=0)


#Naive Bayes
MNB = sklearn.naive_bayes.MultinomialNB()
MNB.fit(X_train,y_train)
y_nb = MNB.predict(X_test)
print("Running Naive Bayes Model...")
print("Naive Bayes Accuracy -",sklearn.metrics.accuracy_score(y_test,y_nb)*100)


#Linear SVM
linear_svm = sklearn.svm.LinearSVC()
linear_svm.fit(X_train,y_train)
y_svm = linear_svm.predict(X_test)
print("Running Linear SVM Model...")
print("Linear SVM Accuracy -",sklearn.metrics.accuracy_score(y_test,y_svm)*100)


#Data Visualization
def plot_naive_bayes(df):
    counts = pd.Series(y_nb).value_counts()
    counts.plot(kind='bar', figsize=(14,6))
    plt.xlabel('Sentiment Analysis')
    plt.ylabel('Number of Predictions')
    plt.title('Naive Bayes Model')
    plt.show()

def naive_bayes_heatmap(df):
    cm_nb = sklearn.metrics.confusion_matrix(y_test, y_nb, labels=['Positive', 'Neutral', 'Negative'])
    plt.imshow(cm_nb, cmap='Blues')
    plt.colorbar()
    plt.xticks([0,1,2], ['Positive', 'Neutral', 'Negative'])
    plt.yticks([0,1,2], ['Positive', 'Neutral', 'Negative'])
    for i in range(len(cm_nb)):
        for j in range(len(cm_nb[i])):
            plt.text(j, i, cm_nb[i, j], ha='center', va='center', color='black')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Naive Bayes Confusion Matrix')
    plt.show()

def plot_linear_svm(df):
    counts = pd.Series(y_svm).value_counts()
    counts.plot(kind='bar', figsize=(14,6))
    plt.xlabel('Sentiment Analysis')
    plt.ylabel('Number of Predictions')
    plt.title('Linear SVM Model')
    plt.show()

def linear_svm_heatmap(df):
    cm_svm = sklearn.metrics.confusion_matrix(y_test, y_svm, labels=['Positive', 'Neutral', 'Negative'])
    plt.imshow(cm_svm, cmap='Blues')
    plt.colorbar()
    plt.xticks([0,1,2], ['Positive', 'Neutral', 'Negative'])
    plt.yticks([0,1,2], ['Positive', 'Neutral', 'Negative'])
    for i in range(len(cm_svm)):
        for j in range(len(cm_svm[i])):
            plt.text(j, i, cm_svm[i, j], ha='center', va='center', color='black')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Linear SVM Confusion Matrix')
    plt.show()

def plot_linear_svm_vs_naive_bayes(df):
    categories = ['Positive', 'Neutral', 'Negative']
    nb_counts = pd.Series(y_nb).value_counts().reindex(categories, fill_value=0)
    svm_counts = pd.Series(y_svm).value_counts().reindex(categories, fill_value=0)
    df_counts = pd.DataFrame({'Naive Bayes': nb_counts,'Linear SVM': svm_counts}, index=categories)
    df_counts.plot(kind='bar')
    plt.xlabel('Predicted Sentiment')
    plt.ylabel('Number of Predictions')
    plt.title('Naive Bayes vs Linear SVM')
    plt.show()


while True:
    print("\n Data Visualizations Menu")
    print("1. Plot of Naive Bayes Model")
    print("2. Heatmap for Naive Bayes")
    print("3. Plot of Linear SVM Model")
    print("4. Heatmaps for Linear SVM")
    print("5. Plot for Naive Bayes vs Linear SVM Model")
    print("6. Exit")

    choice = input("Enter your choice (1-6):")

    if choice == '1':
        plot_naive_bayes(dataset)
    elif choice == '2':
        naive_bayes_heatmap(dataset)
    elif choice == '3':
        plot_linear_svm(dataset)
    elif choice == '4':
        linear_svm_heatmap(dataset)
    elif choice == '5':
        plot_linear_svm_vs_naive_bayes(dataset)
    elif choice == '6':
        print("Program terminated successfully. Have a great day!")
        break
    else:
        print("Invalid Choice. Please enter again")


            
