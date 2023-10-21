import json
import argparse 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss 

def parse_json_data(column):
    """
    Replace single quotes with double quotes and parse JSON data.
    """
    return column.str.replace("'", '"').apply(json.loads).apply(pd.Series)

def consolidate_features(csv_path):
    """
    Consolidate human and AI features from a given CSV.
    """
    df = pd.read_csv(csv_path)
    
    human_features_df = parse_json_data(df['human_features'])
    human_features_df['type'] = 0

    ai_features_df = parse_json_data(df['ai_features'])
    ai_features_df['type'] = 1

    consolidated_df = pd.concat([human_features_df, ai_features_df], ignore_index=True)
    
    return consolidated_df

def format_tabular_data_for_classifier():
    '''This function consolidates the features from the human and AI generated stories into a single dataframe.'''
    csv_path = 'datasets/experiment1/experiment1.csv'
    consolidated_df = consolidate_features(csv_path)
    
    # (Optional) Save the consolidated dataframe to a new CSV
    consolidated_df.to_csv('datasets/experiment1/experiment1_classifier.csv', index=False)
    
    print("Consolidation complete!")

def preprocess_data(X, Y):
    """Preprocesses the data by splitting and scaling."""
    # Split the data into training and test sets (80-20)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

    # Standardize the features (mean=0 and variance=1)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    return x_train_scaled, x_test_scaled, y_train, y_test

def train_logistic(x_train_scaled, y_train):
    """Trains a logistic regression model and returns the model and metrics."""
    logreg = LogisticRegression(random_state=42, max_iter=1000)
    logreg.fit(x_train_scaled, y_train)

    # Predictions and probabilities on the training set
    logreg_train_preds = logreg.predict(x_train_scaled)
    logreg_train_probs = logreg.predict_proba(x_train_scaled)

    # Calculate training accuracy and log loss
    logreg_train_accuracy = accuracy_score(y_train, logreg_train_preds)
    logreg_train_logloss = log_loss(y_train, logreg_train_probs)

    return logreg, logreg_train_accuracy, logreg_train_logloss

def test_logistic(model, x_test_scaled, y_test):
    """Tests a logistic regression model and returns the metrics."""
    # Predictions and probabilities on the test set
    logreg_test_preds = model.predict(x_test_scaled)
    logreg_test_probs = model.predict_proba(x_test_scaled)

    # Calculate test accuracy and log loss
    logreg_test_accuracy = accuracy_score(y_test, logreg_test_preds)
    logreg_test_logloss = log_loss(y_test, logreg_test_probs)

    return logreg_test_accuracy, logreg_test_logloss

def plot_logistic_feature_importance(model):
    """Plots the feature importance for a logistic regression model."""
    
    # Get feature coefficients from the logistic regression model
    feature_coeffs = model.coef_[0]

    # create a dataframe of X.columns and feature_coeffs and sort by descending
    df = pd.DataFrame({'feature': X.columns, 'coeff': feature_coeffs})
    df = df.sort_values(by='coeff', ascending=False)

    # Plotting
    plt.figure(figsize=(15,10))
    plt.barh(df['feature'], df['coeff'], color='green')
    plt.xlabel('Coefficient Magnitude')
    plt.ylabel('Feature')
    plt.title('Feature Importance in Logistic Regression')
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.savefig('datasets/experiment1/experiment1_logistic_feature_importance.png')
    plt.show()

def main(args):
    # Read the consolidated dataframe from argparse
    data = pd.read_csv(args.data_path)
    X = data.drop("type", axis=1)
    Y = data["type"]

    # preprocess the data for training and testing
    x_train_scaled, x_test_scaled, y_train, y_test = preprocess_data(X,Y)

    # train and test the logistic regression model
    model, train_accuracy, train_logloss = train_logistic(x_train_scaled, y_train)
    test_accuracy, test_logloss = test_logistic(model, x_test_scaled, y_test)

    # print the training and test metrics
    print(f'Training accuracy: {train_accuracy:.4f}')
    print(f'Training log loss: {train_logloss:.4f}')
    print(f'Test accuracy: {test_accuracy:.4f}')
    print(f'Test log loss: {test_logloss:.4f}')

    # plot the feature importance
    plot_logistic_feature_importance(model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training a classifier for experiment 1')
    parser.add_argument('--data_path', default='datasets/experiment1/experiment1-classifier-data.csv', help='Path to the consolidated data for experiment 1 (default: datasets/experiment1/experiment1-classifier-data.csv)')
    args = parser.parse_args()
    main(args)