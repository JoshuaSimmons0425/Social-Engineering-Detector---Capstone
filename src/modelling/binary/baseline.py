# Baseline model class

import pandas as pd
import os
import pickle
import json
from sklearn import model_selection, preprocessing, metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from typing import Dict

class TFIDFBaselineModel:
    def __init__(self, split_items: Dict[str, str], df: pd.DataFrame, text_column: str, label_column: str):
        self.df = df
        self.X_train = split_items['x_train']
        self.X_validation = split_items['x_validation']
        self.y_train = split_items['y_train']
        self.y_validation = split_items['y_validation']
        self.text_column = text_column
        self.label_column = label_column
        self.vectorizer = TfidfVectorizer()
        self.label_encoder = LabelEncoder()
        self.model = LogisticRegression(max_iter=1000)

    def preprocess_data(self):
        # Fit the TF-IDF vectorizer on the training data and transform both training and validation data

        self.vectorizer.fit(self.df[self.text_column])

        self.X_train = self.vectorizer.fit_transform(self.X_train)
        self.X_validation = self.vectorizer.transform(self.X_validation)

        # Encode the labels
        self.y_train = self.label_encoder.fit_transform(self.y_train)
        self.y_validation = self.label_encoder.transform(self.y_validation)

    def train_model(self):
        # Train the logistic regression model
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        # Make predictions on the validation set
        y_pred = self.model.predict(self.X_validation)

        # Calculate accuracy
        accuracy = metrics.accuracy_score(self.y_validation, y_pred)
        classification_report = metrics.classification_report(self.y_validation, y_pred, target_names=self.label_encoder.classes_)
        print(f'Validation Accuracy: {accuracy:.4f}')
        print(f'Classification Report: \n{classification_report}')

    def run_pipeline(self):
        self.preprocess_data()
        self.train_model()
        self.evaluate_model()

    def save_model(self, output_dir):
        # Save the trained model, vectorizer, and label encoder to the specified output directory
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'model.pkl'), 'wb') as f:
            pickle.dump(self.model, f)
        with open(os.path.join(output_dir, 'vectorizer.pkl'), 'wb') as f:
            pickle.dump(self.vectorizer, f)
        with open(os.path.join(output_dir, 'label_encoder.pkl'), 'wb') as f:
            pickle.dump(self.label_encoder, f)

    def save_metrics(self, output_dir):
        # Save the evaluation metrics to a text file in the specified output directory
        os.makedirs(output_dir, exist_ok=True)
        y_pred = self.model.predict(self.X_validation)
        accuracy = metrics.accuracy_score(self.y_validation, y_pred)
        classification_report = metrics.classification_report(self.y_validation, y_pred, target_names=self.label_encoder.classes_)
        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            json.dump({
                'validation_accuracy': accuracy,
                'classification_report': classification_report
            }, f)

        