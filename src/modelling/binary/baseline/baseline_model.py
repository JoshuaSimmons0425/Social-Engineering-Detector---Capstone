# Baseline model class

import pandas as pd
import os
import joblib
import pickle
import json
import matplotlib.pyplot as plt
from sklearn import model_selection, preprocessing, metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize_scalar
from typing import Dict

class TFIDFBaselineModel:
    def __init__(self, training_set, validation_set, text_column: str, label_column: str):
        self.training_set = training_set
        self.validation_set = validation_set
        self.text_column = text_column
        self.label_column = label_column
        self.vectorizer = TfidfVectorizer()
        self.label_encoder = LabelEncoder()
        self.model = LogisticRegression(max_iter=1000)

    def preprocess_data(self):
        # Fit the TF-IDF vectorizer on the training data and transform both training and validation data

        self.X_train = self.vectorizer.fit_transform(self.training_set[self.text_column])
        self.X_validation = self.vectorizer.transform(self.validation_set[self.text_column])

        # Encode the labels
        self.y_train = self.label_encoder.fit_transform(self.training_set[self.label_column])
        self.y_validation = self.label_encoder.transform(self.validation_set[self.label_column])

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
        artifacts = {
                    'model': self.model,
                    'vectorizer': self.vectorizer,
                    'encoder': self.label_encoder
                    }
                
        filepath = os.path.join(output_dir, 'uncalibrated_baseline_meta.pkl')
        with open(filepath, 'wb') as f:
            pickle.dump(artifacts, f)
        print(f'Uncalibrated model artifacts saved to {filepath}')

    def save_metrics(self, output_dir, output_format='txt'):
        # Save the evaluation metrics to a text file in the specified output directory
        os.makedirs(output_dir, exist_ok=True)
        y_pred = self.model.predict(self.X_validation)
        accuracy = metrics.accuracy_score(self.y_validation, y_pred)
        classification_report = metrics.classification_report(self.y_validation, y_pred, digits=4, target_names=self.label_encoder.classes_)
        if output_format == 'txt':
            with open(os.path.join(output_dir, 'baseline_validation_metrics.txt'), 'w') as f:
                f.write(f'Validation Accuracy: {accuracy:.4f}\n')
                f.write(f'Classification Report: \n{classification_report}\n')
        elif output_format == 'json':
            with open(os.path.join(output_dir, 'baseline_validation_metrics.json'), 'w') as f:
                json.dump({
                    'validation_accuracy': accuracy,
                    'classification_report': classification_report
                }, f)

        cm = metrics.confusion_matrix(self.y_validation, y_pred)
        class_names = self.label_encoder.classes_
        cm_filepath = os.path.join(output_dir, 'baseline_confusion_matrix.png')
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        fig_cm, ax = plt.subplots(figsize=(6, 6))
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        fig_cm.savefig(cm_filepath, bbox_inches='tight')
        


        