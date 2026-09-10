import pandas as pd
import os
import joblib
import pickle
import json
import matplotlib.pyplot as plt
from sklearn import model_selection, preprocessing, metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize_scalar
from typing import Dict

class MultiLabelTFIDFModel:
    def __init__(self, training_set, validation_set, text_column: str, label_columns: list):
        self.training_set = training_set
        self.validation_set = validation_set
        self.text_column = text_column
        self.label_columns = label_columns
        self.vectorizer = TfidfVectorizer()
        self.label_encoder = LabelEncoder()
        self.model = MultiOutputClassifier(LogisticRegression(max_iter=1000))

        self.metrics = {}
    
    def preprocess_data(self):
        # Fit the TF-IDF vectorizer on the training data and transform both training and validation data
        self.X_train = self.vectorizer.fit_transform(self.training_set[self.text_column])
        self.X_validation = self.vectorizer.transform(self.validation_set[self.text_column])

        # Encode the labels for each label column
        self.y_train = self.training_set[self.label_columns].apply(self.label_encoder.fit_transform)
        self.y_validation = self.validation_set[self.label_columns].apply(self.label_encoder.transform)

    def train_model(self):
        # Train the multi-output logistic regression model
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        # Make predictions on the validation set
        y_pred = self.model.predict(self.X_validation)

        # Calculate accuracy for each label column
        for i, label in enumerate(self.label_columns):
            self.metrics[label] = {}

            accuracy = metrics.accuracy_score(self.y_validation[label], y_pred[:, i])
            self.metrics[label]['accuracy'] = accuracy
            self.metrics[label]['classification_report'] = metrics.classification_report(self.y_validation[label], y_pred[:, i], digits=4)
            print(f'Validation Accuracy for {label}: {accuracy:.4f}')
            print(f'Classification Report for {label}: \n{self.metrics[label]["classification_report"]}')

        macro_f1_score = metrics.f1_score(self.y_validation, y_pred, average='macro')
        self.metrics['macro_f1_score'] = macro_f1_score

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
        joblib.dump(artifacts, os.path.join(output_dir, 'multi_label_model.pkl'))

    def save_metrics(self, output_dir, output_format='txt'):
        # Save the evaluation metrics to a JSON file
        os.makedirs(output_dir, exist_ok=True)
        if output_format == 'txt':
            with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
                for label, metrics_dict in self.metrics.items():
                    f.write(f'Metrics for {label}:\n')
                    if isinstance(metrics_dict, dict):
                        for metric_name, metric_value in metrics_dict.items():
                            f.write(f'{metric_name}: {metric_value}\n')
                    else:
                        f.write(f'{metrics_dict:.4f}\n')
                f.write('\n')
        elif output_format == 'json':
            with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
                json.dump(self.metrics, f, indent=4)

    