# Create a class that evaluates the performance of the baseline binary classifier both uncalibrated and calibrated
import os
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy.optimize import minimize
from sklearn.calibration import calibration_curve
from sklearn.metrics import log_loss, brier_score_loss
from sklearn import metrics

class BaselineEvaluator:
    def __init__(self, model, encoder, vectorizer, temperature, test_set_50_50, test_set_80_20, text_column, label_column): 
        self.model = model
        self.encoder = encoder
        self.vectorizer = vectorizer
        self.temperature = temperature
        self.test_set_50_50 = test_set_50_50
        self.test_set_80_20 = test_set_80_20
        self.text_column = text_column
        self.label_column = label_column

        self.uncalibrated_results = None
        self.calibrated_results = None

        self.uncalibrated_diagrams = None
        self.calibrated_diagrams = None

    def preprocess(self, X):
        X_encoded = self.encoder.transform(X)
        X_vectorized = self.vectorizer.transform(X_encoded)
        return X_vectorized

    def get_test_data(self, test_set):
        X_test, y_test = test_set[self.text_column], test_set[self.label_column]
        X_test_preprocessed = self.preprocess(X_test)
        return X_test_preprocessed, y_test

    def evaluate_uncalibrated(self):

        X_test_50_50, y_test_50_50 = self.get_test_data(self.test_set_50_50)

        y_pred = self.model.predict(X_test_50_50)
        accuracy = metrics.accuracy_score(y_test_50_50, y_pred)
        precision = metrics.precision_score(y_test_50_50, y_pred)
        recall = metrics.recall_score(y_test_50_50, y_pred)
        f1 = metrics.f1_score(y_test_50_50, y_pred)
        self.uncalibrated_results = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    def _get_logits(self, X):
        # Get raw logits from the model
        return self.model.decision_function(X)

    def predict_proba_uncalibrated(self, X):
        # Predict probabilities using the uncalibrated model
        raw_logits = self._get_logits(X)
        probs = 1 / (1 + np.exp(-raw_logits))
        return np.vstack([1 - probs, probs]).T  # Return as a 2D array with shape (n_samples, 2)

    def predict_proba_temp_scaled(self, X):
        logits = self.model.predict_proba(X)
        scaled_logits = logits ** (1 / self.temperature)
        scaled_probs = scaled_logits / scaled_logits.sum(axis=1, keepdims=True)
        return scaled_probs

    def evaluate_calibrated(self):

        X_test_80_20, y_test_80_20 = self.get_test_data(self.test_set_80_20)

        y_pred_proba = self.predict_proba_temp_scaled(X_test_80_20)
        y_pred = np.argmax(y_pred_proba, axis=1)
        brier = metrics.brier_score_loss(y_test_80_20, y_pred_proba[:, 1])
        log_loss = metrics.log_loss(y_test_80_20, y_pred_proba)
        accuracy = metrics.accuracy_score(y_test_80_20, y_pred)
        precision = metrics.precision_score(y_test_80_20, y_pred)
        recall = metrics.recall_score(y_test_80_20, y_pred)
        f1 = metrics.f1_score(y_test_80_20, y_pred)
        self.calibrated_results = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "brier": brier,
            "log_loss": log_loss
        }

    def plot_uncalibrated_results(self):

        # Fetch the 50:50 test dataset
        X_test_50_50, y_test_50_50 = self.get_test_data(self.test_set_50_50)
        
        y_pred = self.model.predict(X_test_50_50)
        y_prob = self.predict_proba_uncalibrated(X_test_50_50)[:, 1]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        ## 1. Create confusion matrix for uncalibrated results
        cm = metrics.confusion_matrix(y_test_50_50, y_pred)
        class_names = self.encoder.classes_
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(ax=ax1, cmap=plt.cm.Blues)
        ax1.set_title("Uncalibrated Confusion Matrix")

        ## 2. Create ROC-AUC curve for uncalibrated results (using probabilities)
        fpr, tpr, _ = metrics.roc_curve(y_test_50_50, y_prob)
        roc_auc = metrics.auc(fpr, tpr)
        ax2.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        ax2.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('Uncalibrated ROC Curve')
        ax2.legend(loc="lower right")

        plt.tight_layout()
        self.uncalibrated_diagrams = fig

    def plot_calibrated_results(self):

         # Fetch the 80:20 imbalanced test dataset
        X_test_80_20, y_test_80_20 = self.get_test_data(self.test_set_80_20)
        
        y_uncal_proba = self.predict_proba_uncalibrated(X_test_80_20)[:, 1]
        y_scaled_proba = self.predict_proba_temp_scaled(X_test_80_20)
        y_cal_proba_pos = y_scaled_proba[:, 1]
        y_pred_cal = np.argmax(y_scaled_proba, axis=1)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        ## 1. Create Reliability Diagram comparing uncalibrated vs calibrated models
        prob_true_uncal, prob_pred_uncal = calibration_curve(y_test_80_20, y_uncal_proba, n_bins=10)
        prob_true_cal, prob_pred_cal = calibration_curve(y_test_80_20, y_cal_proba_pos, n_bins=10)
        
        ax1.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
        ax1.plot(prob_pred_uncal, prob_true_uncal, "s-", color="red", label="Uncalibrated")
        ax1.plot(prob_pred_cal, prob_true_cal, "s-", color="green", label="Calibrated (Temp Scaled)")
        ax1.set_xlabel("Mean Predicted Probability")
        ax1.set_ylabel("Fraction of Positives")
        ax1.set_title("Reliability Diagram")
        ax1.legend(loc="lower right")

        ## 2. Create confusion matrix for calibrated results
        cm = metrics.confusion_matrix(y_test_80_20, y_pred_cal)
        class_names = self.encoder.classes_
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(ax=ax2, cmap=plt.cm.Greens)
        ax2.set_title("Calibrated Confusion Matrix")

        ## 3. Create Precision-Recall AUC Curve
        precision, recall, _ = metrics.precision_recall_curve(y_test_80_20, y_cal_proba_pos)
        pr_auc = metrics.auc(recall, precision)
        ax3.plot(recall, precision, color='purple', lw=2, label='PR curve (area = %0.2f)' % pr_auc)
        ax3.set_xlim([0.0, 1.0])
        ax3.set_ylim([0.0, 1.05])
        ax3.set_xlabel('Recall')
        ax3.set_ylabel('Precision')
        ax3.set_title('Calibrated PR Curve')
        ax3.legend(loc="lower left")

        plt.tight_layout()
        self.calibrated_diagrams = fig


