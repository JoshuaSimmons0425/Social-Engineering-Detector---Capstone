import os
import pickle
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy.optimize import minimize
from sklearn.calibration import calibration_curve
from sklearn.metrics import log_loss, brier_score_loss
from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

class BaselineMultiCalibrator:
    def __init__(self, model, vectorizer, calibration_set, validation_set, text_column, all_labels):
        self.model = model
        self.vectorizer = vectorizer
        self.calibration_set = calibration_set
        self.validation_set = validation_set
        self.text_column = text_column
        self.all_labels = all_labels
        self.y_calibration = self.calibration_set[self.all_labels]
        self.y_validation = self.validation_set[self.all_labels]
        self.platt_parameters = {}

        self.optimal_thresholds = {}
        self.calibrated_metrics = {}
        self.uncalibrated_metrics = {}

    def preprocess_data(self):
        self.X_calibration = self.vectorizer.transform(self.calibration_set[self.text_column])
        self.X_validation = self.vectorizer.transform(self.validation_set[self.text_column])

    def _get_logits(self, X):
        return self.model.decision_function(X)

    def _calculate_ece(self, y_true, y_prob, n_bins=10):
        """Helper method to compute Expected Calibration Error (ECE)."""
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
        
        ece = 0.0
        n_samples = len(y_true)
        
        # Bin the predictions to calculate weights
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        bin_assignments = np.digitize(y_prob, bins) - 1
        
        for i in range(len(prob_true)):
            # Find elements belonging to the current bin indices safely
            bin_idx = np.where(bin_assignments == i)[0]
            if len(bin_idx) > 0:
                bin_weight = len(bin_idx) / n_samples
                # ECE is the weighted average absolute difference between true and predicted probabilities
                ece += bin_weight * np.abs(prob_true[i] - prob_pred[i])
                
        return ece

    def calibrate(self):
        self.calibrated_estimators = [
            CalibratedClassifierCV(
            estimator=FrozenEstimator(estimator), method="sigmoid")
            for estimator in self.model.estimators_
        ]

        for calibrator, label in zip(self.calibrated_estimators, self.all_labels):
            calibrator.fit(self.X_calibration, self.y_calibration[label])
            sigmoid_model = calibrator.calibrated_classifiers_[0].calibrators[0]
            self.platt_parameters[label] = {
                "A": float(sigmoid_model.a_),
                "B": float(sigmoid_model.b_)
            }

    def optimize_thresholds(self):
        """
        Finds and stores the decision threshold that maximizes the F1-score 
        for each calibrated label estimator using the validation set.
        """
        # Get calibrated probabilities on validation set
        y_cal_probs = np.column_stack([
            calibrator.predict_proba(self.X_validation)[:, 1]
            for calibrator in self.calibrated_estimators
        ])
        
        print("--- Optimising Thresholds based on F1-Score ---")
        for i, label in enumerate(self.all_labels):
            true_labels = self.y_validation[label].values
            label_probs = y_cal_probs[:, i]
            
            best_threshold = 0.5
            best_f1 = 0.0
            
            # Grid search candidates from 0.01 to 0.99
            thresholds = np.linspace(0.01, 0.99, 99)
            
            for t in thresholds:
                preds = (label_probs >= t).astype(int)
                # Using binary average since it evaluates each single label independently
                current_f1 = metrics.fbeta_score(true_labels, preds, average='binary', zero_division=0, beta=2)
                
                if current_f1 > best_f1:
                    best_f1 = current_f1
                    best_threshold = t
            
            self.optimal_thresholds[label] = float(best_threshold)
            print(f"Optimal Threshold for {label}: {best_threshold:.2f} (Validation F2: {best_f1:.4f})")

    def evaluate_uncalibrated(self):
        y_prob = self.model.predict_proba(self.X_validation)
        y_pred = self.model.predict(self.X_validation)

        for i, label in enumerate(self.all_labels):
            self.uncalibrated_metrics[label] = {}
            label_probs = y_prob[:, i]
            label_preds = y_pred[:, i]
            true_labels = self.y_validation[label].values

            log_loss = metrics.log_loss(true_labels, label_probs)
            brier_score = metrics.brier_score_loss(true_labels, label_probs)
            accuracy = metrics.accuracy_score(true_labels, label_preds)
            ece = self._calculate_ece(true_labels, label_probs)
            self.uncalibrated_metrics[label]['log_loss'] = log_loss
            self.uncalibrated_metrics[label]['brier_score'] = brier_score
            self.uncalibrated_metrics[label]['accuracy'] = accuracy
            self.uncalibrated_metrics[label]['ECE Before'] = ece
            self.uncalibrated_metrics[label]['classification_report'] = metrics.classification_report(self.y_validation[label], y_pred[:, i], digits=4)
            print(f'Validation Accuracy for {label}: {accuracy:.4f}')
            print(f"Negative Log Loss for {label}: {log_loss:.4f}")
            print(f"Brier Score for {label}: {brier_score:.4f}")
            print(f"Expected Calibration Error (Before): {ece:.4f}")
            print(f'Classification Report for {label}: \n{self.uncalibrated_metrics[label]["classification_report"]}')

        macro_f1_score = metrics.f1_score(self.y_validation, y_pred, average='macro')
        self.uncalibrated_metrics['macro_f1_score'] = macro_f1_score

    def evaluate_calibrated(self):
        y_cal_probs = np.column_stack([
            calibrator.predict_proba(self.X_validation)[:, 1]
            for calibrator in self.calibrated_estimators
        ])

        y_cal_preds = np.zeros_like(y_cal_probs, dtype=int)
        for i, label in enumerate(self.all_labels):
            t = self.optimal_thresholds.get(label, 0.5)
            y_cal_preds[:, i] = (y_cal_probs[:, i] >= t).astype(int)

        for i, label in enumerate(self.all_labels):
            self.calibrated_metrics[label] = {}
            label_probs = y_cal_probs[:, i]
            label_preds = y_cal_preds[:, i]
            true_labels = self.y_validation[label].values

            log_loss = metrics.log_loss(true_labels, label_probs)
            brier_score = metrics.brier_score_loss(true_labels, label_probs)
            accuracy = metrics.accuracy_score(true_labels, label_preds)
            ece = self._calculate_ece(true_labels, label_probs)
            self.calibrated_metrics[label]['log_loss'] = log_loss
            self.calibrated_metrics[label]['brier_score'] = brier_score
            self.calibrated_metrics[label]['accuracy'] = accuracy
            self.calibrated_metrics[label]['ECE After'] = ece
            self.calibrated_metrics[label]['classification_report'] = metrics.classification_report(true_labels, label_preds, digits=4)
            print(f'Validation Accuracy for {label}: {accuracy:.4f}')
            print(f"Negative Log Loss for {label}: {log_loss:.4f}")
            print(f"Brier Score for {label}: {brier_score:.4f}")
            print(f"Expected Calibration Error (After): {ece:.4f}")
            print(f'Classification Report for {label}: \n{self.calibrated_metrics[label]["classification_report"]}')

        macro_f1_score = metrics.f1_score(self.y_validation, y_cal_preds, average='macro')
        self.calibrated_metrics['macro_f1_score'] = macro_f1_score

    def run_pipeline(self):
        self.preprocess_data()
        self.calibrate()
        self.evaluate_uncalibrated()
        self.optimize_thresholds()
        self.evaluate_calibrated()

    def save_calibration_artifacts(self, save_path):
        os.makedirs(save_path, exist_ok=True)
        artifact_path = os.path.join(save_path, "ml_calibrated_pipeline")

        pipeline_artifact = {
            "calibrated_estimators": self.calibrated_estimators,
            "optimal_thresholds": self.optimal_thresholds,
            "all_labels": self.all_labels,
            "vectorizer": self.vectorizer
        }

        with open(artifact_path, "wb") as f:
            pickle.dump(pipeline_artifact, f)
            
        print(f"\n[INFO] Successfully serialized calibrated pipeline components to: {artifact_path}")

    def save_metrics(self, save_path, output_format='txt'):
        os.makedirs(save_path, exist_ok=True)
        
        if output_format == 'json':
            # Recommended approach for programmatic parsing later
            with open(os.path.join(save_path, 'uncalibrated_metrics.json'), 'w') as f:
                json.dump(self.uncalibrated_metrics, f, indent=4)
            with open(os.path.join(save_path, 'calibrated_metrics.json'), 'w') as f:
                json.dump(self.calibrated_metrics, f, indent=4)
                
        elif output_format == 'txt':
            # Human-readable formatting with fixed indentation scopes
            for filename, metrics_data in [('uncalibrated_metrics.txt', self.uncalibrated_metrics), 
                                           ('calibrated_metrics.txt', self.calibrated_metrics)]:
                with open(os.path.join(save_path, filename), 'w') as f:
                    for key, val in metrics_data.items():
                        if isinstance(val, dict):
                            f.write(f'=== Metrics for {key} ===\n')
                            for metric_name, metric_value in val.items():
                                if metric_name == 'classification_report':
                                    f.write(f'{metric_name}:\n{metric_value}\n')
                                else:
                                    f.write(f'{metric_name}: {metric_value}\n')
                            f.write('\n')  # FIXED: Correct loop wrapping separation
                        else:
                            f.write(f'{key}: {val:.4f}\n\n')
