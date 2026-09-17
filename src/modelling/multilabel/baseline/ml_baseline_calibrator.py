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

    def evaluate_calibration(self):
        """Evaluates uncalibrated and calibrated quality profiles without using discrete thresholds."""
        # 1. Gather predicted probabilities on validation set
        y_uncal_probs = self.model.predict_proba(self.X_validation)
        y_cal_probs = np.column_stack(
            [
                calibrator.predict_proba(self.X_validation)[:, 1]
                for calibrator in self.calibrated_estimators
            ]
        )

        # 2. Score metrics per label
        for i, label in enumerate(self.all_labels):
            true_labels = self.y_validation[label].values

            # Uncalibrated processing
            uncal_label_probs = y_uncal_probs[:, i]
            self.uncalibrated_metrics[label] = {
                "brier_score": float(
                    brier_score_loss(true_labels, uncal_label_probs)
                ),
                "log_loss": float(log_loss(true_labels, uncal_label_probs)),
                "ece": float(self._calculate_ece(true_labels, uncal_label_probs)),
            }

            # Calibrated processing
            cal_label_probs = y_cal_probs[:, i]
            self.calibrated_metrics[label] = {
                "brier_score": float(
                    brier_score_loss(true_labels, cal_label_probs)
                ),
                "log_loss": float(log_loss(true_labels, cal_label_probs)),
                "ece": float(self._calculate_ece(true_labels, cal_label_probs)),
            }
            print(f"Evaluated continuous metrics for baseline label: {label}")

    def run_pipeline(self):
        self.preprocess_data()
        self.calibrate()
        self.evaluate_calibration()

    def save_calibration_artifacts(self, save_path):
        os.makedirs(save_path, exist_ok=True)
        # Save weights
        with open(os.path.join(save_path, "calibrated_estimators.pkl"), "wb") as f:
            pickle.dump(self.calibrated_estimators, f)

        # Save parameters extracted
        with open(os.path.join(save_path, "platt_parameters.json"), "w") as f:
            json.dump(self.platt_parameters, f, indent=4)
            
        print(f"\n[INFO] Successfully serialized calibrated pipeline components to: {save_path}")

    def save_metrics(self, save_path, output_format='txt'):
        """Saves baseline evaluation metrics cleanly to disk"""
        os.makedirs(save_path, exist_ok=True)

        if output_format == "json":
            results = {
                "uncalibrated_metrics": self.uncalibrated_metrics,
                "calibrated_metrics": self.calibrated_metrics,
            }
            with open(os.path.join(save_path, "results.json"), "w") as f:
                json.dump(results, f, indent=4)

        elif output_format == "txt":
            with open(os.path.join(save_path, "results.txt"), "w") as f:
                f.write(
                    "==================================================\n"
                )
                f.write("BASELINE LAYER PROBABILITY EVALUATIONS\n")
                f.write(
                    "==================================================\n\n"
                )

                for label in self.all_labels:
                    f.write(f"Label: {label}\n")
                    f.write("  [Uncalibrated Profile]\n")
                    f.write(
                        f"    Brier Score : {self.uncalibrated_metrics[label]['brier_score']:.5f}\n"
                    )
                    f.write(
                        f"    Log Loss    : {self.uncalibrated_metrics[label]['log_loss']:.5f}\n"
                    )
                    f.write(
                        f"    ECE         : {self.uncalibrated_metrics[label]['ece']:.5f}\n"
                    )

                    f.write("  [Calibrated Profile]\n")
                    f.write(
                        f"    Brier Score : {self.calibrated_metrics[label]['brier_score']:.5f}\n"
                    )
                    f.write(
                        f"    Log Loss    : {self.calibrated_metrics[label]['log_loss']:.5f}\n"
                    )
                    f.write(
                        f"    ECE         : {self.calibrated_metrics[label]['ece']:.5f}\n"
                    )
                    f.write("-" * 50 + "\n\n")

        print(f"Baseline calibration results saved to {save_path}")