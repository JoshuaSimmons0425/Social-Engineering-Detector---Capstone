import os
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy.optimize import minimize
from sklearn.calibration import calibration_curve
from sklearn.metrics import log_loss, brier_score_loss

class MLBaselineEvaluator:
    def __init__(self, model, estimators, vectorizer, platt_params, test_set_50_50, test_set_80_20, text_column, label_columns):
        self.model = model
        self.estimators = estimators
        self.vectorizer = vectorizer
        self.platt_params = platt_params

        # Keep raw sets to extract targets, preprocess transforms text fields
        self.raw_test_set_50_50 = test_set_50_50
        self.raw_test_set_80_20 = test_set_80_20
        self.text_column = text_column
        self.label_columns = label_columns

        self.X_test_50_50 = None
        self.X_test_80_20 = None

        self.discrimination_results = {}
        self.uncalibrated_results = {}
        self.calibrated_results = {}

    def preprocess(self):
        self.X_test_50_50 = self.vectorizer.transform(self.raw_test_set_50_50[self.text_column])
        self.X_test_80_20 = self.vectorizer.transform(self.raw_test_set_80_20[self.text_column])
        
    def _get_test_data(self, test_set):
        y_test = test_set[self.label_columns]
        return y_test

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

    def evaluate_uncalibration(self):
        # Make predictions on the validation set
        y_test = self._get_test_data(self.raw_test_set_50_50)
        y_pred = self.model.predict(self.X_test_50_50)

        # Calculate accuracy for each label column
        for i, label in enumerate(self.label_columns):
            self.discrimination_results[label] = {}

            accuracy = metrics.accuracy_score(y_test[label], y_pred[:, i])
            self.discrimination_results[label]['accuracy'] = float(accuracy)
            self.discrimination_results[label]['classification_report'] = metrics.classification_report(
                y_test[label], y_pred[:, i], digits=4, output_dict=True
            )
            self.discrimination_results[label]['classification_report_str'] = metrics.classification_report(
                y_test[label], y_pred[:, i], digits=4
            )
            print(f'Validation Accuracy for {label} (50:50): {accuracy:.4f}')
            print(f'Classification Report for {label}: \n{self.discrimination_results[label]["classification_report_str"]}')

        macro_f1_score = metrics.f1_score(y_test, y_pred, average='macro')
        self.discrimination_results['macro_f1_score'] = float(macro_f1_score)
        print(f"Overall Macro F1 Score (50:50): {macro_f1_score:.4f}\n")

    def evaluate_calibration(self):
        """Evaluates uncalibrated and calibrated quality profiles without using discrete thresholds."""
        # 1. Gather predicted probabilities on validation set
        y_test = self._get_test_data(self.raw_test_set_80_20)
        y_uncal_probs = self.model.predict_proba(self.X_test_80_20)

        y_cal_probs = np.column_stack(
            [
                calibrator.predict_proba(self.X_test_80_20)[:, 1]
                for calibrator in self.estimators
            ]
        )

        # Score continuous probability metrics per label
        for i, label in enumerate(self.label_columns):
            y_test = self._get_test_data(self.raw_test_set_80_20)
            true_labels = y_test[label].values

            # Uncalibrated processing
            uncal_label_probs = y_uncal_probs[:, i]
            self.uncalibrated_results[label] = {
                "brier_score": float(brier_score_loss(true_labels, uncal_label_probs)),
                "log_loss": float(log_loss(true_labels, uncal_label_probs)),
                "ece": float(self._calculate_ece(true_labels, uncal_label_probs)),
            }

            # Calibrated processing
            cal_label_probs = y_cal_probs[:, i]
            self.calibrated_results[label] = {
                "brier_score": float(brier_score_loss(true_labels, cal_label_probs)),
                "log_loss": float(log_loss(true_labels, cal_label_probs)),
                "ece": float(self._calculate_ece(true_labels, cal_label_probs)),
            }
            print(f"Evaluated continuous metrics for baseline label (80:20): {label}")

    def run_evaluation(self):
        self.preprocess()
        self.evaluate_uncalibration()
        self.evaluate_calibration()

    def save_results(self, save_path):
        """Serializes pipeline metric evaluation results into structured JSON and TXT format."""
        os.makedirs(save_path, exist_ok=True)

        # Assemble unified results payload
        combined_json_results = {
            "discrimination_50_50_set": {
                "macro_f1_score": self.discrimination_results.get('macro_f1_score', None),
                "labels": {
                    lbl: {
                        "accuracy": self.discrimination_results[lbl]["accuracy"],
                        "classification_report": self.discrimination_results[lbl]["classification_report"]
                    } for lbl in self.label_columns if lbl in self.discrimination_results
                }
            },
            "calibration_80_20_set": {
                "uncalibrated_profiles": self.uncalibrated_results,
                "calibrated_profiles": self.calibrated_results
            }
        }

        # 1. Output structured JSON
        json_file_path = os.path.join(save_path, "evaluation_metrics.json")
        with open(json_file_path, "w") as f:
            json.dump(combined_json_results, f, indent=4)

        # 2. Output human-scannable TXT Report
        txt_file_path = os.path.join(save_path, "evaluation_report.txt")
        with open(txt_file_path, "w") as f:
            f.write("==================================================\n")
            f.write("       MULTILABEL BASELINE EVALUATION REPORT      \n")
            f.write("==================================================\n\n")

            # Section A: Discrimination Profiles
            f.write("--------------------------------------------------\n")
            f.write("1. DISCRIMINATION RESULTS (50:50 Balanced Test Set)\n")
            f.write("--------------------------------------------------\n")
            f.write(f"Macro F1 Score: {self.discrimination_results.get('macro_f1_score', 0.0):.5f}\n\n")
            
            for label in self.label_columns:
                if label in self.discrimination_results:
                    f.write(f"Label Name: {label}\n")
                    f.write(f"Accuracy  : {self.discrimination_results[label]['accuracy']:.5f}\n")
                    f.write("Classification Matrix Profile:\n")
                    f.write(f"{self.discrimination_results[label]['classification_report_str']}\n")
                    f.write("-" * 40 + "\n")

            # Section B: Calibration Profiles
            f.write("\n--------------------------------------------------\n")
            f.write("2. PROBABILITY CALIBRATION RESULTS (80:20 Test Set)\n")
            f.write("--------------------------------------------------\n")
            
            for label in self.label_columns:
                if label in self.uncalibrated_results and label in self.calibrated_results:
                    f.write(f"Label Name: {label}\n")
                    f.write("  [Uncalibrated Metrics]\n")
                    f.write(f"    Brier Score : {self.uncalibrated_results[label]['brier_score']:.5f}\n")
                    f.write(f"    Log Loss    : {self.uncalibrated_results[label]['log_loss']:.5f}\n")
                    f.write(f"    ECE         : {self.uncalibrated_results[label]['ece']:.5f}\n")
                    f.write("  [Calibrated Metrics]\n")
                    f.write(f"    Brier Score : {self.calibrated_results[label]['brier_score']:.5f}\n")
                    f.write(f"    Log Loss    : {self.calibrated_results[label]['log_loss']:.5f}\n")
                    f.write(f"    ECE         : {self.calibrated_results[label]['ece']:.5f}\n")
                    f.write("-" * 40 + "\n")

        print(f"[INFO] Evaluation results successfully saved down to path location: '{save_path}'")