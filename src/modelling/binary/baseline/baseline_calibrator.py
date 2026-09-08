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

# A class for calibrating the baseline model using temperature scaling

class TFIDFBaselineCalibrator:
    def __init__(self, model, vectorizer, encoder, calibration_set, validation_set, text_column: str, label_column: str, temperature: float = 1.0):
        self.original_model = model
        self.calibration_set = calibration_set
        self.validation_set = validation_set
        self.text_column = text_column
        self.label_column = label_column
        self.vectorizer = vectorizer
        self.label_encoder = encoder
        self.A = None
        self.B = None

        self.uncalibrated_metrics = None
        self.calibrated_metrics = None

    def preprocess_data(self):
        # Fit the TF-IDF vectorizer on the calibration data and transform both calibration and validation data
        self.X_calibration = self.vectorizer.transform(self.calibration_set[self.text_column])
        self.X_validation = self.vectorizer.transform(self.validation_set[self.text_column])

        # Encode the labels
        self.y_calibration = self.label_encoder.transform(self.calibration_set[self.label_column])
        self.y_validation = self.label_encoder.transform(self.validation_set[self.label_column])

    def _get_logits(self, X):
        # Extract raw logits (before sigmoid/softmax conversion) from LogisticRegression
        # decision_function returns: w * X + b
        return self.original_model.decision_function(X)

    def calibrate(self):
        # Optimize the temperature parameter to minimize the loss
        raw_logits = self._get_logits(self.X_calibration)
        
        def objective(params):
            # Prevent division by zero
            A = params[0]
            B = params[1]

            # Apply affine transformation to logits
            scaled_logits = A * raw_logits + B

            # Convert scaled logits to probabilities using sigmoid
            probs = 1 / (1 + np.exp(-scaled_logits))

            # Return log loss against true labels
            return log_loss(self.y_calibration, probs)
        
        result = minimize(objective, x0=[1.0, 0.0], method='Nelder-Mead')
        self.A = result.x[0]
        self.B = result.x[1]

        print(f"Optimized A: {self.A:.4f}, B: {self.B:.4f}")

    def predict_proba(self, X):
        # Predict probabilities using the calibrated model
        raw_logits = self._get_logits(X)
        scaled_logits = self.A * raw_logits + self.B
        probs = 1 / (1 + np.exp(-scaled_logits))
        return np.vstack([1 - probs, probs]).T  # Return as a 2D array with shape (n_samples, 2)

    def predict_uncalibrated_proba(self, X):
        # Predict probabilities using the uncalibrated model
        raw_logits = self._get_logits(X)
        probs = 1 / (1 + np.exp(-raw_logits))
        return np.vstack([1 - probs, probs]).T  # Return as a 2D array with shape (n_samples, 2)

    def evaluate_calibration(self):
        # Evaluate the calibrated model on the validation set
        scaled_logits = self.predict_proba(self.X_validation)
        loss = log_loss(self.y_validation, scaled_logits)
        brier = brier_score_loss(self.y_validation, scaled_logits[:, 1])
        accuracy = np.mean(np.argmax(scaled_logits, axis=1) == self.y_validation)
        classification_report = metrics.classification_report(self.y_validation, np.argmax(scaled_logits, axis=1), target_names=self.label_encoder.classes_)

        print(f'Validation Loss after calibration: {loss:.4f}')
        print(f'Validation Brier score after calibration: {brier:.4f}')
        print(f'Validation Accuracy after calibration: {accuracy:.4f}')
        print(f'Classification Report after calibration: \n{classification_report}')

        self.calibrated_metrics = {
            'loss': loss,
            'brier_score': brier,
            'accuracy': accuracy,
            'classification_report': classification_report
        }

    def evaluate_uncalibrated_model(self):
        # Evaluate the uncalibrated model on the validation set
        raw_logits = self._get_logits(self.X_validation)
        probs = 1 / (1 + np.exp(-raw_logits))
        scaled_logits = np.vstack([1 - probs, probs]).T  # Return as a 2D array with shape (n_samples, 2)

        loss = log_loss(self.y_validation, scaled_logits)
        brier = brier_score_loss(self.y_validation, scaled_logits[:, 1])
        accuracy = np.mean(np.argmax(scaled_logits, axis=1) == self.y_validation)
        classification_report = metrics.classification_report(self.y_validation, np.argmax(scaled_logits, axis=1), target_names=self.label_encoder.classes_)

        print(f'Validation Loss before calibration: {loss:.4f}')
        print(f'Validation Brier score before calibration: {brier:.4f}')
        print(f'Validation Accuracy before calibration: {accuracy:.4f}')
        print(f'Classification Report before calibration: \n{classification_report}')

        self.uncalibrated_metrics = {
            'loss': loss,
            'brier_score': brier,
            'accuracy': accuracy,
            'classification_report': classification_report
        }

    def run_calibration_pipeline(self):
        self.preprocess_data()
        self.calibrate()
        self.evaluate_calibration()
        self.evaluate_uncalibrated_model()

    def save_calibrated_model(self, output_dir):
        # Save both the original model structure and the optimal temperature setting
        artifacts = {
            'model': self.original_model,
            'A': self.A,
            'B': self.B,
            'vectorizer': self.vectorizer,
            'encoder': self.label_encoder
        }
        
        filepath = os.path.join(output_dir, 'calibrated_baseline_meta.pkl')
        with open(filepath, 'wb') as f:
            pickle.dump(artifacts, f)
        print(f'Calibrated model artifacts saved to {filepath}')

    def save_calibration_metrics(self, output_dir, output_format='txt'):
        # Save the evaluation metrics to a text file in the specified output directory
        assert self.calibrated_metrics is not None and self.uncalibrated_metrics is not None, "Please run the calibration pipeline before saving metrics."
        os.makedirs(output_dir, exist_ok=True)
        
        if output_format == 'txt':
            with open(os.path.join(output_dir, 'calibrated_baseline_metrics.txt'), 'w') as f:
                f.write(f'Validation Loss after calibration: {self.calibrated_metrics["loss"]:.4f}\n')
                f.write(f'Validation Brier score after calibration: {self.calibrated_metrics["brier_score"]:.4f}\n')
                f.write(f'Validation Accuracy after calibration: {self.calibrated_metrics["accuracy"]:.4f}\n')
                f.write(f'Classification Report after calibration: \n{self.calibrated_metrics["classification_report"]}\n')

            with open(os.path.join(output_dir, 'uncalibrated_baseline_metrics.txt'), 'w') as f:
                f.write(f'Validation Loss before calibration: {self.uncalibrated_metrics["loss"]:.4f}\n')
                f.write(f'Validation Brier score before calibration: {self.uncalibrated_metrics["brier_score"]:.4f}\n')
                f.write(f'Validation Accuracy before calibration: {self.uncalibrated_metrics["accuracy"]:.4f}\n')
                f.write(f'Classification Report before calibration: \n{self.uncalibrated_metrics["classification_report"]}\n')

        elif output_format == 'json':
            with open(os.path.join(output_dir, 'calibrated_baseline_metrics.json'), 'w') as f:
                json.dump({
                    'validation_loss': self.calibrated_metrics["loss"],
                    'validation_brier_score': self.calibrated_metrics["brier_score"],
                    'validation_accuracy': self.calibrated_metrics["accuracy"],
                    'classification_report': self.calibrated_metrics["classification_report"]
                }, f)
            with open(os.path.join(output_dir, 'uncalibrated_baseline_metrics.json'), 'w') as f:
                json.dump({
                    'validation_loss': self.uncalibrated_metrics["loss"],
                    'validation_brier_score': self.uncalibrated_metrics["brier_score"],
                    'validation_accuracy': self.uncalibrated_metrics["accuracy"],
                    'classification_report': self.uncalibrated_metrics["classification_report"]
                }, f)
        probs_before = self.predict_uncalibrated_proba(self.X_validation)[:, 1]
        probs_after = self.predict_proba(self.X_validation)[:, 1]
        
        preds_before = np.argmax(self.predict_uncalibrated_proba(self.X_validation), axis=1)
        preds_after = np.argmax(self.predict_proba(self.X_validation), axis=1)

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 5))

        cm_before = metrics.confusion_matrix(self.y_validation, preds_before)
        disp_before = metrics.ConfusionMatrixDisplay(confusion_matrix=cm_before, display_labels=self.label_encoder.classes_)
        disp_before.plot(ax=ax1, cmap=plt.cm.Blues)
        ax1.set_title('Confusion Matrix (Before Calibration)')
        ax1.set_xlabel('Predicted Labels')
        ax1.set_ylabel('True Labels')

        
        cm_after = metrics.confusion_matrix(self.y_validation, preds_after)
        disp_after = metrics.ConfusionMatrixDisplay(confusion_matrix=cm_after, display_labels=self.label_encoder.classes_)
        disp_after.plot(ax=ax2, cmap=plt.cm.Blues)
        ax2.set_title('Confusion Matrix (After Calibration)')
        ax2.set_xlabel('Predicted Labels')
        ax2.set_ylabel('True Labels')

        prec_before, recall_before, _ = metrics.precision_recall_curve(self.y_validation, probs_before)
        pr_auc_before = metrics.auc(recall_before, prec_before)

        prec_after, recall_after, _ = metrics.precision_recall_curve(self.y_validation, probs_after)
        pr_auc_after = metrics.auc(recall_after, prec_after)

        no_skill = len(self.y_validation[self.y_validation == 1]) / len(self.y_validation)
        ax3.plot([0, 1], [no_skill, no_skill], 'k--', label=f'No Skill (AUC = {no_skill:.2f})')        
        ax3.plot(recall_before, prec_before, color='tab:red', label=f'Before (AUC = {pr_auc_before:.4f})')
        ax3.plot(recall_after, prec_after, color='tab:green', label=f'After (AUC = {pr_auc_after:.4f})')
        ax3.set_xlabel('Recall')
        ax3.set_ylabel('Precision')
        ax3.set_title('Precision-Recall Curve')
        ax3.legend(loc='lower left')
        ax3.grid(True, linestyle=':')

        n_bins = 10

        true_before, pred_before = calibration_curve(self.y_validation, probs_before, n_bins=n_bins, strategy='uniform')
        cal_true, cal_pred = calibration_curve(self.y_validation, probs_after, n_bins=n_bins, strategy='uniform')

        bin_edges = np.linspace(0, 1, n_bins + 1)
                        
        # np.digitize returns 1-indexed bins; subtract 1 to match 0-indexing
        uncal_bin_idx = np.digitize(probs_before, bin_edges) - 1
        cal_bin_idx = np.digitize(probs_after, bin_edges) - 1
        
        # Clip upper outliers (like exactly 1.0) into the topmost bin
        uncal_bin_idx = np.clip(uncal_bin_idx, 0, n_bins - 1)
        cal_bin_idx = np.clip(cal_bin_idx, 0, n_bins - 1)

        # 3. Calculate sample sizes per bin
        bin_total_uncal = np.bincount(uncal_bin_idx, minlength=n_bins) 
        bin_total_cal = np.bincount(cal_bin_idx, minlength=n_bins) 

        # 4. Filter missing bins to match scikit-learn's shortened output arrays
        uncal_mask = bin_total_uncal > 0
        cal_mask = bin_total_cal > 0

        # 5. Compute accurate ECE weights
        uncal_ece = np.sum(np.abs(true_before - pred_before) * bin_total_uncal[uncal_mask]) / len(self.y_validation)
        cal_ece = np.sum(np.abs(cal_true - cal_pred) * bin_total_cal[cal_mask]) / len(self.y_validation)

        ax4.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
        ax4.plot(pred_before, true_before, marker='o', color='tab:red', label='Before Calibration')
        ax4.plot(cal_pred, cal_true, marker='o', color='tab:green', label='After Calibration')
        ax4.set_xlabel('Mean Predicted Probability')
        ax4.set_ylabel('Fraction of Positives')
        ax4.set_title('Calibration Curve')
        ax4.legend(loc='upper left')
        ax1.text(0.95, 0.05, f'Before ECE = {uncal_ece:.4f}\nAfter ECE = {cal_ece:.4f}', 
                                    verticalalignment='bottom', horizontalalignment='right',
                                    transform=ax1.transAxes,
                                    color='black', fontsize=10)
        ax4.grid(True, linestyle=':')

        plt.tight_layout()

        plt.savefig(os.path.join(output_dir, 'calibration_evaluation_plots.png'), bbox_inches='tight')