import os
import pickle
import pickle
import json
import numpy as np
from sklearn import metrics
from scipy.optimize import minimize
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
        self.temperature = temperature

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

        def objective(T):
            # Prevent division by zero
            T = max(T[0], 1e-5) 
            
            # Apply temperature scaling to logits
            scaled_logits = raw_logits / T
            
            # Convert scaled logits to probabilities using sigmoid
            probs = 1 / (1 + np.exp(-scaled_logits))
            
            # Return log loss against true labels
            return log_loss(self.y_calibration, probs)

        # 3. Optimize the temperature parameter starting at T=1.0
        result = minimize(objective, x0=[1.0], method='Nelder-Mead')
        self.temperature = max(result.x[0], 1e-5)
        print(f"Optimized Temperature: {self.temperature:.4f}")

    def predict_proba(self, X):
        # Predict probabilities using the calibrated model
        raw_logits = self._get_logits(X)
        scaled_logits = raw_logits / self.temperature
        probs = 1 / (1 + np.exp(-scaled_logits))
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
            'temperature': self.temperature,
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
