import os
import pickle
import json
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.calibration import calibration_curve
from scipy.optimize import  minimize_scalar
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score, classification_report

class TemperatureScaler(nn.Module):
    def __init__(self):
        super(TemperatureScaler, self).__init__()
        # Initialise temperature parameter at 1.5
        self.temperature = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))

    def forward(self, logits):
        # Enforce a strict minimum temperature value to prevent division by zero
        clamped_temp = torch.clamp(self.temperature, min=1e-4)
        return logits / clamped_temp

    def fit(self, logits_list, labels_list, device):
        self.requires_grad_(True)
        
        # Convert collected lists to tensors
        logits_tensor = torch.cat(logits_list, dim=0).to(device)
        labels_tensor = torch.cat(labels_list, dim=0).to(device).float()

        logits = logits_tensor.squeeze(-1)

        def objective(T):
            T = float(T)
            scaled_logits = logits.squeeze(-1) / T
            loss = nn.functional.binary_cross_entropy_with_logits(
                scaled_logits,
                labels_tensor
            )
            return loss.item()

        result = minimize_scalar(
            objective,
            bounds=(0.05, 20.0),
            method='bounded',
            options={'xatol': 1e-6}
        )

        if not result.success:
            raise RuntimeError(
                f"Temperature optimisation failed: {result.message}"
            )

        self.temperature.data = torch.tensor(
            [result.x],
            dtype=torch.float32,
            device=device
        )
        print(f"Optimized temperature: {self.temperature.item():.4f}")

class BinaryBERTCalibrator:
    def __init__(self, model, device, calibration_loader, validation_loader, scaler=None):
        self.model = model
        self.device = device
        self.calibration_loader = calibration_loader
        self.validation_loader = validation_loader
        self.scaler = scaler if scaler is not None else TemperatureScaler().to(device)

        self.uncal_val_probs = None
        self.cal_val_probs = None
        self.val_labels = None

        self.uncalibrated_metrics = {}
        self.calibrated_metrics = {}

        self.diagrams = None
        self.optimal_threshold = 0.5

    def forward(self, input_ids, attention_mask):
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = output.logits

        calibrated_logits = self.scaler(logits)
        return calibrated_logits

    def calibrate(self):
        self.model.eval()
        logits_list = []
        labels_list = []

        with torch.no_grad():
            for batch in self.calibration_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels']

                output = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = output.cpu()

                logits_list.append(logits)
                labels_list.append(labels)

        self.scaler.fit(logits_list, labels_list, self.device)

    def get_probs_from_loader(self, data_loader):
            """
            Helper method to extract both raw and calibrated probabilities 
            over the validation dataset for a fair visualization baseline.
            """
            self.model.eval()
            self.scaler.to(self.device)
            raw_probs_list = []
            cal_probs_list = []
            labels_list = []
    
            with torch.no_grad():
                for batch in data_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].numpy()
    
                    # 1. Fetch raw logit outputs on the active target hardware engine (e.g. cuda)
                    raw_logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    
                    # Compute raw uncalibrated probabilities on GPU, then extract to CPU NumPy array
                    raw_probs = torch.sigmoid(raw_logits.squeeze(-1)).cpu().numpy()

                    # 2. Compute calibrated logit outputs directly on the same hardware engine
                    # Pass the raw GPU logits directly to your GPU-bound scaler
                    cal_logits = self.scaler(raw_logits)
                    
                    # Compute calibrated probabilities on GPU, then extract to CPU NumPy array
                    cal_probs = torch.sigmoid(cal_logits.squeeze(-1)).cpu().numpy()
    
                    raw_probs_list.append(raw_probs)
                    cal_probs_list.append(cal_probs)
                    labels_list.append(labels)
    
            # Force clean 1D tracking arrays 
            raw_probs_flat = np.concatenate(raw_probs_list, axis=0).ravel()
            cal_probs_flat = np.concatenate(cal_probs_list, axis=0).ravel()
            labels_flat = np.concatenate(labels_list, axis=0).ravel()
        
            return raw_probs_flat, cal_probs_flat, labels_flat

    def get_before_and_after_probs(self):
        self.uncal_val_probs, self.cal_val_probs, self.val_labels = self.get_probs_from_loader(self.validation_loader)

    def dynamic_decision_threshold(self, metric = "f1"):

            _, cal_cal_probs, cal_labels = self.get_probs_from_loader(self.calibration_loader)
            
            thresholds = np.linspace(0, 1, 101)
            best_threshold = 0.5
            best_metric_value = -np.inf
    
            for threshold in thresholds:
                preds = (cal_cal_probs >= threshold).astype(int)
                if metric == "f1":
                    metric_value = metrics.f1_score(cal_labels, preds)
                elif metric == "f2":
                    metric_value = metrics.fbeta_score(cal_labels, preds, beta=2)
                elif metric == "f0.5":
                    metric_value = metrics.fbeta_score(cal_labels, preds, beta=0.5)
                else:
                    raise ValueError(f"Unsupported metric: {metric}")
    
                if metric_value > best_metric_value:
                    best_metric_value = metric_value
                    best_threshold = threshold
    
            print(f"Optimal decision threshold for {metric}: {best_threshold:.4f} with {metric} score: {best_metric_value:.4f}")
            self.optimal_threshold = best_threshold
    

    def evaluate(self):
        if self.uncal_val_probs is None or self.cal_val_probs is None or self.val_labels is None:
            self.get_before_and_after_probs()

        self.calibrated_metrics['brier_score'] = brier_score_loss(self.val_labels, self.cal_val_probs)
        self.calibrated_metrics['log_loss'] = log_loss(self.val_labels, self.cal_val_probs)
        self.calibrated_metrics['accuracy'] = accuracy_score(self.val_labels, (self.cal_val_probs >= self.optimal_threshold).astype(int))
        self.calibrated_metrics['classification_report'] = classification_report(self.val_labels, (self.cal_val_probs >= self.optimal_threshold).astype(int), digits = 4,  output_dict=True)

        self.uncalibrated_metrics['brier_score'] = brier_score_loss(self.val_labels, self.uncal_val_probs)
        self.uncalibrated_metrics['log_loss'] = log_loss(self.val_labels, self.uncal_val_probs)
        self.uncalibrated_metrics['accuracy'] = accuracy_score(self.val_labels, (self.uncal_val_probs >= 0.5).astype(int))
        self.uncalibrated_metrics['classification_report'] = classification_report(self.val_labels, (self.uncal_val_probs >= 0.5).astype(int), digits = 4, output_dict=True)

        print("\n --- Performance Comparison Before vs After Calibration ---")
        print(f"Brier Score -> Before: {self.uncalibrated_metrics['brier_score']:.4f} | After: {self.calibrated_metrics['brier_score']:.4f}")
        print(f"Log Loss    -> Before: {self.uncalibrated_metrics['log_loss']:.4f} | After: {self.calibrated_metrics['log_loss']:.4f}")
        print(f"Accuracy    -> Before: {self.uncalibrated_metrics['accuracy']:.4f} | After: {self.calibrated_metrics['accuracy']:.4f}")
        print(f"Classification Report -> Before: {self.uncalibrated_metrics['classification_report']} | After: {self.calibrated_metrics['classification_report']}")

    def plot_visualizations(self, n_bins=10):
            """
            Generates a 1x3 dashboard featuring the Reliability Diagram, 
            PR-AUC curve, and Confusion Matrix comparing performance before/after.
            """

            if self.uncal_val_probs is None or self.cal_val_probs is None or self.val_labels is None:
                self.get_before_and_after_probs()
            # Construct single layout canvas
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 6))
    
            # ==========================================
            # 1. RELIABILITY DIAGRAM
            # ==========================================
            
            # Calculate curve details
            uncal_true, uncal_pred, uncal_bin_idx = calibration_curve(self.val_labels, self.uncal_val_probs, n_bins=n_bins, strategy='uniform')
            cal_true, cal_pred, cal_bin_idx = calibration_curve(self.val_labels, self.cal_val_probs, n_bins=n_bins)

            # Calculate ECE 

            bin_total_uncal = np.bincount(uncal_bin_idx, minlength=n_bins) 
            bin_total_cal = np.bincount(cal_bin_idx, minlength=n_bins) 

            bin_total_uncal_filtered = bin_total_uncal[bin_total_uncal > 0]
            bin_total_cal_filtered = bin_total_cal[bin_total_cal > 0]

            uncal_ece = np.sum(np.abs(uncal_true - uncal_pred) * bin_total_uncal_filtered) / np.sum(bin_total_uncal_filtered)
            cal_ece = np.sum(np.abs(cal_true - cal_pred) * bin_total_cal_filtered) / np.sum(bin_total_cal_filtered)
    
            ax1.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
            ax1.plot(uncal_pred, uncal_true, marker='s', color='tab:red', label='Before Calibration')
            ax1.plot(cal_pred, cal_true, marker='o', color='tab:green', label='After Calibration')
            
            ax1.set_xlabel('Mean Predicted Probability')
            ax1.set_ylabel('Fraction of Positives')
            ax1.set_title('Reliability Diagram')
            ax1.legend(loc='lower right')
            ax1.text(0.95, 0.05, f'Before ECE = {uncal_ece:.4f}\nAfter ECE = {cal_ece:.4f}', 
                     verticalalignment='bottom', horizontalalignment='right',
                     transform=ax1.transAxes,
                     color='black', fontsize=10)
            ax1.grid(True, linestyle=':')
    
            # ==========================================
            # 2. PRECISION-RECALL AUC CURVE
            # ==========================================
            # Uncalibrated curve stats
            uncal_prec, uncal_rec, _ = metrics.precision_recall_curve(self.val_labels, self.uncal_val_probs)
            uncal_auc = metrics.auc(uncal_rec, uncal_prec)
    
            # Calibrated curve stats
            cal_prec, cal_rec, _ = metrics.precision_recall_curve(self.val_labels, self.cal_val_probs)
            cal_auc = metrics.auc(cal_rec, cal_prec)
    
            # Operational baseline (No skill model matches the positive class ratio)
            no_skill = len(self.val_labels[self.val_labels == 1]) / len(self.val_labels)
            ax2.plot([0, 1], [no_skill, no_skill], 'k--', label=f'No Skill (AUC = {no_skill:.2f})')
            
            ax2.plot(uncal_rec, uncal_prec, color='tab:red', label=f'Before (AUC = {uncal_auc:.4f})')
            ax2.plot(cal_rec, cal_prec, color='tab:green', label=f'After (AUC = {cal_auc:.4f})')
            
            ax2.set_xlabel('Recall')
            ax2.set_ylabel('Precision')
            ax2.set_title('Precision-Recall Curve')
            ax2.legend(loc='lower left')
            ax2.grid(True, linestyle=':')
    
            # ==========================================
            # 3. CONFUSION MATRIX (POST-CALIBRATION)
            # ==========================================
            # Assign classification flags using optimized threshold
            preds = (self.cal_val_probs >= self.optimal_threshold).astype(int)
            cm = metrics.confusion_matrix(self.val_labels, preds)
            disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
            # Render visual overlay context straight into third canvas slot
            disp.plot(cmap=plt.cm.Blues, ax=ax3, values_format='d')
            ax3.set_title(f'Confusion Matrix\n(Threshold: {self.optimal_threshold:.2f})')
    
            plt.tight_layout()
            
            # Archive tracking references back to wrapper
            self.diagrams = fig

    def run_calibration_pipeline(self, metric="f1", n_bins=10):
        self.calibrate()
        self.dynamic_decision_threshold(metric=metric)  # Can change the metric as needed
        self.evaluate()
        self.plot_visualizations(n_bins=n_bins)

    def save_calibration_artifacts(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        # Save the temperature scaler and model state dict
        scaler_path = os.path.join(save_dir, 'temperature_scaler.pt')
        torch.save(self.scaler.state_dict(), scaler_path)

        model_path = os.path.join(save_dir, 'model_state_dict.pt')
        torch.save(self.model.state_dict(), model_path)

        decision_threshold_path = os.path.join(save_dir, 'optimal_threshold.json')
        with open(decision_threshold_path, 'w') as f:
            json.dump({'optimal_threshold': self.optimal_threshold}, f)

    @staticmethod
    def load_calibration_artifacts(path, device):
        scaler_path = os.path.join(path, 'temperature_scaler.pt')
        decision_threshold_path = os.path.join(path, 'optimal_threshold.json')

        # Load the temperature scaler
        scaler = TemperatureScaler().to(device)
        scaler_state_dict = torch.load(scaler_path, map_location=device)
        scaler.load_state_dict(scaler_state_dict)

        # Load the optimal decision threshold
        with open(decision_threshold_path, 'r') as f:
            threshold_data = json.load(f)
            optimal_threshold = threshold_data['optimal_threshold']

        return scaler, optimal_threshold

    def save_metrics(self, save_dir, output_format='txt'):

        if self.uncalibrated_metrics is not None and self.calibrated_metrics is not None:
            os.makedirs(save_dir, exist_ok=True)
            metrics_path = os.path.join(save_dir, f'uncalibrated_metrics.{output_format}')
            if output_format == 'txt':
                with open(metrics_path, 'w') as f:
                    for key, value in self.uncalibrated_metrics.items():
                        f.write(f"{key}: {value}\n")
            elif output_format == 'json':
                with open(metrics_path, 'w') as f:
                    json.dump(self.uncalibrated_metrics, f)
            print(f"Saved uncalibrated metrics to {metrics_path}")
            calibrated_metrics_path = os.path.join(save_dir, f'calibrated_metrics.{output_format}')
            if output_format == 'txt':
                with open(calibrated_metrics_path, 'w') as f:
                    for key, value in self.calibrated_metrics.items():
                        f.write(f"{key}: {value}\n")
            elif output_format == 'json':
                with open(calibrated_metrics_path, 'w') as f:
                    json.dump(self.calibrated_metrics, f)
            print(f"Saved calibrated metrics to {calibrated_metrics_path}")

        if self.diagrams is not None:
            os.makedirs(save_dir, exist_ok=True)
            viz_path = os.path.join(save_dir, 'calibration_visualizations.png')
            self.diagrams.savefig(viz_path)
            print(f"Saved calibration visualizations to {viz_path}")
        else:
            print("No visualizations to save. Please run plot_visualizations() first.")