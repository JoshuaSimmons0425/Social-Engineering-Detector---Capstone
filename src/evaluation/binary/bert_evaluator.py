import os
import json
import torch
import pickle
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.calibration import calibration_curve


class BinaryBertEvaluator:
    def __init__(self, model, device, temperature, threshold, test_50_50, test_80_20):
        self.model = model
        self.device = device
        self.temperature = temperature
        self.threshold = threshold
        self.test_50_50 = test_50_50
        self.test_80_20 = test_80_20

        self._50_50_uncal_probs = []
        self._80_20_uncal_probs = []
        self._80_20_cal_probs = []

        self._50_50_labels = []
        self._80_20_labels = []

        self.uncalibrated_results = {}
        self.calibrated_results = {}
        
        self.uncalibrated_diagrams = None
        self.calibrated_diagrams = None

    def evaluate_uncalibrated(self):

        self.model.eval()
        self.model.to(self.device)

        for batch in self.test_50_50:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).long() # Use 0.5 as the baseline threshold to evaluate model discriminative ability

            self._50_50_labels.extend(labels.cpu().numpy())
            self._50_50_uncal_probs.extend(probs.cpu().numpy())
            

        self.uncalibrated_results['50_50'] = {
            'accuracy': metrics.accuracy_score(self._50_50_labels, (torch.tensor(self._50_50_uncal_probs) > 0.5).long().numpy()),
            'precision': metrics.precision_score(self._50_50_labels, (torch.tensor(self._50_50_uncal_probs) > 0.5).long().numpy()),
            'recall': metrics.recall_score(self._50_50_labels, (torch.tensor(self._50_50_uncal_probs) > 0.5).long().numpy()),
            'f1': metrics.f1_score(self._50_50_labels, (torch.tensor(self._50_50_uncal_probs) > 0.5).long().numpy())
        }

        
        all_preds_80_20 = []

        for batch in self.test_80_20:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).long() # Use 0.5 as the baseline threshold to evaluate model discriminative ability

            self._80_20_labels.extend(labels.cpu().numpy())
            self._80_20_uncal_probs.extend(probs.cpu().numpy())
            all_preds_80_20.extend(preds.cpu().numpy())

        self.uncalibrated_results['80_20'] = {
            'accuracy': metrics.accuracy_score(self._80_20_labels, all_preds_80_20),
            'precision': metrics.precision_score(self._80_20_labels, all_preds_80_20),
            'recall': metrics.recall_score(self._80_20_labels, all_preds_80_20),
            'f1': metrics.f1_score(self._80_20_labels, all_preds_80_20),
            'brier_score': metrics.mean_squared_error(self._80_20_labels, self._80_20_uncal_probs),
            'log_loss': metrics.log_loss(self._80_20_labels, self._80_20_uncal_probs)
        }

    def evaluate_calibrated(self):

        self.model.eval()
        self.model.to(self.device)

        all_labels_80_20 = []
        all_probs_80_20 = []
        all_preds_80_20 = []

        for batch in self.test_80_20:

            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                scaled_logits = self.temperature(logits)
                probs = torch.sigmoid(scaled_logits.squeeze(-1).cpu().numpy())
                preds = (probs > self.threshold).long() # Use 0.5 as the baseline threshold to evaluate model discriminative ability

            all_labels_80_20.extend(labels.cpu().numpy())
            all_probs_80_20.extend(probs.cpu().numpy())
            all_preds_80_20.extend(preds.cpu().numpy())

        self.calibrated_results['80_20'] = {
            'accuracy': metrics.accuracy_score(all_labels_80_20, all_preds_80_20),
            'precision': metrics.precision_score(all_labels_80_20, all_preds_80_20),
            'recall': metrics.recall_score(all_labels_80_20, all_preds_80_20),
            'f1': metrics.f1_score(all_labels_80_20, all_preds_80_20),
            'brier_score': metrics.mean_squared_error(all_labels_80_20, all_probs_80_20),
            'log_loss': metrics.log_loss(all_labels_80_20, all_probs_80_20)
        }

    def plot_uncal_visualisations(self):

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 6))

        # Visualisation logic for the uncalibrated results, involving plotting the roc_curve and confusion matrix with the 50_50 probs

        preds = (self._50_50_uncal_probs >= 0.5).astype(int)
        cm = metrics.confusion_matrix(self._50_50_labels, preds)
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax1)
        ax1.set_title("Confusion Matrix (50_50 Uncalibrated)")

        fpr, tpr, _ = metrics.roc_curve(self._50_50_labels, self._50_50_uncal_probs)
        ax2.plot(fpr, tpr, marker='.')
        ax2.set_title("ROC Curve (50_50 Uncalibrated)")
        ax2.set_xlabel("False Positive Rate")
        ax2.set_ylabel("True Positive Rate")
        ax2.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Diagonal line for random classifier

        self.uncalibrated_diagrams = fig

    def plot_cal_visualisations(self):

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 6))

        # Visualisation logic for the calibrated results, involving plotting the roc_curve and confusion matrix with the 80_20 probs

        # Calculate curve details
        n_bins = 10  # Number of bins for the calibration curve
        uncal_true, uncal_pred = calibration_curve(self._80_20_labels, self._80_20_uncal_probs, n_bins=n_bins)
        cal_true, cal_pred = calibration_curve(self._80_20_labels, self._80_20_cal_probs, n_bins=n_bins)
    
        ax1.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
        ax1.plot(uncal_pred, uncal_true, marker='s', color='tab:red', label='Before Calibration')
        ax1.plot(cal_pred, cal_true, marker='o', color='tab:green', label='After Calibration')
            
        ax1.set_xlabel('Mean Predicted Probability')
        ax1.set_ylabel('Fraction of Positives')
        ax1.set_title('Reliability Diagram')
        ax1.legend(loc='lower right')
        ax1.grid(True, linestyle=':')
    
        # ==========================================
        # 2. PRECISION-RECALL AUC CURVE
        # ==========================================
        # Uncalibrated curve stats
        uncal_prec, uncal_rec, _ = metrics.precision_recall_curve(self._80_20_labels, self._80_20_uncal_probs)
        uncal_auc = metrics.auc(uncal_rec, uncal_prec)
    
        # Calibrated curve stats
        cal_prec, cal_rec, _ = metrics.precision_recall_curve(self._80_20_labels, self._80_20_cal_probs)
        cal_auc = metrics.auc(cal_rec, cal_prec)

        # Operational baseline (No skill model matches the positive class ratio)
        no_skill = len(self._80_20_labels[self._80_20_labels == 1]) / len(self._80_20_labels)
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
        preds = (self._80_20_cal_probs >= self.threshold).astype(int)
        cm = metrics.confusion_matrix(self._80_20_labels, preds)
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
        # Render visual overlay context straight into third canvas slot
        disp.plot(cmap=plt.cm.Blues, ax=ax3, values_format='d')
        ax3.set_title(f'Confusion Matrix\n(Threshold: {self.threshold:.2f})')

        plt.tight_layout()
        
        # Archive tracking references back to wrapper
        self.calibrated_diagrams = fig

    def run_evaluation(self):
        self.evaluate_uncalibrated()
        self.evaluate_calibrated()
        self.plot_uncal_visualisations()
        self.plot_cal_visualisations()

    def save_metrics(self, filepath, output_format = 'txt'):

        with open(filepath, 'w') as f:
            if output_format == 'txt':
                f.write("Uncalibrated Results (80_20):\n")
                for key, value in self.uncalibrated_results['80_20'].items():
                    f.write(f"{key}: {value}\n")
                f.write("\nCalibrated Results (80_20):\n")
                for key, value in self.calibrated_results['80_20'].items():
                    f.write(f"{key}: {value}\n")
            elif output_format == 'json':
                import json
                json.dump({
                    'uncalibrated_results': self.uncalibrated_results['80_20'],
                    'calibrated_results': self.calibrated_results['80_20']
                }, f)