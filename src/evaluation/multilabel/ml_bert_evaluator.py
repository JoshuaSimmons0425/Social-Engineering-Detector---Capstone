import os
import pickle
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy.optimize import minimize
from sklearn.calibration import calibration_curve
from sklearn.metrics import log_loss, brier_score_loss

class MLBertEvaluator:
    def __init__(self, model, device,platt_scalers, test_loader_50_50, test_loader_80_20, all_labels):
        self.model = model
        self.device = device
        self.platt_scalers = platt_scalers
        self.test_loader_50_50 = test_loader_50_50
        self.test_loader_80_20 = test_loader_80_20
        self.all_labels = all_labels

        self.discrimination_results = {}
        self.uncalibrated_results = {}
        self.calibrated_results = {}

    def forward(self, input_ids, attention_mask):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

    def _get_probs_from_loader(self, loader):
        """Helper to collect all model raw logits and true targets from a given DataLoader.

        Returns two dictionaries mapping label -> torch.Tensor.
        """
        self.model.eval()

        all_logits = {label: [] for label in self.all_labels}
        all_targets = {label: [] for label in self.all_labels}

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                # Batch single forward pass for efficiency across all labels
                outputs = self.model(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                logits = outputs.logits  # Shape: (batch_size, num_labels)

                for i, label in enumerate(self.all_labels):
                    all_logits[label].append(logits[:, i].cpu())
                    all_targets[label].append(batch["labels"][:, i].cpu())

        # Concatenate lists into single tensors per label
        for label in self.all_labels:
            all_logits[label] = torch.cat(all_logits[label]).to(self.device)
            all_targets[label] = torch.cat(all_targets[label]).to(self.device)

        return all_logits, all_targets

    def _calculate_calibration_metrics(self, targets, probabilities):
        """Calculates standard classification and calibration metrics."""

        # Expected Calibration Error (ECE) implementation
        prob_true, prob_pred = calibration_curve(
            targets, probabilities, n_bins=10, strategy="uniform"
        )
        ece = np.mean(np.abs(prob_true - prob_pred))

        return {
            "brier_score": float(brier_score_loss(targets, probabilities)),
            "log_loss": float(log_loss(targets, probabilities, labels=[0, 1])),
            "ece": float(ece),
        }

    def evaluate_calibration(self):
        """Finds optimal decision thresholds based on F1-score and calculates metric summaries."""

        all_logits, all_targets = self._get_probs_from_loader(self.test_loader_80_20)

        for label in self.all_labels:
            # Native tensor mapping using the stored PyTorch PlattScaler modules
            raw_label_logits = all_logits[label]
            scaler_module = self.platt_scalers[label]
            
            with torch.no_grad():
                cal_label_logits = scaler_module(raw_label_logits)

            # Move back safely down to CPU Numpy arrays for evaluation processing
            targets = all_targets[label].cpu().numpy()
            raw_probs = 1 / (1 + np.exp(-raw_label_logits.cpu().numpy()))
            cal_probs = 1 / (1 + np.exp(-cal_label_logits.cpu().numpy()))

            # 3. Calculate metrics
            self.uncalibrated_results[label] = self._calculate_calibration_metrics(
                targets, raw_probs
            )
            self.calibrated_results[label] = self._calculate_calibration_metrics(
                targets, cal_probs
            )
            print(f"Evaluated continuous calibrated metrics for label: {label}")

    def evaluate_uncalibration(self, device):
        eval_probs = []
        eval_true = []

        print("Running uncalibrated evaluation with 0.5 decision threshold...")
        for batch in self.test_loader_50_50:
            with torch.no_grad():
                output = self.forward(batch['input_ids'].to(device), batch['attention_mask'].to(device))
                probs = torch.sigmoid(output)
            eval_probs.extend(probs.cpu().numpy())
            eval_true.extend(batch['labels'].cpu().numpy())

        eval_probs = np.vstack(eval_probs)
        eval_true = np.vstack(eval_true)
        preds = (eval_probs >= 0.5).astype(int)

        for i, label in enumerate(self.all_labels):
            self.discrimination_results[label] = {}
            
            accuracy = metrics.accuracy_score(eval_true[:, i], preds[:, i])
            self.discrimination_results[label]['accuracy'] = float(accuracy)
            self.discrimination_results[label]['classification_report'] = metrics.classification_report(
                eval_true[:, i], preds[:, i], digits=4, output_dict=True, zero_division=0
            )
            self.discrimination_results[label]['classification_report_str'] = metrics.classification_report(
                eval_true[:, i], preds[:, i], digits=4, zero_division=0
            )

            print(f'Validation Accuracy for {label} (50:50): {accuracy:.4f}')

        macro_f1_score = metrics.f1_score(eval_true, preds, average='macro', zero_division=0)
        self.discrimination_results['macro_f1_score'] = float(macro_f1_score)
        print(f"Overall BERT Macro F1 Score (50:50): {macro_f1_score:.4f}\n")


    def run_pipeline(self, device):
        self.evaluate_uncalibration(device)
        self.evaluate_calibration()
        

    def save_results(self, save_path):
        """Saves discrimination and calibration profiles down into identical TXT and JSON formats."""
        os.makedirs(save_path, exist_ok=True)

        # 1. Package unified data into structured JSON
        combined_json_results = {
            "discrimination_50_50_set": {
                "macro_f1_score": self.discrimination_results.get('macro_f1_score', None),
                "labels": {
                    lbl: {
                        "accuracy": self.discrimination_results[lbl]["accuracy"],
                        "classification_report": self.discrimination_results[lbl]["classification_report"]
                    } for lbl in self.all_labels if lbl in self.discrimination_results
                }
            },
            "calibration_80_20_set": {
                "uncalibrated_profiles": self.uncalibrated_results,
                "calibrated_profiles": self.calibrated_results
            }
        }

        json_file_path = os.path.join(save_path, "bert_evaluation_metrics.json")
        with open(json_file_path, "w") as f:
            json.dump(combined_json_results, f, indent=4)

        # 2. Output detailed structural text logs report
        txt_file_path = os.path.join(save_path, "bert_evaluation_report.txt")
        with open(txt_file_path, "w") as f:
            f.write("==================================================\n")
            f.write("         BERT MULTILABEL EVALUATION REPORT        \n")
            f.write("==================================================\n\n")

            f.write("--------------------------------------------------\n")
            f.write("1. DISCRIMINATION RESULTS (50:50 Balanced Test Set)\n")
            f.write("--------------------------------------------------\n")
            f.write(f"Macro F1 Score: {self.discrimination_results.get('macro_f1_score', 0.0):.5f}\n\n")
            
            for label in self.all_labels:
                if label in self.discrimination_results:
                    f.write(f"Label Name: {label}\n")
                    f.write(f"Accuracy  : {self.discrimination_results[label]['accuracy']:.5f}\n")
                    f.write("Classification Matrix Profile:\n")
                    f.write(f"{self.discrimination_results[label]['classification_report_str']}\n")
                    f.write("-" * 40 + "\n")

            f.write("\n--------------------------------------------------\n")
            f.write("2. PROBABILITY CALIBRATION RESULTS (80:20 Test Set)\n")
            f.write("--------------------------------------------------\n")
            
            for label in self.all_labels:
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

        print(f"[INFO] BERT Evaluation results successfully saved down to: '{save_path}'")