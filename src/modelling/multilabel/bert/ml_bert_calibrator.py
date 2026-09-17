import os
import pickle
import json
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.calibration import calibration_curve
from scipy.optimize import minimize, minimize_scalar
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score, classification_report
from src.calibration.platt_scaler import PlattScaler

class MultiBERTCalibrator:
    def __init__(self, model, device, all_labels, calibration_loader, validation_loader):
        super(MultiBERTCalibrator, self).__init__()
        self.model = model
        self.all_labels = all_labels
        self.device = device
        self.calibration_loader = calibration_loader
        self.validation_loader = validation_loader
        self.calibrators = None

        self.calibrated_metrics = {}
        self.uncalibrated_metrics = {}

        self.raw_logits = {label: [] for label in self.all_labels}
        self.calibrated_logits = {label: [] for label in self.all_labels}
        self.val_labels = {label: None for label in self.all_labels}

    def forward(self, input_ids, attention_mask):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        return logits

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

    def calibrate(self):
        self.calibrators = {label: PlattScaler().to(self.device) for label in self.all_labels}
        for label in self.all_labels:
            logits_list = []
            targets_list = []

            print(f"Calibrating label: {label}")
            with torch.no_grad():
                for batch in self.calibration_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    targets = batch['labels'][:, self.all_labels.index(label)].to(self.device)
                    logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits[:, self.all_labels.index(label)]
                    
                    logits_list.append(logits)
                    targets_list.append(targets)

            # Pass the lists directly without pre-concatenating them
            self.calibrators[label].fit(logits_list, targets_list, self.device)

        val_raw_logits, val_targets = self._get_probs_from_loader(
            self.validation_loader
        )
        self.val_labels = val_targets
        for label in self.all_labels:
            self.raw_logits[label] = val_raw_logits[label].cpu().numpy()

            # Apply the calibrated PlattScaler to get calibrated logits/probabilities
            with torch.no_grad():
                # Some PyTorch-based PlattScalers return calibrated logits or calibrated probabilities
                # We assume it acts on the tensor and outputs calibrated scales
                cal_logits = self.calibrators[label](val_raw_logits[label]).unsqueeze(-1)
                self.calibrated_logits[label] = cal_logits.cpu().numpy()

    def _calculate_metrics(self, targets, probabilities):
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
        for label in self.all_labels:
            # 1. Convert raw and calibrated logits to probabilities via Sigmoid
            raw_probs = 1 / (1 + np.exp(-self.raw_logits[label]))
            cal_probs = 1 / (1 + np.exp(-self.calibrated_logits[label]))
            targets = self.val_labels[label].cpu().numpy()

            # 3. Calculate metrics
            self.uncalibrated_metrics[label] = self._calculate_metrics(
                targets, raw_probs
            )
            self.calibrated_metrics[label] = self._calculate_metrics(
                targets, cal_probs
            )
            print(f"Evaluated continuous metrics for label: {label}")

    def run_pipeline(self):
        self.calibrate()
        self.evaluate_calibration()

    def save_artifacts(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        # Save weights
        with open(os.path.join(output_dir, "calibrators.pkl"), "wb") as f:
            pickle.dump(self.calibrators, f)

        model_path = os.path.join(output_dir, 'model_state_dict.pt')
        torch.save(self.model.state_dict(), model_path)
        
    def save_calibration_results(self, output_dir, output_format="txt"):
        """Saves evaluation metrics cleanly to disk."""
        os.makedirs(output_dir, exist_ok=True)

        if output_format == "json":
            results = {
                "uncalibrated_metrics": self.uncalibrated_metrics,
                "calibrated_metrics": self.calibrated_metrics,
            }
            with open(os.path.join(output_dir, "results.json"), "w") as f:
                json.dump(results, f, indent=4)

        elif output_format == "txt":
            with open(os.path.join(output_dir, "results.txt"), "w") as f:
                f.write(
                    "==================================================\n"
                )
                f.write("CALIBRATION LAYER PROBABILITY EVALUATIONS\n")
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

        print(f"Calibration results saved to {output_dir}")