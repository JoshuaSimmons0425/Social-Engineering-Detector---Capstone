import os
import pickle
import json
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from scipy.optimize import minimize
from sklearn.metrics import log_loss, brier_score_loss

class TemperatureScaler(nn.Module):
    def __init__(self):
        super(TemperatureScaler, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature

    def fit(self, logits, labels):
        nll_criterion = nn.BCEWithLogitsLoss()
        logits_tensor = torch.tensor(logits, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.float32)

        def loss_fn(temp):
            temp = torch.tensor(temp, requires_grad=True)
            scaled_logits = logits_tensor / temp
            loss = nll_criterion(scaled_logits, labels_tensor)
            return loss.item()

        result = minimize(loss_fn, x0=[1.5], bounds=[(1e-6, None)], method='L-BFGS-B')
        self.temperature.data = torch.tensor(result.x[0], dtype=torch.float32)


class BinaryBERTCalibrator:
    def __init__(self, model, device, scaler=None):
        self.model = model
        self.device = device
        self.scaler = scaler if scaler is not None else TemperatureScaler().to(device)

        self.brier_score = None
        self.log_loss = None
        
    def calibrate(self, logits, labels):
        self.scaler.fit(logits, labels)

    def predict(self, logits):
        logits_tensor = torch.tensor(logits, dtype=torch.float32).to(self.device)
        scaled_logits = self.scaler(logits_tensor)
        probs = torch.sigmoid(scaled_logits).cpu().detach().numpy()
        return probs

    def evaluate(self, logits, labels):
        probs = self.predict(logits)
        self.brier_score = brier_score_loss(labels, probs)
        self.log_loss = log_loss(labels, probs)
        return self.brier_score, self.log_loss