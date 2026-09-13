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
    def __init__(self, model, device, calibration_loader, validation_loader, scaler=None):
        super(MultiBERTCalibrator, self).__init__()
        self.model = model
        self.device = device
        self.calibration_loader = calibration_loader
        self.validation_loader = validation_loader
        self.scaler = scaler if scaler is not None else PlattScaler().to(device)