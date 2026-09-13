import torch
import numpy as np
from torch import nn
from scipy.optimize import minimize

class PlattScaler(nn.Module):
    def __init__(self):
        super(PlattScaler, self).__init__()
        # Initial values: A = 1, B = 0
        self.A = nn.Parameter(torch.tensor(1.0))
        self.B = nn.Parameter(torch.tensor(0.0))

    def forward(self, logits):
        return self.A * logits + self.B

    def fit(self, logits_list, labels_list, device):

        # Combine calibration data
        logits = torch.cat(logits_list, dim=0).to(device).squeeze(-1)
        labels = torch.cat(labels_list, dim=0).to(device).float()

        # SciPy operates on CPU/NumPy
        logits_np = logits.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()

        def objective(params):

            A, B = params

            scaled_logits = A * logits_np + B

            # Numerically stable BCE
            loss = (
                np.maximum(scaled_logits, 0)
                - scaled_logits * labels_np
                + np.log1p(np.exp(-np.abs(scaled_logits)))
            )

            return np.mean(loss)

        result = minimize(
            objective,
            x0=np.array([1.0, 0.0]),
            method="L-BFGS-B",
            bounds=[
                (1e-4, 50.0),
                (-20.0, 20.0),
            ],  # Keep parameters physically stable
        )

        if not result.success:
            raise RuntimeError(
                f"Platt scaling optimization failed: {result.message}"
            )

        # Store optimized parameters on the correct device
        with torch.no_grad():
            self.A.copy_(
                torch.tensor(result.x[0], dtype=torch.float32, device=device)
            )
            self.B.copy_(
                torch.tensor(result.x[1], dtype=torch.float32, device=device)
            )

        print("Optimized Platt parameters:")
        print(f"A = {self.A.item():.6f}")
        print(f"B = {self.B.item():.6f}")
        print(f"Calibration NLL = {result.fun:.6f}")

