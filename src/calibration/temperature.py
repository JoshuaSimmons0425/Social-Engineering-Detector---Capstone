import torch
from torch import nn
from scipy.optimize import minimize_scalar

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

        # With an 80/20 split, this evaluates to 3200 / 800 = 4.0
        num_benign = (labels_tensor == 0).sum().item()
        num_malicious = (labels_tensor == 1).sum().item()
        
        # Guard against zero-division just in case, default to 4.0 if empty
        malicious_weight_factor = num_benign / max(num_malicious, 1) if num_malicious > 0 else 4.0
        
        # Convert to a tensor matching the data's device
        pos_weight_tensor = torch.tensor([malicious_weight_factor], dtype=torch.float32, device=device)

        def objective(T):
            T = float(T)
            scaled_logits = logits.squeeze(-1) / T
            loss = nn.functional.binary_cross_entropy_with_logits(
                scaled_logits,
                labels_tensor,
                pos_weight=pos_weight_tensor
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