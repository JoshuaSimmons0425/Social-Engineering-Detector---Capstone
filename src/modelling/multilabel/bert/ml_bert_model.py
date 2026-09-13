import os
import json
import time
import torch
import numpy as np
import torch.nn as nn
from transformers import AutoModel
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn import metrics

class MultiLabelBERTClassifier(nn.Module):
    def __init__(self, n_labels, all_labels, train_loader, val_loader, optimizer, epochs, learning_rate, pretrained_model_name = "answerdotai/ModernBERT-base"):
        super(MultiLabelBERTClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_labels)
        self.all_labels = all_labels
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.criterion = None

        self.training_losses = []
        self.validation_losses = []
        self.eval_probs = []
        self.eval_true = []

        self.accuracy = None
        self.classification_report = None
        self.diagrams = None

    def forward(self, input_ids, attention_mask):

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        last_hidden = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        sum_embeddings = torch.sum(last_hidden * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask

        output = self.drop(pooled_output)
        return self.out(output)

    def calculate_pos_weights(self, device):
        # Calculate positive weights for each label based on the training dataset for cost-sensitive learning
        raw_matrix = self.train_loader.dataset.labels 
        labels_matrix = torch.tensor(raw_matrix, dtype=torch.float32)
        num_samples = labels_matrix.shape[0]
        pos_counts = torch.sum(labels_matrix, dim=0)
        neg_counts = num_samples - pos_counts

        pos_counts = torch.clamp(pos_counts, min=1)
        neg_counts = torch.clamp(neg_counts, min=1)

        pos_weights = neg_counts / pos_counts

        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights.to(device))

    def train_model(self, device):

        self.to(device)

        if self.optimizer == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        assert self.criterion is not None, "Criterion (loss function) must be set before training. Call calculate_pos_weights() first."

        criterion = self.criterion
        epochs = self.epochs

        dev_type = 'cuda' if 'cuda' in str(device) else 'cpu'

        amp_dtype = torch.float16 if dev_type == 'cuda' else torch.bfloat16

        scaler = torch.amp.GradScaler(enabled=(dev_type == 'cuda'))
        accumulation_steps = 4

        if self.training_losses is not None and self.validation_losses is not None:
            self.training_losses = [] # Reset loss lists at the start of training
            self.validation_losses = []

        print(f"Starting training on {dev_type.upper()} for {epochs} epochs with optimizer {self.optimizer} and learning rate {self.learning_rate}" + "\n")

        time_per_epoch = []

        for epoch in range(epochs):
            self.train()
            start_time = time.time()
            total_loss = 0
            optimizer.zero_grad()

            for batch_idx, batch in enumerate(self.train_loader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                with torch.amp.autocast(device_type=dev_type, dtype=amp_dtype):
                    outputs = self(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(outputs, labels)
                    loss = loss / accumulation_steps  # Normalize the loss for gradient accumulation

                if scaler is not None:
                    scaler.scale(loss).backward()
                    if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()

                else:
                    loss.backward()
                    if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                        optimizer.step()
                        optimizer.zero_grad()

                total_loss += loss.item() * accumulation_steps  # Scale the loss back up for reporting

            self.training_losses.append(total_loss / len(self.train_loader))
            epoch_time = (time.time() - start_time) / 60 # time in minutes
            time_per_epoch.append(epoch_time)

            self.eval()
            validation_loss = 0
            with torch.no_grad():
                for val_batch in self.val_loader:
                    val_input_ids = val_batch['input_ids'].to(device)
                    val_attention_mask = val_batch['attention_mask'].to(device)
                    val_labels = val_batch['labels'].to(device)

                    val_outputs = self(val_input_ids, val_attention_mask)
                    val_loss = criterion(val_outputs, val_labels)
                    validation_loss += val_loss.item()

            self.validation_losses.append(validation_loss / len(self.val_loader))

            print(f"Epoch [{epoch + 1}/{epochs}], Training Loss: {self.training_losses[-1]:.4f}, Validation Loss: {self.validation_losses[-1]:.4f}, Time: {epoch_time:.2f} minutes")
            print(f"Average Time per Epoch: {sum(time_per_epoch) / len(time_per_epoch):.2f} minutes" + "\n")

        print(f"Final Training Loss: {self.training_losses[-1]:.4f}, Final Validation Loss: {self.validation_losses[-1]:.4f}" + "\n")

    def evaluate_model(self, device):
        self.eval()
        self.to(device)

        self.eval_probs = []
        self.eval_true = []

        print("Running validation evaluation with 0.5 decision threshold...")
        for batch in self.val_loader:
            with torch.no_grad():
                output = self(batch['input_ids'].to(device), batch['attention_mask'].to(device))
                probs = torch.sigmoid(output)

            self.eval_probs.extend(probs.cpu().numpy())
            self.eval_true.extend(batch['labels'].cpu().numpy())

        self.eval_probs = np.vstack(self.eval_probs)
        self.eval_true = np.vstack(self.eval_true)

        preds = (self.eval_probs >= 0.5).astype(int)
        self.accuracy = metrics.accuracy_score(self.eval_true, preds)
        class_names = [f"Label {i}" for i in self.all_labels]
        self.classification_report = classification_report(self.eval_true, preds, target_names=class_names, zero_division=0)

        print(f"Validation Accuracy: {self.accuracy:.4f}")
        print("Classification Report:")
        print(self.classification_report)
    
    def plot_loss_curves(self):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.training_losses, label='Training Loss')
        ax.plot(self.validation_losses, label='Validation Loss')
        ax.set_title('Loss Curves of Multi-Label BERT Classifier')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid()

        fig.tight_layout()
        self.diagrams = fig

    def run_pipeline(self, device):
        self.to(device)
        self.calculate_pos_weights(device)
        self.train_model(device)
        self.evaluate_model(device)
        self.plot_loss_curves()

    def save_model(self, save_path):
        os.makedirs(save_path, exist_ok=True)
        model_save_path = os.path.join(save_path, "multi_label_uncal_model.pt")
        torch.save(self.state_dict(), model_save_path)

    def save_metrics(self, save_path, output_format="txt"):
        os.makedirs(save_path, exist_ok=True)
        if output_format == "json":
            metrics_save_path = os.path.join(save_path, "metrics.json")
            metrics_data = {
                "accuracy": self.accuracy,
                "classification_report": self.classification_report
            }
            with open(metrics_save_path, 'w') as f:
                json.dump(metrics_data, f, indent=4)

        elif output_format == "txt":
            metrics_save_path = os.path.join(save_path, "metrics.txt")
            with open(metrics_save_path, 'w') as f:
                f.write(f"Validation Accuracy: {self.accuracy:.4f}\n")
                f.write("Classification Report:\n")
                f.write(self.classification_report)

        self.diagrams.savefig(os.path.join(save_path, "loss_curves.png"))