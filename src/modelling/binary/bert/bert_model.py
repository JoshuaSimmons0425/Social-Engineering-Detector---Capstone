import os

import torch
import torch.nn as nn
from torch.amp import autocast
from transformers import AutoModel
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn import metrics

class BERTClassifier(nn.Module):
    def __init__(self, n_classes, train_loader, val_loader, pretrained_model_name='answerdotai/ModernBERT-base'):
        super(BERTClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.training_losses = []
        self.validation_losses = []

        # metrics for evaluation
        self.accuracy = None
        self.classification_report = None
        self.confusion_matrix = None
        self.loss_curve = None

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            pooled_output = outputs.last_hidden_state[:, 0, :]
        output = self.drop(pooled_output)
        return self.out(output)

    def train_model(self, device, optimizer, criterion, epochs):

        # Determine device type string safely
        dev_type = 'cuda' if 'cuda' in str(device) else 'cpu'
        # CPU autocast uses bfloat16, GPU uses float16
        amp_dtype = torch.float16 if dev_type == 'cuda' else torch.bfloat16

        if self.training_losses is not None and self.validation_losses is not None:
            self.training_losses = [] # Reset loss lists at the start of training
            self.validation_losses = []
        
        for epoch in range(epochs):
            self.train()
            total_loss = 0
            for batch in self.train_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                optimizer.zero_grad()

                with autocast(device_type=dev_type, dtype=amp_dtype):
                    outputs = self(input_ids, attention_mask)
                    loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            self.training_losses.append(total_loss / len(self.train_loader))

            self.eval()
            validation_loss = 0
            with torch.no_grad():
                for batch in self.val_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)

                    outputs = self(input_ids, attention_mask)
                    loss = criterion(outputs, labels)
                    validation_loss += loss.item()

            self.validation_losses.append(validation_loss / len(self.val_loader))

            print(f'Epoch {epoch + 1}/{epochs}, Training Loss: {total_loss / len(self.train_loader)}, Validation Loss: {validation_loss / len(self.val_loader)}')
            
        print(f'Training Loss: {total_loss / len(self.train_loader)}, Validation Loss: {validation_loss / len(self.val_loader)}')

    def evaluate_model(self, device):
        self.eval()
        all_preds = []
        all_labels = []

        for batch in self.val_loader:
            with torch.no_grad():
                outputs = self(batch['input_ids'].to(device), batch['attention_mask'].to(device))

            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())
                
        self.accuracy = metrics.accuracy_score(all_labels, all_preds)
        self.classification_report = classification_report(all_labels, all_preds, target_names = self.val_loader.dataset.classes)
        
        print(f'Validation Accuracy: {self.accuracy}')
        print(f'Classification Report:\n{self.classification_report}')

    def plot_visuals(self):
        assert self.classification_report is not None, "Please run evaluate_model() before plotting visuals."
        cm = confusion_matrix(self.val_loader.dataset.targets, self.val_loader.dataset.classes.index(self.classification_report.splitlines()[1].split()[0]))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.val_loader.dataset.classes)
        disp.plot(cmap=plt.cm.Blues)
        self.confusion_matrix = disp.figure_

        loss_curve = plt.figure()
        plt.plot(self.training_losses, label='Training Loss')
        plt.plot(self.validation_losses, label='Validation Loss')
        plt.title('Loss Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        self.loss_curve = loss_curve

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def save_metrics(self, path, output_format='txt'):

        os.makedirs(os.path.dirname(path), exist_ok=True)
        if output_format == 'txt':
            with open(path, 'w') as f:
                f.write(f'Validation Accuracy: {self.accuracy}\n')
                f.write(f'Classification Report:\n{self.classification_report}\n')

        elif output_format == 'json':
            import json
            with open(path, 'w') as f:
                json.dump({
                    'validation_accuracy': self.accuracy,
                    'classification_report': self.classification_report
                }, f)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

        assert self.confusion_matrix is not None and self.loss_curve is not None, "Please run plot_visuals() before saving the visuals."
        self.confusion_matrix.savefig(os.path.join(os.path.dirname(path), 'confusion_matrix.png'))
        self.loss_curve.savefig(os.path.join(os.path.dirname(path), 'loss_curve.png'))

        