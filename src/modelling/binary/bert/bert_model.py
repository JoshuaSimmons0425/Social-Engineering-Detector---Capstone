import os
import json
import torch
import torch.nn as nn
from transformers import AutoModel
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn import metrics

class BERTClassifier(nn.Module):
    def __init__(self, n_classes, train_loader, val_loader, optimizer, criterion, epochs, learning_rate, pretrained_model_name='answerdotai/ModernBERT-base'):
        super(BERTClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.epochs = epochs
        self.learning_rate = learning_rate

        self.training_losses = []
        self.validation_losses = []
        self.all_preds = []
        self.all_labels = []

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
        last_hidden = outputs.last_hidden_state 
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        sum_embeddings = torch.sum(last_hidden * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        pooled_output = sum_embeddings / sum_mask
        
        output = self.drop(pooled_output)
        return self.out(output)
    
    def train_model(self, device):
        if self.optimizer == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        criterion = self.criterion
        epochs = self.epochs

        # Determine device type string safely
        dev_type = 'cuda' if 'cuda' in str(device) else 'cpu'
        # CPU autocast uses bfloat16, GPU uses float16
        amp_dtype = torch.float16 if dev_type == 'cuda' else torch.bfloat16

        scaler = torch.amp.GradScaler(device=dev_type) if dev_type == 'cuda' else None

        accumulation_steps = 4

        if self.training_losses is not None and self.validation_losses is not None:
            self.training_losses = [] # Reset loss lists at the start of training
            self.validation_losses = []

        print(f"Starting training on {dev_type.upper()} for {epochs} epochs with optimizer {self.optimizer} and learning rate {self.learning_rate}" + "\n")
        
        for epoch in range(epochs):
            self.train()
            total_loss = 0
            optimizer.zero_grad()

            for batch_idx, batch in enumerate(self.train_loader):

                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                with torch.amp.autocast(device_type=dev_type, dtype=amp_dtype):
                    outputs = self(input_ids, attention_mask)
                    loss = criterion(outputs, labels)   

                    loss = loss / accumulation_steps  # Normalize loss for gradient accumulation

                # 4. Scale loss and step using the scaler
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

                total_loss += loss.item() * accumulation_steps  # Multiply back to get the original loss value

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
        self.all_preds = []
        self.all_labels = []

        for batch in self.val_loader:
            with torch.no_grad():
                outputs = self(batch['input_ids'].to(device), batch['attention_mask'].to(device))

            preds = torch.argmax(outputs, dim=1)
            self.all_preds.extend(preds.cpu().numpy())
            self.all_labels.extend(batch['labels'].cpu().numpy())
                
        self.accuracy = metrics.accuracy_score(self.all_labels, self.all_preds)
        class_names = getattr(self.val_loader.dataset, 'classes', None)
        self.classification_report = classification_report(self.all_labels, self.all_preds, target_names=class_names)
        
        print(f'Validation Accuracy: {self.accuracy}')
        print(f'Classification Report:\n{self.classification_report}')

    def run_pipeline(self, device):
        self.to(device)
        self.train_model(device)
        self.evaluate_model(device)
        self.plot_visuals()

    def plot_visuals(self):
        assert hasattr(self, 'all_labels'), "Please run evaluate_model() before plotting visuals."
        cm = confusion_matrix(self.all_labels, self.all_preds)
        class_names = getattr(self.val_loader.dataset, 'classes', None)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        fig_cm, ax = plt.subplots(figsize=(6, 6))
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        self.confusion_matrix = fig_cm

        loss_curve = plt.figure()
        plt.plot(self.training_losses, label='Training Loss')
        plt.plot(self.validation_losses, label='Validation Loss')
        plt.title('Loss Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        self.loss_curve = loss_curve

    def save_model(self, path):
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def save_metrics(self, path, output_format='txt'):

        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
        if output_format == 'txt':
            with open(path, 'w') as f:
                f.write(f'Validation Accuracy: {self.accuracy}\n')
                f.write(f'Classification Report:\n{self.classification_report}\n')

        elif output_format == 'json':
            with open(path, 'w') as f:
                json.dump({
                    'validation_accuracy': self.accuracy,
                    'classification_report': self.classification_report
                }, f, indent=4)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

        assert hasattr(self, 'confusion_matrix') and hasattr(self, 'loss_curve'), "Please run plot_visuals() before saving the visuals."
        
        output_dir = os.path.dirname(path) if os.path.dirname(path) else '.'
        self.confusion_matrix.savefig(os.path.join(output_dir, 'confusion_matrix.png'), bbox_inches='tight')
        self.loss_curve.savefig(os.path.join(output_dir, 'loss_curve.png'), bbox_inches='tight')

        plt.close(self.confusion_matrix)
        plt.close(self.loss_curve)

        