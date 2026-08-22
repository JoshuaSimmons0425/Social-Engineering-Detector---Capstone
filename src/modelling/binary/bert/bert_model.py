import torch
import torch.nn as nn
from transformers import AutoModel

class BERTClassifier(nn.Module):
    def __init__(self, n_classes, train_loader, val_loader, pretrained_model_name='answerdotai/ModernBERT-base'):
        super(BERTClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.train_loader = train_loader
        self.val_loader = val_loader

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
        
        for epoch in range(epochs):
            self.train()
            total_loss = 0
            for batch in self.train_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                optimizer.zero_grad()
                outputs = self(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

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

            print(f'Epoch {epoch + 1}/{epochs}, Training Loss: {total_loss / len(self.train_loader)}, Validation Loss: {validation_loss / len(self.val_loader)}')
            
        return total_loss / len(self.train_loader), validation_loss / len(self.val_loader)