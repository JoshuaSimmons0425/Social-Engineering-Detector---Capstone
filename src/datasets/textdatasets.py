import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class TextDataset(Dataset):

    def __init__(self, dataset, mode, max_len):
        self.texts = dataset['Full_Text'].values
        self.labels = dataset['Label'].values
        self.encoder = LabelEncoder()
        self.tokenizer = AutoTokenizer.from_pretrained(mode)
        self.max_len = max_len
        self.attention_mask = None

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        if self.tokenizer:

            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_len,
                return_token_type_ids=False,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            )

            return {
                'text': text,
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)
            }

        return {'text': text, 'labels': torch.tensor(label, dtype=torch.long)}
