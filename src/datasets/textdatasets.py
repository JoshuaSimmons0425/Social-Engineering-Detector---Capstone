import os
import pickle
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class TextDataset(Dataset):

    def __init__(self, dataset, mode, max_len, encoder=None):
        self.texts = dataset['Full_Text'].values
        self.labels = dataset['Label'].values
        self.tokenizer = AutoTokenizer.from_pretrained(mode)
        self.max_len = max_len
        self.encoder = encoder if encoder is not None else LabelEncoder()

    def preprocess_labels(self, training=True):
        if training:
            self.labels = self.encoder.fit_transform(self.labels)
        else:
            self.labels = self.encoder.transform(self.labels)
        return self.encoder

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

    def save_encoder(self, output_dir):
        # Save the fitted label encoder to the specified output directory
        os.makedirs(output_dir, exist_ok=True)
        encoder_path = os.path.join(output_dir, 'label_encoder.pkl')
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.encoder, f)
        print(f'Label encoder saved to {encoder_path}')
