import gc
from logging import config
import sys
import torch 
import yaml
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
import pandas as pd
from src.datasets.textdatasets import MultiLabelTextDataset

from src.modelling.multilabel.baseline.ml_baseline_model import MultiLabelTFIDFModel
from src.modelling.multilabel.bert.ml_bert_model import MultiLabelBERTClassifier

def main():

    gc.collect()
    torch.cuda.empty_cache()

    with open("data/splits/training_50_50.csv", 'r', encoding='utf-8', errors='replace') as f:
        training_set = pd.read_csv(f)

    with open("data/splits/validation_50_50.csv", 'r', encoding='utf-8', errors='replace') as f:
        validation_set = pd.read_csv(f)

    text_column = 'Full_Text'
    label_columns = ['urgency_label', 
                     'fear_label', 
                     'authority_label', 
                     'reciprocity_label', 
                     'curiosity_label', 
                     'pretexting_label', 
                     'promotional_label',
                     'transactional_label',
                     'reminder_label',
                     'personal_label'
                     ]

    baseline_model = MultiLabelTFIDFModel(training_set, validation_set, text_column, label_columns)
    baseline_model.run_pipeline()

    
    with open('config/technique_model.yaml', 'r') as f:
        config = yaml.safe_load(f)

    batch_size = config['data']['batch_size']
    tokenizer_name = config['data']['tokenizer_name']
    max_len = config['data']['max_length']

    training_data = MultiLabelTextDataset(training_set, mode=tokenizer_name, max_len=max_len, label_columns=label_columns)
    validation_data = MultiLabelTextDataset(validation_set, mode=tokenizer_name, max_len=max_len, label_columns=label_columns)

    training_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False)

    learning_rate = float(config['bert_model']['learning_rate'])
    n_classes = config['bert_model']['n_classes']
    epochs = config['bert_model']['epochs']
    optimizer = config['bert_model']['optimizer']
    model_name = config['bert_model']['model_name']

    multi_label_model = MultiLabelBERTClassifier(
        n_labels=n_classes,
        all_labels=label_columns,
        train_loader=training_loader,
        val_loader=validation_loader,
        optimizer=optimizer,
        epochs=epochs,
        learning_rate=learning_rate,
        pretrained_model_name=model_name
    )

    multi_label_model.run_pipeline(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            
    save_baseline_model_path = "models/multilabel/baseline/uncalibrated"
    save_baseline_metrics_path = "experiments/multilabel/baseline/uncalibrated"

    baseline_model.save_model(save_baseline_model_path)
    baseline_model.save_metrics(save_baseline_metrics_path)

    save_multilabel_bert_model_path = "models/multilabel/bert/uncalibrated"
    save_multilabel_bert_metrics_path = "experiments/multilabel/bert/uncalibrated"

    multi_label_model.save_model(save_multilabel_bert_model_path)
    multi_label_model.save_metrics(save_multilabel_bert_metrics_path)

    sys.exit(0)

if __name__ == "__main__":
    main()