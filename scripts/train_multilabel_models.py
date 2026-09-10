import gc
import sys
import torch 
import yaml
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
import pandas as pd
from src.datasets.textdatasets import TextDataset

from src.modelling.multilabel.baseline.ml_baseline_model import MultiLabelTFIDFModel

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

    model = MultiLabelTFIDFModel(training_set, validation_set, text_column, label_columns)
    model.run_pipeline()

    save_model_path = "models/multilabel/baseline/uncalibrated"
    save_metrics_path = "experiments/multilabel/baseline/uncalibrated"

    model.save_model(save_model_path)
    model.save_metrics(save_metrics_path)

    sys.exit(0)

if __name__ == "__main__":
    main()