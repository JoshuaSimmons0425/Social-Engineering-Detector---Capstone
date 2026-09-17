import sys
import gc
import joblib
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from src.modelling.multilabel.baseline.ml_baseline_calibrator import BaselineMultiCalibrator 
from src.modelling.multilabel.bert.ml_bert_calibrator import MultiBERTCalibrator
from src.datasets.textdatasets import MultiLabelTextDataset

def main():
    gc.collect()
    torch.cuda.empty_cache()

    baseline_artifacts_path = 'models/multilabel/baseline/uncalibrated/multi_label_model.pkl'

    with open(baseline_artifacts_path, "rb") as f:
        artifacts = joblib.load(f)

    model = artifacts['model']
    vectorizer = artifacts['vectorizer']

    with open('data/splits/calibration_80_20.csv', 'r', encoding='utf-8', errors='replace') as f:
        calibration_set = pd.read_csv(f)

    with open('data/splits/validation_80_20.csv', 'r', encoding='utf-8', errors='replace') as f:
        validation_set = pd.read_csv(f)

    text_column = "Full_Text"
    all_labels = [
        'urgency_label', 
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

    baseline_calibrator = BaselineMultiCalibrator(model, vectorizer, calibration_set, validation_set, text_column, all_labels)
    baseline_calibrator.run_pipeline()

    mode = 'answerdotai/ModernBERT-base'
    bert_model = AutoModelForSequenceClassification.from_pretrained(mode, num_labels=len(all_labels))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_model.to(device)

    calibration_set = MultiLabelTextDataset(calibration_set, mode, 1024, all_labels)
    validation_set = MultiLabelTextDataset(validation_set, mode, 1024, all_labels)

    calibration_loader = DataLoader(calibration_set, batch_size=4, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=4, shuffle=False)

    bert_calibrator = MultiBERTCalibrator(bert_model, device, all_labels, calibration_loader, validation_loader)
    bert_calibrator.run_pipeline()

    baseline_calibrator.save_metrics("experiments/multilabel/baseline/calibrated")
    baseline_calibrator.save_calibration_artifacts("models/multilabel/baseline/calibrated")

    bert_calibrator.save_calibration_results("experiments/multilabel/bert/calibrated")
    bert_calibrator.save_artifacts("models/multilabel/bert/calibrated")

    sys.exit(1)

if __name__ == '__main__':
    main()