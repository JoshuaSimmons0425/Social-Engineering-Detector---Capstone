import os
import sys
import torch
import gc
import joblib
import pandas as pd
import json
import pickle
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from src.datasets.textdatasets import MultiLabelTextDataset
from src.evaluation.multilabel.ml_bert_evaluator import MLBertEvaluator
from src.evaluation.multilabel.ml_baseline_evaluator import MLBaselineEvaluator


def main():

    gc.collect()
    torch.cuda.empty_cache()

    test_50_50_path = "data/splits/test_50_50.csv"
    test_80_20_path = "data/splits/test_80_20.csv"

    with open(test_50_50_path, "r", encoding="utf-8") as f:
        test_50_50_df = pd.read_csv(f)

    with open(test_80_20_path, "r", encoding="utf-8") as f:
        test_80_20_df = pd.read_csv(f)

    baseline_model_path = "models/multilabel/baseline/uncalibrated/multi_label_model.pkl"

    with open(baseline_model_path, "rb") as f:
        baseline_model_artifacts = joblib.load(f)

    baseline_calibration_artifacts_path = "models/multilabel/baseline/calibrated"

    with open(os.path.join(baseline_calibration_artifacts_path, "calibrated_estimators.pkl"), "rb") as f:
        calibrated_estimators = pickle.load(f)

    with open(os.path.join(baseline_calibration_artifacts_path, "platt_parameters.json"), "r") as f:
        platt_parameters = json.load(f)

    model = baseline_model_artifacts["model"]
    vectorizer = baseline_model_artifacts["vectorizer"]

    text_column = "Full_Text"
    label_columns = [
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


    baseline_evaluator = MLBaselineEvaluator(
        model=model,
        estimators=calibrated_estimators,
        vectorizer=vectorizer,
        platt_params=platt_parameters,
        test_set_50_50=test_50_50_df,
        test_set_80_20=test_80_20_df,
        text_column=text_column,
        label_columns=label_columns,
    )

    ml_bert_artifacts_path = "models/multilabel/bert/calibrated"

    with open(os.path.join(ml_bert_artifacts_path, "calibrators.pkl"), "rb") as f:
        platt_parameters = pickle.load(f)

    with open(os.path.join(ml_bert_artifacts_path, "model_state_dict.pt"), "rb") as f:
        model_state_dict = torch.load(f)

    mode = 'answerdotai/ModernBERT-base'

    model = AutoModelForSequenceClassification.from_pretrained(mode, num_labels=len(label_columns))
    model.load_state_dict(model_state_dict)

    test_50_50_dataset = MultiLabelTextDataset(
        dataset=test_50_50_df,
        mode=mode,
        max_len = 1024,
        label_columns=label_columns
    )

    test_80_20_dataset = MultiLabelTextDataset(
        dataset=test_80_20_df,
        mode=mode,
        max_len = 1024,
        label_columns=label_columns
    )

    test_loader_50_50 = DataLoader(test_50_50_dataset, batch_size=8, shuffle=False)
    test_loader_80_20 = DataLoader(test_80_20_dataset, batch_size=8, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ml_bert_evaluator = MLBertEvaluator(
        model=model,
        device=device,
        platt_scalers=platt_parameters,
        test_loader_50_50=test_loader_50_50,
        test_loader_80_20=test_loader_80_20,
        all_labels=label_columns
    )

    baseline_evaluator.run_evaluation()
    ml_bert_evaluator.run_pipeline(device=device)

    baseline_results_save_path = "results/multilabel/baseline"
    baseline_evaluator.save_results(baseline_results_save_path)

    bert_results_save_path = "results/multilabel/bert"
    ml_bert_evaluator.save_results(bert_results_save_path)

    sys.exit(0)

if __name__ == "__main__":
    main()