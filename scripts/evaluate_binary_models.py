import sys
import gc
import pickle
import json
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.modelling.binary.bert.bert_calibrator import BinaryBERTCalibrator
from transformers import AutoModelForSequenceClassification
from src.modelling.binary.bert.bert_model import BERTClassifier
from src.datasets.textdatasets import TextDataset
from src.evaluation.binary.baseline_evaluator import BaselineEvaluator
from src.evaluation.binary.bert_evaluator import BinaryBertEvaluator

def main():

    gc.collect()
    torch.cuda.empty_cache()

    test_50_50_path = "data/splits/test_50_50.csv"
    test_80_20_path = "data/splits/test_80_20.csv"

    with open(test_50_50_path, "r", encoding="utf-8") as f:
        test_50_50_df = pd.read_csv(f)

    with open(test_80_20_path, "r", encoding="utf-8") as f:
        test_80_20_df = pd.read_csv(f)

    baseline_artifacts_path = "models/binary/baseline/calibrated/calibrated_baseline_meta.pkl"
    with open(baseline_artifacts_path, "rb") as f:
        baseline_artifacts = pickle.load(f) 

    baseline_model = baseline_artifacts["model"]
    baseline_vectorizer= baseline_artifacts["vectorizer"]
    baseline_A = baseline_artifacts["A"]
    baseline_B = baseline_artifacts["B"]
    baseline_encoder = baseline_artifacts["encoder"]

    text_column, label_column = "Full_Text", "Label"

    baseline_evaluator = BaselineEvaluator(
        model=baseline_model,
        vectorizer=baseline_vectorizer,
        A=baseline_A,
        B=baseline_B,
        encoder=baseline_encoder,
        test_set_50_50=test_50_50_df,
        test_set_80_20=test_80_20_df,
        text_column=text_column,
        label_column=label_column
    )

    baseline_evaluator.run_evaluation()

    calibrated_artifacts_path = "models/binary/bert/calibrated"
    model_state_dict_path = "models/binary/bert/calibrated/model_state_dict.pt"
    bert_encoder_path = "models/binary/bert/uncalibrated/label_encoder.pkl"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(bert_encoder_path, "rb") as f:
        bert_encoder = pickle.load(f)

    architecture = 'answerdotai/ModernBERT-base'

    model = BERTClassifier.load_model(
        path=model_state_dict_path,
        n_classes=1,
        device=device
    )

    scaler, optimal_threshold = BinaryBERTCalibrator.load_calibration_artifacts(
        path=calibrated_artifacts_path,
        device=device
    )

    test_50_50_dataset = TextDataset(test_50_50_df, mode = architecture, max_len=1024, encoder = bert_encoder)
    test_50_50_dataset.preprocess_labels()
    test_80_20_dataset = TextDataset(test_80_20_df, mode = architecture, max_len=1024, encoder = bert_encoder)
    test_80_20_dataset.preprocess_labels()

    test_50_50_loader = DataLoader(test_50_50_dataset, batch_size=4, shuffle=False)
    test_80_20_loader = DataLoader(test_80_20_dataset, batch_size=4, shuffle=False)

    bert_evaluator = BinaryBertEvaluator(
        model=model,
        device=device,
        scaler=scaler,
        threshold=optimal_threshold,
        test_50_50=test_50_50_loader,
        test_80_20=test_80_20_loader
    )

    baseline_target_path = "results/binary/baseline"
    baseline_evaluator.save_metrics(baseline_target_path)
    baseline_evaluator.save_visualizations(baseline_target_path)

    bert_evaluator.run_evaluation()
    bert_target_path = "results/binary/bert"
    bert_evaluator.save_metrics(bert_target_path)

    sys.exit(0)

if __name__ == "__main__":
    main()
