import sys
import gc
import pickle
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from src.datasets.textdatasets import TextDataset
from src.evaluation.binary.baseline_evaluator import BaselineEvaluator

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
    baseline_temperature = baseline_artifacts["temperature"]
    baseline_encoder = baseline_artifacts["encoder"]

    text_column, label_column = "Full_Text", "Label"

    baseline_evaluator = BaselineEvaluator(
        model=baseline_model,
        vectorizer=baseline_vectorizer,
        temperature=baseline_temperature,
        encoder=baseline_encoder,
        test_set_50_50=test_50_50_df,
        test_set_80_20=test_80_20_df,
        text_column=text_column,
        label_column=label_column
    )

    baseline_evaluator.run_evaluation()

    baseline_target_path = "results/binary/baseline"

    baseline_evaluator.save_metrics(baseline_target_path)
    baseline_evaluator.save_visualizations(baseline_target_path)

    sys.exit(0)

if __name__ == "__main__":
    main()
