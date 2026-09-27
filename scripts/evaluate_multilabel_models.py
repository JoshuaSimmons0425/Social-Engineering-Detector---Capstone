import os
import sys
import torch
import gc
import joblib
import pandas as pd
import json
import pickle
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

    baseline_evaluator.run_evaluation()

    baseline_results_save_path = "results/multilabel/baseline/baseline_evaluation_results"
    baseline_evaluator.save_results(baseline_results_save_path)

    sys.exit(0)

if __name__ == "__main__":
    main()