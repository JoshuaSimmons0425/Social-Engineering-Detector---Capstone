import os
import json
import pickle
import torch
import gc
import yaml
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.inference.explainable_bert import ExplainableBert
from src.inference.explanable_baseline import ExplainableBaseline
from src.modelling.binary.bert.bert_calibrator import BinaryBERTCalibrator
from src.modelling.binary.bert.bert_model import BERTClassifier

def main():

    gc.collect()
    torch.cuda.empty_cache()

    config_file_path = "config/risk_assessment.yaml"
    with open(config_file_path, "r") as f:
        config = yaml.safe_load(f) 
    input_text = config["xAI_input"]["input_text"]

    multi_label_names = [
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
    

    baseline_binary_artifacts_path = "models/binary/baseline/calibrated"
    with open(os.path.join(baseline_binary_artifacts_path, "calibrated_baseline_meta.pkl"), "rb") as f:
        baseline_meta = pickle.load(f)

    binary_baseline_model = baseline_meta['model']
    baseline_binary_A = baseline_meta['A']
    baseline_binary_B = baseline_meta['B']
    baseline_vectorizer = baseline_meta['vectorizer']
    baseline_binary_threshold = baseline_meta['threshold']

    binary_platt_scalers = {'A': baseline_binary_A, 'B': baseline_binary_B}

    baseline_multilabel_artifacts_path = "models/multilabel/baseline/calibrated"
    baseline_multilabel_model_path = "models/multilabel/baseline/uncalibrated"

    with open(os.path.join(baseline_multilabel_model_path, "multi_label_model.pkl"), "rb") as f:
        baseline_multilabel_model = joblib.load(f)

    with open(os.path.join(baseline_multilabel_artifacts_path, "calibrated_estimators.pkl"), "rb") as f:
        baseline_multilabel_estimators = pickle.load(f)

    with open(os.path.join(baseline_multilabel_artifacts_path, "platt_parameters.json"), "r") as f:
        ml_platt_scalers = json.load(f)

    baseline_explainer = ExplainableBaseline(
        binary_model=binary_baseline_model,
        multi_label_model=baseline_multilabel_model,
        estimators=baseline_multilabel_estimators,
        binary_calibrators=binary_platt_scalers,
        multilabel_calibrators=ml_platt_scalers,
        vectorizer=baseline_vectorizer,
        decision_threshold=baseline_binary_threshold,
        input_text=input_text,
        multi_label_names=multi_label_names,
    )

    calibrated_artifacts_path = "models/binary/bert/calibrated"
    model_state_dict_path = "models/binary/bert/calibrated/model_state_dict.pt"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    architecture = 'answerdotai/ModernBERT-base'

    binary_model = BERTClassifier.load_model(
        path=model_state_dict_path,
        n_classes=1,
        device=device
    )

    scaler, optimal_threshold = BinaryBERTCalibrator.load_calibration_artifacts(
        path=calibrated_artifacts_path,
        device=device
    )

    multi_label_calibrator_path = "models/multilabel/bert/calibrated"
    with open(os.path.join(multi_label_calibrator_path, "calibrators.pkl"), "rb") as f:
        multi_label_calibrators = pickle.load(f)

    multi_label_state_dict_path = "models/multilabel/bert/calibrated/model_state_dict.pt"

    with open(multi_label_state_dict_path, "rb") as f:
        state_dict = torch.load(f, map_location=device)
        
    multi_label_model = AutoModelForSequenceClassification.from_pretrained(architecture, num_labels=len(multi_label_names))
    multi_label_model.load_state_dict(state_dict)

    bert_explainer = ExplainableBert(
        binary_model=binary_model,
        multilabel_model=multi_label_model,
        binary_calibrators=scaler,
        multilabel_calibrators=multi_label_calibrators,  
        tokenizer=AutoTokenizer.from_pretrained(architecture),
        decision_threshold=optimal_threshold,  # Add a default decision threshold for binary classification
        input_text=input_text,
        multi_label_names=multi_label_names,
        device=device
    )
    # Add any additional code to load data, make predictions, and generate explanations here

    baseline_explainer.run_explanations()
    bert_explainer.run_explanations()

    baseline_output_path = "xAI_outputs/baseline"
    bert_output_path = "xAI_outputs/bert"
    cleaned_text_path = "xAI_outputs/input_text"
    baseline_explainer.save_explanations(os.path.join(baseline_output_path, "explanations.txt"))
    bert_explainer.save_explanations(os.path.join(bert_output_path, "explanations.txt"))
    bert_explainer.save_input(os.path.join(cleaned_text_path, "cleaned_input.txt"))

if __name__ == "__main__":
    main()