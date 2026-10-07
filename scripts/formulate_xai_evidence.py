import os
import json
import pickle
import pandas as pd
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

    unused_data_path = "data/splits/unused_data.csv"
    with open(unused_data_path, "r", encoding="utf-8") as f:
        unused_data = pd.read_csv(f)

    stratified_df = unused_data.groupby('Label', group_keys=False).apply(lambda x: x.sample(n=10, random_state=42))
    stratified_df = stratified_df['Full_Text'] # Only keep the input message column
    input_texts = stratified_df.tolist()

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

    tokenizer = AutoTokenizer.from_pretrained(architecture)

    # Root output directory
    base_output_dir = "xAI_outputs"

    # Loop through each input text individually
    for idx, input_text in enumerate(input_texts):
        print(f"Processing text {idx + 1}/{len(input_texts)}...")

        # Create experiment subfolders
        experiment_dir = os.path.join(base_output_dir, f"experiment_{idx}")
        baseline_output_path = os.path.join(experiment_dir, "baseline")
        bert_output_path = os.path.join(experiment_dir, "bert")
        cleaned_text_path = os.path.join(experiment_dir, "input_text")

        os.makedirs(baseline_output_path, exist_ok=True)
        os.makedirs(bert_output_path, exist_ok=True)
        os.makedirs(cleaned_text_path, exist_ok=True)

        # Initialize explainers with the current single text entry
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
        
        bert_explainer = ExplainableBert(
            binary_model=binary_model,
            multilabel_model=multi_label_model,
            binary_calibrators=scaler,
            multilabel_calibrators=multi_label_calibrators,  
            tokenizer=tokenizer,
            decision_threshold=optimal_threshold,
            input_text=input_text,
            multi_label_names=multi_label_names,
            device=device
        )

        # Run explanations for this text instance
        baseline_explainer.run_explanations()
        bert_explainer.run_explanations()

        # Save files to their respective experiment folders
        baseline_explainer.save_explanations(os.path.join(baseline_output_path, "explanations.txt"))
        bert_explainer.save_explanations(os.path.join(bert_output_path, "explanations.txt"))
        bert_explainer.save_input(os.path.join(cleaned_text_path, "cleaned_input.txt"))

        # Explicitly delete explainers to free up memory before the next iteration
        del baseline_explainer
        del bert_explainer

        gc.collect() 
        if device == 'cuda':
            torch.cuda.empty_cache() 

if __name__ == "__main__":
    main()