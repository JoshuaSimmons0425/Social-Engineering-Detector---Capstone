import os
import json
import pickle
import torch
import gc
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.inference.explainable_bert import ExplainableBert
from src.modelling.binary.bert.bert_calibrator import BinaryBERTCalibrator
from src.modelling.binary.bert.bert_model import BERTClassifier

def main():

    gc.collect()
    torch.cuda.empty_cache()

    calibrated_artifacts_path = "models/binary/bert/calibrated"
    model_state_dict_path = "models/binary/bert/calibrated/model_state_dict.pt"
    bert_encoder_path = "models/binary/bert/uncalibrated/label_encoder.pkl"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(bert_encoder_path, "rb") as f:
        bert_encoder = pickle.load(f)

    architecture = 'answerdotai/ModernBERT-base'

    binrary_model = BERTClassifier.load_model(
        path=model_state_dict_path,
        n_classes=1,
        device=device
    )

    scaler, optimal_threshold = BinaryBERTCalibrator.load_calibration_artifacts(
        path=calibrated_artifacts_path,
        device=device
    )

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

    multi_label_model = AutoModelForSequenceClassification.from_pretrained(architecture, num_labels=len(multi_label_names))

    multi_label_calibrator_path = "models/multilabel/bert/calibrated"
    with open(os.path.join(multi_label_calibrator_path, "calibrators.pkl"), "rb") as f:
        multi_label_calibrators = pickle.load(f)

    input_text = """Buck up, your troubles caused by small dimension will soon be over!
Become a lover no woman will be able to resist!
http://whitedone.com/


come. Even as Nazi tanks were rolling down the streets, the dreamersphilosopher or a journalist. He was still not sure.I do the same."""

    explainer = ExplainableBert(
        binary_model=binrary_model,
        multilabel_model=multi_label_model,
        binary_calibrators=scaler,
        multilabel_calibrators=multi_label_calibrators,  
        tokenizer=AutoTokenizer.from_pretrained(architecture),
        input_text=input_text,
        multi_label_names=multi_label_names,
        device=device
    )
    # Add any additional code to load data, make predictions, and generate explanations here

    explainer.run_explanations()

    bert_output_path = "xAI_outputs/bert"
    explainer.save_explanations(os.path.join(bert_output_path, "explanations.txt"))

if __name__ == "__main__":
    main()