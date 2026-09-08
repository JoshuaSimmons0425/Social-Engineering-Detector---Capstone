import sys
import gc
import pickle
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from src.modelling.binary.bert.bert_model import BERTClassifier
from src.modelling.binary.baseline.baseline_model import TFIDFBaselineModel
from src.modelling.binary.baseline.baseline_calibrator import TFIDFBaselineCalibrator
from src.modelling.binary.bert.bert_calibrator import BinaryBERTCalibrator
from src.datasets.textdatasets import TextDataset

def main():

    gc.collect()
    torch.cuda.empty_cache()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, vectorizer, encoder = TFIDFBaselineModel.load_model(model_path='models/binary/baseline/uncalibrated/uncalibrated_baseline_meta.pkl')

    calibration_set = pd.read_csv('data/splits/calibration_80_20.csv')
    validation_set = pd.read_csv('data/splits/validation_80_20.csv')

    baseline_calibrator = TFIDFBaselineCalibrator(model, vectorizer, encoder, calibration_set, validation_set, text_column='Full_Text', label_column='Label')
    baseline_calibrator.run_calibration_pipeline()

    bert_model = AutoModelForSequenceClassification.from_pretrained('answerdotai/ModernBERT-base', num_labels=1)

    label_encoder = pickle.load(open('models/binary/bert/uncalibrated/label_encoder.pkl', 'rb'))
    bert_model_path = 'models/binary/bert/uncalibrated/uncalibrated_bert_model.pt'

    bert_model = BERTClassifier.load_model(
        path=bert_model_path,
        n_classes=1,
        device=device
    )

    calibration_dataset = TextDataset(calibration_set, mode="answerdotai/ModernBERT-base", max_len=1024, encoder=label_encoder)
    calibration_dataset.preprocess_labels(training=False)
    
    validation_dataset = TextDataset(validation_set, mode="answerdotai/ModernBERT-base", max_len=1024, encoder=label_encoder)
    validation_dataset.preprocess_labels(training=False)

    calibration_loader = DataLoader(calibration_dataset, batch_size=4, shuffle=False)
    validation_loader = DataLoader(validation_dataset, batch_size=4, shuffle=False)

    bert_calibrator = BinaryBERTCalibrator(model=bert_model,
                                           calibration_loader=calibration_loader,
                                           validation_loader=validation_loader,
                                           device=device)
    bert_calibrator.run_calibration_pipeline(metric="f1")

    baseline_calibrator.save_calibrated_model(output_dir='models/binary/baseline/calibrated')
    baseline_calibrator.save_calibration_metrics(output_dir='experiments/binary/baseline/calibrated', output_format='txt')
    bert_calibrator.save_calibration_artifacts(save_dir='models/binary/bert/calibrated')
    bert_calibrator.save_metrics(save_dir='experiments/binary/bert/calibrated', output_format='txt')

    sys.exit(0)

if __name__ == "__main__":
    main()
