import sys
import pickle
import pandas as pd
from src.modelling.binary.baseline.baseline_calibrator import TFIDFBaselineCalibrator

def main():

    with open('models/binary/baseline/uncalibrated/uncalibrated_baseline_meta.pkl', 'rb') as f:
        uncalibrated_artifacts = pickle.load(f)

    model = uncalibrated_artifacts['model']
    vectorizer = uncalibrated_artifacts['vectorizer']
    encoder = uncalibrated_artifacts['encoder']

    calibration_set = pd.read_csv('data/splits/calibration_80_20.csv')
    validation_set = pd.read_csv('data/splits/validation_80_20.csv')

    calibrator = TFIDFBaselineCalibrator(model, vectorizer, encoder, calibration_set, validation_set, text_column='Full_Text', label_column='Label')
    calibrator.run_calibration_pipeline()

    calibrator.save_calibrated_model(output_dir='models/binary/baseline/calibrated')
    calibrator.save_calibration_metrics(output_dir='experiments/binary/baseline/calibrated', output_format='txt')

    sys.exit(0)

if __name__ == "__main__":
    main()
