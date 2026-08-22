import sys
import pandas as pd
from src.modelling.binary.baseline.baseline_model import TFIDFBaselineModel

def main():

    with open('data/splits/training_50_50.csv', 'r', encoding='utf-8', errors='replace') as f:
        training_set = pd.read_csv(f)
    with open('data/splits/validation_50_50.csv', 'r', encoding='utf-8', errors='replace') as f:
        validation_set = pd.read_csv(f)

    text_column = 'Full_Text'
    label_column = 'Label'

    # Train and evaluate the baseline model

    baseline_model = TFIDFBaselineModel(training_set, validation_set, text_column=text_column, label_column=label_column)
    baseline_model.run_pipeline()
    baseline_model.save_model(output_dir='models/binary/baseline/uncalibrated')
    baseline_model.save_metrics(output_dir='experiments/binary/baseline/uncalibrated', output_format='txt')

    sys.exit(0)

if __name__ == "__main__":
    main()