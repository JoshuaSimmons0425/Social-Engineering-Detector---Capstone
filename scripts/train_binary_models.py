import os
import pandas as pd

from modelling.binary.baseline.baseline_model import TFIDFBaselineModel

training_set = pd.read_csv('data/splits/training_50_50.csv')
validation_set = pd.read_csv('data/splits/validation_50_50.csv')

text_column = 'Full_Text'
label_column = 'Label'

# Train and evaluate the baseline model

baseline_model = TFIDFBaselineModel(training_set, validation_set, text_column=text_column, label_column=label_column)

baseline_model.run_pipeline()

baseline_model.save_model(output_dir='models/binary/baseline_uncalibrated')

baseline_model.save_metrics(output_dir='experiments/binary/uncalibrated', output_format='txt')

os._exit(0)

