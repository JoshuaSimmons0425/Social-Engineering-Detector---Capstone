import os

import pandas as pd
from src.datasets.dataengine import DataEngine

engine = DataEngine()

df = pd.read_csv("data/bronze/phishing/phishing_legit_dataset_KD_10000.csv") 
text_label = "label"
text_name = "text"
irrelevant_columns = ["phishing_type", "severity", "confidence"]

# Perform any necessary data processing or initialization

df = engine.clear_irrelevant_columns(df, irrelevant_columns)
df = engine.rename_columns(df, text_label, text_name)

print(df.head())

os._exit(0) # Exit the script after loading the dataset and initializing the DataEngine