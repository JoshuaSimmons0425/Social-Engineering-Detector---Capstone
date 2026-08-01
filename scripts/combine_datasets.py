import os

import pandas as pd
from src.datasets.dataengine import DataEngine

df = pd.read_csv("data/bronze/phishing/phishing_legit_dataset_KD_10000.csv")  # Load your dataset here
engine = DataEngine(df)

os._exit(0) # Exit the script after loading the dataset and initializing the DataEngine