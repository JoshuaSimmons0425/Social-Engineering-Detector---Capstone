import os

import pandas as pd
from src.datasets.dataengine import DataEngine

standardised_benign = "Benign"
standardised_malicious = "Malicious"
engine = DataEngine()

df_one = pd.read_csv("data/bronze/phishing/phishing_legit_dataset_KD_10000.csv") 
label_name = "label"
text_name = "text"
irrelevant_columns = ["phishing_type", "severity", "confidence"]
label_mapping = {0: standardised_benign, 1: standardised_malicious}
drop_on_condition = r'Keywords:.*\r?\n'

# Perform any necessary data processing or initialization
df_one = engine.clear_irrelevant_columns(df_one, irrelevant_columns)
df_one = engine.strip_with_regex(df_one, text_name, drop_on_condition)
df_one = engine.rename_classes(df_one, label_name, label_mapping)
df_one = engine.rename_columns(df_one, label_name, text_name)

df_one.to_csv("temp.csv", index=False)

print(df_one.head())

df_two = pd.read_csv("data/bronze/sms_spam/Dataset_5971.csv")
text_name = "TEXT"
label_name = "LABEL"
irrelevant_columns = ["URL", "EMAIL", "PHONE"]
label_mapping = {"ham": standardised_benign, "Smishing": standardised_malicious}
drop_row_condition = "spam"

df_two = engine.clear_irrelevant_columns(df_two, irrelevant_columns)
df_two = engine.clear_rows(df_two, label_name, drop_row_condition)
df_two = engine.rename_classes(df_two, label_name, label_mapping)
df_two = engine.rename_columns(df_two, label_name, text_name)

print(df_two.head())

os._exit(0) # Exit the script after loading the dataset and initializing the DataEngine