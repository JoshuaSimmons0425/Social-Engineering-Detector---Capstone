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
strip_pattern = r'Keywords:.*\r?\n'

# Perform any necessary data processing per data source

df_one = engine.clear_irrelevant_columns(df_one, irrelevant_columns)
df_one = engine.strip_with_regex(df_one, text_name, strip_pattern)
df_one = engine.rename_classes(df_one, label_name, label_mapping)
df_one = engine.rename_columns(df_one, label_name, text_name)

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

df_three = pd.read_csv("data/bronze/archive/CEAS_08.csv")
text_name = "body"
subject_name = "subject"
label_name = "label"
irrelevant_columns = ["sender", "receiver", "date", "urls"]
label_mapping = {0: standardised_benign, 1: standardised_malicious}
drop_on_condition = r'CNN.com Daily Top 10'

df_three = engine.clear_irrelevant_columns(df_three, irrelevant_columns)
df_three = engine.concat_subject2body(df_three, subject_name, text_name)
df_three = engine.clear_rows(df_three, text_name, drop_on_condition)
df_three = engine.rename_classes(df_three, label_name, label_mapping)
df_three = engine.rename_columns(df_three, label_name, text_name)

print(df_three.head())

datasets = [df_one, df_two, df_three]

unified_df = engine.unify_datasets(datasets)

unified_df = engine.clear_na_rows(unified_df, engine.text_name)
unified_df = engine.clear_na_rows(unified_df, engine.label_name)
unified_df = engine.mask_money(unified_df, engine.text_name)
unified_df = engine.anonymize_data(unified_df, engine.text_name)
unified_df = engine.deduplicate_rows(unified_df, engine.text_name)

print(unified_df.head())

print(f"Unified dataset rows: {unified_df.shape[0]}")

unified_df.to_csv("data/silver/unified_dataset.csv", index=False)

os._exit(0) # Exit the script after loading the dataset and initializing the DataEngine