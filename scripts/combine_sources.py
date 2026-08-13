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
df_one = engine.deduplicate_rows(df_one, text_name)
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
df_two = engine.deduplicate_rows(df_two, text_name)
df_two = engine.clear_rows(df_two, label_name, drop_row_condition)
df_two = engine.rename_classes(df_two, label_name, label_mapping)
df_two = engine.rename_columns(df_two, label_name, text_name)

print(df_two.head())

df_three = pd.read_csv("data/bronze/smishtank/Dataset_10191.csv")
text_name = "TEXT"
label_name = "LABEL"
irrelevant_columns = ["URL", "EMAIL", "PHONE"]
label_mapping = {"ham": standardised_benign, "smishing": standardised_malicious}
drop_row_condition = "spam"

df_three = engine.clear_irrelevant_columns(df_three, irrelevant_columns)
df_three = engine.deduplicate_rows(df_three, text_name)
df_three = engine.clear_rows(df_three, label_name, drop_row_condition)
df_three = engine.rename_classes(df_three, label_name, label_mapping)
df_three = engine.rename_columns(df_three, label_name, text_name)

print(df_three.head())

df_four = pd.read_csv("data/bronze/archive/CEAS_08.csv")
text_name = "body"
subject_name = "subject"
label_name = "label"
irrelevant_columns = ["sender", "receiver", "date", "urls"]
label_mapping = {0: standardised_benign, 1: standardised_malicious}
drop_on_condition = r'CNN\.com Daily Top 10|CNN Alerts: My Custom Alert'

df_four = engine.clear_irrelevant_columns(df_four, irrelevant_columns)
df_four = engine.deduplicate_rows(df_four, text_name)
df_four = engine.clear_rows(df_four, subject_name, drop_on_condition, regex=True)
df_four = engine.concat_subject2body(df_four, subject_name, text_name)
df_four = engine.rename_classes(df_four, label_name, label_mapping)
df_four = engine.rename_columns(df_four, label_name, text_name)

print(df_four.head())

df_five = pd.read_csv("data/bronze/archive/SpamAssasin.csv")
text_name = "body"
subject_name = "subject"
label_name = "label"
irrelevant_columns = ["sender", "receiver", "date", "urls"]
label_mapping = {0: standardised_benign, 1: standardised_malicious}

df_five = engine.clear_irrelevant_columns(df_five, irrelevant_columns)
df_five = engine.deduplicate_rows(df_five, text_name)
df_five = engine.concat_subject2body(df_five, subject_name, text_name)
df_five = engine.rename_classes(df_five, label_name, label_mapping)
df_five = engine.rename_columns(df_five, label_name, text_name)

print(df_five.head())

df_six = pd.read_csv("data/bronze/archive/Ling.csv")

text_name = "body"
subject_name = "subject"
label_name = "label"
label_mapping = {0: standardised_benign, 1: standardised_malicious}

df_six = engine.deduplicate_rows(df_six, text_name)
df_six = engine.concat_subject2body(df_six, subject_name, text_name)
df_six = engine.rename_classes(df_six, label_name, label_mapping)
df_six = engine.rename_columns(df_six, label_name, text_name)

df_seven = pd.read_csv("data/bronze/phishing/Nazario.csv")
body_name = "body"
subject_name = "subject"
label_name = "label"
irrelevant_columns = ["sender", "receiver", "date", "urls"]
label_mapping = {0: standardised_benign, 1: standardised_malicious}

df_seven = engine.clear_irrelevant_columns(df_seven, irrelevant_columns)
df_seven = engine.deduplicate_rows(df_seven, body_name)
df_seven = engine.concat_subject2body(df_seven, subject_name, body_name)
df_seven = engine.rename_classes(df_seven, label_name, label_mapping)
df_seven = engine.rename_columns(df_seven, label_name, body_name)

print(df_seven.head())

df_eight = pd.read_csv("data/bronze/phishing/Nigerian_Fraud.csv")
body_name = "body"
subject_name = "subject"
label_name = "label"
irrelevant_columns = ["sender", "receiver", "date", "urls"]
label_mapping = {0: standardised_benign, 1: standardised_malicious}

df_eight = engine.clear_irrelevant_columns(df_eight, irrelevant_columns)
df_eight = engine.deduplicate_rows(df_eight, body_name)
df_eight = engine.concat_subject2body(df_eight, subject_name, body_name)
df_eight = engine.rename_classes(df_eight, label_name, label_mapping)
df_eight = engine.rename_columns(df_eight, label_name, body_name)

print(df_eight.head())

df_nine = pd.read_csv("data/bronze/sms+spam+collection/final_dataset.csv")
body_name = "text"
label_name = "label"
label_mapping = {0: standardised_benign, 2: standardised_malicious}
drop_row_condition = 1

df_nine = engine.clear_rows(df_nine, label_name, drop_row_condition)
df_nine = engine.deduplicate_rows(df_nine, body_name)
df_nine = engine.rename_classes(df_nine, label_name, label_mapping)
df_nine = engine.sample_dataset(df_nine, label_name, sample_size=10000, ratio=0.15)
df_nine = engine.rename_columns(df_nine, label_name, body_name)

print(df_nine.head())

datasets = [df_one, df_two, df_three, df_four, df_five, df_six, df_seven, df_eight, df_nine]

unified_df = engine.unify_datasets(datasets)

unified_df = engine.clear_na_rows(unified_df, engine.text_name)
unified_df = engine.clear_na_rows(unified_df, engine.label_name)
unified_df = engine.clear_lengthy_rows(unified_df, engine.text_name, 1024)
unified_df = engine.mask_money(unified_df, engine.text_name)
unified_df = engine.anonymize_data(unified_df, engine.text_name)
unified_df = engine.deduplicate_rows(unified_df, engine.text_name)

print(unified_df.head())

print(f"Unified dataset rows: {unified_df.shape[0]}")

unified_df.to_csv("data/silver/unified_dataset.csv", index=False)

os._exit(0) # Exit the script after loading the dataset and initializing the DataEngine