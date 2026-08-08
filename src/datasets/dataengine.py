import pandas as pd
import json
from presidio_analyzer import AnalyzerEngine, BatchAnalyzerEngine
from presidio_anonymizer import BatchAnonymizerEngine
import spacy

# Class to handle data cleaning, preprocessing, and anonymization

class DataEngine:

    def __init__(self):
        self.label_name = "Label"
        self.text_name = "Full_Text"
        self.nlp = spacy.load("en_core_web_sm")
        self.analyzer = AnalyzerEngine()
        self.batch_analyzer = BatchAnalyzerEngine()
        self.batch_anonymizer = BatchAnonymizerEngine()

    def clear_irrelevant_columns(self, df, columns_to_drop: list):
        """
        Remove irrelevant columns from the DataFrame.
        :param columns_to_keep: List of column names to keep in the DataFrame.
        """
        if df is None:
            raise ValueError("DataFrame is None")

        if columns_to_drop is None:
            raise ValueError("Columns to drop is None")

        df = df.drop(columns=columns_to_drop)
        return df
    
    def concat_subject2body(self, df, subject_column, body_column):

        if df is None:
            raise ValueError("DataFrame is None")

        if subject_column is None or body_column is None:
            raise ValueError("Subject or body column is None")

        df[body_column] = df[subject_column].astype(str) + df[body_column].astype(str)
        df = df.drop(columns=[subject_column])
        df = df.rename(columns={subject_column: body_column})
        return df

    def clear_na_rows(self, df, text_column):
        if df is None:
            raise ValueError("DataFrame is None")

        if text_column is None:
            raise ValueError("Text column is None")

        df = df.dropna(subset=[text_column])
        return df

    def deduplicate_rows(self, df, text_column):

        if df is None:
            raise ValueError("DataFrame is None")

        if text_column is None: 
            raise ValueError("Text column is None")     
        
        df = df.drop_duplicates(subset=[text_column])
        return df
    
    def clear_rows(self, df, column, condition, regex=False):

        if df is None:
            raise ValueError("DataFrame is None")

        if column is None:
            raise ValueError("Column is None")

        if condition is None:
            raise ValueError("Condition is required")

        df = df[~df[column].str.contains(condition, case=False, na=False, regex=regex)]
        return df

    def strip_with_regex(self, df, text_column, regex_pattern):

        if df is None:
            raise ValueError("DataFrame is None")

        if text_column is None:
            raise ValueError("Text column is None")

        if regex_pattern is None:
            raise ValueError("Pattern is required")

        df[text_column] = df[text_column].str.replace(regex_pattern, "", regex=True)
        return df

    def rename_columns(self, df, label_column, text_column):

        if df is None:
            raise ValueError("DataFrame is None")

        if label_column is None:
            raise ValueError("Label column is None")

        if text_column is None:
            raise ValueError("Text column is None")
        
        df = df.rename(columns={label_column: self.label_name, text_column: self.text_name})
        return df
    
    def rename_classes(self, df, label_column, label_mapping):

        if df is None:
            raise ValueError("DataFrame is None")

        if label_column is None:
            raise ValueError("Label column is None")

        if label_mapping is None:
            raise ValueError("Label mapping is None")

        df[label_column] = df[label_column].map(label_mapping)
        return df

    def unify_datasets(self, df_list: list):
        if df_list is None:
            raise ValueError("DataFrame list is None")
        
        unified_df = pd.concat(df_list, ignore_index=True)
        return unified_df

    def mask_money(self, df, text_column):
        if df is None:
            raise ValueError("DataFrame is None")

        if text_column is None:
            raise ValueError("Text column is None")

        PII_ENTITIES = {"MONEY"}

        def mask_text(text):
            if not isinstance(text, str):
                return text

            doc = self.nlp(text)
            text_list = list(text)

            ents = sorted(doc.ents, key=lambda e: e.start_char, reverse=True)

            for ent in sorted(doc.ents, key=lambda e: e.start_char, reverse=True):
                if ent.label_ in PII_ENTITIES:
                    text_list[ent.start_char:ent.end_char] = f"[{ent.label_}]"

            return "".join(text_list)

        df = df.copy()
        df[text_column] = df[text_column].apply(mask_text)
        return df

    def anonymize_data(self, df, text_column: str):

        if df is None:
            raise ValueError("DataFrame is None")

        if text_column is None:
            raise ValueError("Text column is None")
        
        texts_dict = {text_column: df[text_column].fillna("").astype(str).tolist()}
        
        analyzer_results = self.batch_analyzer.analyze_dict(
            texts_dict,
            language="en",
            entities=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "ORGANIZATION", "LOCATION", "URL", "DATE_TIME", "CREDIT_CARD"],
        )
        
        anonymizer_results = self.batch_anonymizer.anonymize_dict(
            analyzer_results
        )
        
        masked_list = anonymizer_results.get(text_column, [])
        
        output_df = df.copy()
        output_df[text_column] = masked_list
        
        print(f"Successfully masked {text_column} column")
        return output_df