import pandas as pd
import json
from presidio_analyzer import AnalyzerEngine, BatchAnalyzerEngine
from presidio_anonymizer import BatchAnonymizerEngine

# Class to handle data cleaning, preprocessing, and anonymization

class DataEngine:

    def __init__(self):
        self.label_name = "Label"
        self.text_name = "Full_Text"
        self.analyzer = AnalyzerEngine()
        self.batch_analyzer = BatchAnalyzerEngine()
        self.batch_anonymizer = BatchAnonymizerEngine()

    def clear_irrelevant_columns(self, df, columns_to_drop: list):
        """
        Remove irrelevant columns from the DataFrame.
        :param columns_to_keep: List of column names to keep in the DataFrame.
        """
        df = df.drop(columns=columns_to_drop)
        return df
    
    def concat_subject2body(self, df, subject_column, body_column):
        df[body_column] = df[subject_column].astype(str) + df[body_column].astype(str)
        df = df.drop(columns=[subject_column])
        df = df.rename(columns={subject_column: body_column})
        return df

    def rename_columns(self, df, label_column, text_column):
        
        df = df.rename(columns={label_column: self.label_name, text_column: self.text_name})

        return df

    def anonymize_data(self, df, text_column: str):
        
        texts_dict = {text_column: df[text_column].fillna("").astype(str).tolist()}
        
        analyzer_results = self.batch_analyzer.analyze_dict(
            texts_dict,
            language="en",
            entities=["NAME", "EMAIL_ADDRESS", "PHONE_NUMBER", "ORGANIZATION", "LOCATION", "URL", "DATE_TIME", "CREDIT_CARD"],
        )
        
        anonymizer_results = self.batch_anonymizer.anonymize_dict(
            analyzer_results
        )
        
        masked_list = anonymizer_results.get(text_column, [])
        
        output_df = df.copy()
        output_df[text_column] = masked_list
        
        print(f"Successfully masked {text_column} column")
        return output_df