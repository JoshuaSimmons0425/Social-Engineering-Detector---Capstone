import pandas as pd
import json
from presidio_analyzer import AnalyzerEngine, BatchAnalyzerEngine
from presidio_anonymizer import BatchAnonymizerEngine

# Class to handle data cleaning, preprocessing, and anonymization

class DataEngine:

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.analyzer = AnalyzerEngine()
        self.batch_analyzer = BatchAnalyzerEngine()
        self.batch_anonymizer = BatchAnonymizerEngine()

    def clear_irrelevant_columns(self, columns_to_keep: list):
        """
        Remove irrelevant columns from the DataFrame.
        :param columns_to_keep: List of column names to keep in the DataFrame.
        """
        self.df = self.df[columns_to_keep]

        return self.df

    def rename_columns(self, columns_mapping: dict):
        """
        Rename columns in the DataFrame.
        :param columns_mapping: Dictionary mapping old column names to new column names.
        """
        self.df.rename(columns=columns_mapping, inplace=True)

        return self.df

    def anonymize_data(self, text_column: str):
        
        texts_dict = {text_column: self.df[text_column].fillna("").astype(str).tolist()}
        
        analyzer_results = self.batch_analyzer.analyze_dict(
            texts_dict,
            language="en",
            entities=["NAME", "EMAIL_ADDRESS", "PHONE_NUMBER", "ORGANIZATION", "LOCATION", "URL", "DATE_TIME", "CREDIT_CARD"],
        )
        
        anonymizer_results = self.batch_anonymizer.anonymize_dict(
            analyzer_results
        )
        
        masked_list = anonymizer_results.get(text_column, [])
        
        output_df = self.df.copy()
        output_df[text_column] = masked_list
        
        print(f"Successfully masked {text_column} column")
        return output_df