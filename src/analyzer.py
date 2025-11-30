import pandas as pd
import os

class DataAnalyzer:
    def __init__(self, file_path, analysis_logs=False):
        self.file_path = file_path
        self.df = None
        self.analysis_logs = analysis_logs
    
    def log(self, msg):
        if self.analysis_logs:
            print(msg)
    
    def load_data(self):
        try:
            self.df = pd.read_csv(self.file_path)
            return True
        except Exception as e:
            self.log(f"Error loading file: {e}")
            return False

    def analyze_class_distribution(self, target_col):
        self.log("\n--- Columns ---")
        self.log(self.df.columns.tolist())

        if target_col in self.df.columns:
            self.log(f"\n--- Class Distribution for '{target_col}' ---")
            counts = self.df[target_col].value_counts()
            self.log(counts)
            
            self.log(f"\n--- Percentage Distribution ---")
            self.log(self.df[target_col].value_counts(normalize=True) * 100)
        else:
            self.log(f"\nColumn '{target_col}' not found. Available columns: {self.df.columns}")
