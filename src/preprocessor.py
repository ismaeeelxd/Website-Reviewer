import pandas as pd
from sklearn.utils import resample

class DataPreprocessor:
    def __init__(self, output_path):
        self.output_path = output_path

    def balance_classes(self, df, target_col): 
        class_counts = df[target_col].value_counts()
        majority_class = class_counts.idxmax()
        majority_count = class_counts.max()
        
        df_majority = df[df[target_col] == majority_class]
        balanced_df = df_majority.copy()
        
        minority_classes = class_counts[class_counts.index != majority_class].index
        
        for cls in minority_classes:
            df_minority = df[df[target_col] == cls]
            
            df_minority_upsampled = resample(
                df_minority, 
                replace=True,     
                n_samples=majority_count,    
                random_state=42   
            )
            
            balanced_df = pd.concat([balanced_df, df_minority_upsampled])
            
        return balanced_df

    def save_data(self, df):
        df.to_csv(self.output_path, index=False)
