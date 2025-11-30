import pandas as pd
import numpy as np
import json
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight

class DataPreprocessor:
    def __init__(self, config):
        self.config = config
        self.output_path = config.get('output_path')
        self.weights_path = config.get('weights_path')
        self.method = config.get('method', 'oversample')

    def process(self, df, target_col):
        print(f"Starting preprocessing with method: {self.method}")
        
        if self.method == 'oversample':
            return self._oversample(df, target_col)
        elif self.method == 'undersample':
            return self._undersample(df, target_col)
        elif self.method == 'class_weights':
            return self._compute_weights(df, target_col)
        else:
            print(f"Unknown method: {self.method}")
            return df

    def _oversample(self, df, target_col):
        print("Performing Random Oversampling...")
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

    def _undersample(self, df, target_col):
        print("Performing Random Undersampling...")
        class_counts = df[target_col].value_counts()
        minority_class = class_counts.idxmin()
        minority_count = class_counts.min()
        
        print(f"Downsampling all classes to match minority count: {minority_count}")
        
        balanced_df = pd.DataFrame()
        
        for cls in class_counts.index:
            df_cls = df[df[target_col] == cls]
            df_cls_downsampled = resample(
                df_cls, 
                replace=False,
                n_samples=minority_count,
                random_state=42
            )
            balanced_df = pd.concat([balanced_df, df_cls_downsampled])
            
        return balanced_df

    def _compute_weights(self, df, target_col):
        print("Computing Class Weights...")
        classes = np.unique(df[target_col])
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=df[target_col])
        
        weight_dict = dict(zip(classes, weights))
        
        weight_dict = {k: float(v) for k, v in weight_dict.items()}
        
        print("Class Weights:", weight_dict)
        
        if self.weights_path:
            with open(self.weights_path, 'w') as f:
                json.dump(weight_dict, f, indent=4)
            print(f"Weights saved to {self.weights_path}")
            
        return df

    def save_data(self, df):
        if self.method == 'class_weights':
          pass
        
        if self.output_path and df is not None:
            print(f"Saving processed data to {self.output_path}...")
            df.to_csv(self.output_path, index=False)
            print("Save complete.")
