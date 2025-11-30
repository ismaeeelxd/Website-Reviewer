from src.config_loader import load_config
from src.downloader import download_from_drive
from src.analyzer import DataAnalyzer
import os

def main():
    config = load_config()

    data_config = config['data']
    analysis_config = config['analysis']

    file_id = data_config['file_id']
    raw_path = data_config['raw_path']
    analysis_logs = analysis_config['analysis_logs']
    target_col = analysis_config['target_col']
    
    if not os.path.exists(raw_path):
        download_from_drive(file_id, raw_path)

    analyzer = DataAnalyzer(raw_path, analysis_logs)
    if not analyzer.load_data():
        return
    
    analyzer.analyze_class_distribution(target_col)

    preprocessing_config = config['preprocessing']
    if preprocessing_config['enable']:
        from src.preprocessor import DataPreprocessor
        
        preprocessor = DataPreprocessor(preprocessing_config)
        
        processed_df = preprocessor.process(analyzer.df, target_col)
        preprocessor.save_data(processed_df)
        
if __name__ == "__main__":
    main()
