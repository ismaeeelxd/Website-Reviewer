import gdown
import os

def download_from_drive(file_id: str, output_path: str):
    url = f'https://drive.google.com/uc?id={file_id}'
    directory = os.path.dirname(output_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    gdown.download(url, output_path, quiet=False)
    print("Download complete.")
