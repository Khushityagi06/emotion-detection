import kaggle
import os
import shutil

# Create data directory if it doesn't exist
data_path = os.path.join(os.path.dirname(__file__), 'data')
if not os.path.exists(data_path):
    os.makedirs(data_path)

print("Starting download...")
try:
    # Download the dataset
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files('msambare/fer2013', path=data_path, unzip=True)
    print("Download complete!")
except Exception as e:
    print(f"An error occurred: {e}")
    print("Please check if:")
    print("1. Your internet connection is stable")
    print("2. The kaggle.json file is correctly placed")
    print("3. You have accepted the competition rules on Kaggle website")