from datasets import load_dataset, Dataset
from huggingface_hub import HfApi
import pandas as pd
import os

# Load dataset from the initial Hugging Face upload
# Assuming the initial upload created a dataset named "sahithisaranya/tourism_dataset"
try:
    dataset = load_dataset("sahithisaranya/tourism_dataset")
    df = dataset["train"].to_pandas() # Assuming the dataset has a 'train' split
except Exception as e:
    print(f"Error loading dataset from Hugging Face: {e}")
    # Fallback or error handling if dataset cannot be loaded

# Convert to Hugging Face Dataset (if not already in that format or if modifications were made)
# If df is already a Dataset object, this step might be redundant
if not isinstance(df, Dataset):
    dataset_to_push = Dataset.from_pandas(df)
else:
    dataset_to_push = df


# Push to the processed dataset repository on Hugging Face Hub
# This assumes you want to register the *processed* dataset in this step
repo_id_processed = "sahithisaranya/tourism_dataset_processed"
dataset_to_push.push_to_hub(repo_id_processed)
print(f"✅ Processed Dataset uploaded to {repo_id_processed}")
