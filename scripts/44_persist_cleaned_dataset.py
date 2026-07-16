"""
Cleaned Dataset Persistence Script
==================================

Applies the full NLP preprocessing pipeline (Lemmatization, Stop Words, 
JSON stripping) to the entire dataset and saves it to a new CSV.

Purpose: 
- Ensures consistency across all training and evaluation scripts.
- Speeds up re-training by avoiding redundant preprocessing.
- Provides a transparent view of the "Cleaned" data.

Author: NLP Project Team
"""

import os
import sys
import pandas as pd
from tqdm import tqdm

# Add project root to path
sys.path.append(os.getcwd())

from backend.utils.nlp_pipeline import get_cleaned_diagnostic_string

def main():
    RAW_DATA_PATH = 'data/multimodal_plant_data.csv'
    CLEANED_DATA_PATH = 'data/cleaned_multimodal_plant_data.csv'
    
    if not os.path.exists(RAW_DATA_PATH):
        print(f"❌ Raw dataset not found at {RAW_DATA_PATH}")
        return

    print(f"Loading raw dataset from {RAW_DATA_PATH}...")
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"✅ Loaded {len(df)} samples")
    
    print("\n" + "="*50)
    print("STARTING FULL NLP PREPROCESSING")
    print(" (JSON stripping, Stop Words, Lemmatization)")
    print("="*50)
    
    # Process with tqdm loop for better compatibility
    cleaned_texts = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Cleaning text"):
        cleaned_texts.append(get_cleaned_diagnostic_string(row))
    
    df['cleaned_text'] = cleaned_texts
    
    # Keep only important columns for the new CSV to keep it lean
    # but keep all metadata just in case.
    print(f"\nSaving cleaned dataset to {CLEANED_DATA_PATH}...")
    df.to_csv(CLEANED_DATA_PATH, index=False)
    
    print("\n" + "="*50)
    print("✅ DATA CLEANING COMPLETE!")
    print(f"New file available at: {CLEANED_DATA_PATH}")
    print("="*50)
    
    # Show a few examples
    print("\nSample of Cleaned Text:")
    for i in range(3):
        print(f"\n[{i+1}] LABEL: {df['LABEL'].iloc[i]}")
        print(f"    CLEANED: {df['cleaned_text'].iloc[i][:100]}...")

if __name__ == "__main__":
    main()
