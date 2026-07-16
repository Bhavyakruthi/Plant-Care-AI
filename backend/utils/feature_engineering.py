"""
Standardized Feature Engineering Utility
========================================

Extracts structured features (numeric/binary) from the plant diagnostic
fields. This ensures consistency between training, CV, and inference.

Features:
- Severity (Ordinal: 0, 1, 2)
- Edge characteristics (Binary)
- Lesion characteristics (Binary)
- Distribution patterns (Binary)
- Text statistics (Length, Word Count)

Author: NLP Project Team
"""

import pandas as pd
import numpy as np

def extract_structured_features(df):
    """
    Input: DataFrame with RAW columns
    Output: NumPy array of structured features
    """
    # 1. Severity mapping
    severity_map = {'Low': 0, 'Medium': 1, 'High': 2}
    # Handle both original SEVERITY and potentially messy strings
    sev_encoded = df['SEVERITY'].fillna('Medium').map(lambda x: severity_map.get(str(x).capitalize(), 1)).values
    
    # 2. Binary keywords from morphology/mismatching cols
    # Note: We look for keywords in the raw columns
    def has_keyword(series, keyword):
        return series.astype(str).str.contains(keyword, case=False, na=False).astype(int)

    # Use MORPHOLOGY and other cols for edge/lesion info
    # (Replicating logic from 02_preprocessing.py but more robustly)
    all_text = df['MORPHOLOGY'].astype(str) + " " + df['LESIONS'].astype(str)
    
    feats = [
        sev_encoded,
        has_keyword(all_text, 'irregular'),
        has_keyword(all_text, 'necrotic'),
        has_keyword(all_text, 'curl'),
        has_keyword(all_text, 'brown'),
        has_keyword(all_text, 'yellow'),
        has_keyword(all_text, 'dark|black'),
        has_keyword(all_text, 'circular'),
        has_keyword(all_text, 'coalescing'),
        has_keyword(df['DISTRIBUTION'], 'marginal|margin'),
        has_keyword(df['DISTRIBUTION'], 'interveinal'),
        has_keyword(df['DISTRIBUTION'], 'widespread'),
        # Text stats from the raw columns (combined)
        df['MORPHOLOGY'].astype(str).str.len().values,
        df['MORPHOLOGY'].astype(str).str.split().str.len().values
    ]
    
    # Standardize result as float32 matrix
    return np.column_stack(feats).astype(np.float32)

def get_feature_names():
    return [
        'severity_encoded', 'has_irregular', 'has_necrotic', 
        'has_curled', 'has_brown', 'has_yellow',
        'has_dark', 'has_circular', 'has_coalescing', 
        'has_marginal', 'has_interveinal', 'has_widespread',
        'text_length', 'word_count'
    ]
