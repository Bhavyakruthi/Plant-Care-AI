"""
Plant Disease Prediction - Preprocessing & Feature Engineering
================================================================

This script performs:
1. Text cleaning and normalization
2. Feature extraction from structured fields
3. TF-IDF vectorization (for baseline models)
4. BERT tokenization (for transformer models)
5. Train/validation/test split (stratified)
6. Data preparation for modeling

Author: NLP Project Team
"""

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer
import re
import pickle
import os
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. CHECK CUDA AVAILABILITY
# ==============================================================================

def check_cuda():
    """Check if CUDA is available for GPU acceleration"""
    print("=" * 80)
    print("CUDA AVAILABILITY CHECK")
    print("=" * 80)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"\n✅ CUDA is available!")
        print(f"   • GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"   • CUDA Version: {torch.version.cuda}")
        print(f"   • Number of GPUs: {torch.cuda.device_count()}")
        print(f"   • Current Device: {torch.cuda.current_device()}")
        
        # Memory info
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   • Total GPU Memory: {total_memory:.2f} GB")
        
        print(f"\n🚀 Models will be trained on GPU (much faster!)")
    else:
        device = torch.device("cpu")
        print(f"\n⚠️  CUDA is NOT available. Using CPU.")
        print(f"   💡 Training will be slower on CPU.")
        print(f"   💡 To use GPU: Install CUDA-enabled PyTorch")
    
    return device


# ==============================================================================
# 2. TEXT CLEANING & NORMALIZATION
# ==============================================================================

def clean_text(text):
    """
    Refined clean and normalize text
    
    Why refined?
    - Preserves scientific terms and botanical descriptions
    - Removes punctuation that doesn't add value
    - Normalizes common plant-related terminology
    """
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Expand common contractions
    text = text.replace("can't", "cannot").replace("won't", "will not")
    
    # Remove special characters but keep hyphens/apostrophes/dots (dots for abbreviations)
    text = re.sub(r'[^a-z0-9\s.\-\']', ' ', text)
    
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

# ==============================================================================
# 2.1 DATA AUGMENTATION (NEW)
# ==============================================================================

def augment_minority_classes(df, text_col, label_col, min_samples=200):
    """
    Simple synonym-based augmentation for minority classes
    
    Args:
        df: Input DataFrame
        text_col: Column with text
        label_col: Column with labels
        min_samples: Minimum samples desired per class
    """
    print(f"\n🚀 Augmenting minority classes (target: {min_samples} samples per class)...")
    
    # Identify minority classes
    counts = df[label_col].value_counts()
    minority_classes = counts[counts < min_samples].index
    
    new_rows = []
    
    # Simple synonym dictionary for botanical terms (can be expanded)
    synonyms = {
        'yellow': ['chlorotic', 'pale', 'yellowish'],
        'spots': ['lesions', 'dots', 'patches'],
        'brown': ['necrotic', 'dark', 'bronze'],
        'leaves': ['foliage', 'leafage'],
        'holes': ['perforations', 'punctures'],
        'irregular': ['asymmetrical', 'uneven'],
        'circular': ['round', 'spherical']
    }
    
    for cls in minority_classes:
        cls_df = df[df[label_col] == cls]
        num_to_add = min_samples - len(cls_df)
        print(f"   • {cls}: {len(cls_df)} -> {min_samples} (adding {num_to_add})")
        
        # Keep track of added to avoid infinite loop
        added = 0
        while added < num_to_add:
            for _, row in cls_df.iterrows():
                if added >= num_to_add:
                    break
                
                new_text = row[text_col]
                # Simple augmentation: replace random synonyms
                for word, syn_list in synonyms.items():
                    if word in new_text and np.random.random() > 0.5:
                        syn = np.random.choice(syn_list)
                        new_text = new_text.replace(word, syn)
                
                new_row = row.copy()
                new_row[text_col] = new_text
                new_rows.append(new_row)
                added += 1
                
    if new_rows:
        df_augmented = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        print(f"✅ Augmentation complete! Added {len(new_rows)} samples.")
        print(f"   Total samples: {len(df_augmented):,}")
        return df_augmented
    
    return df


def preprocess_text_columns(df):
    """
    Preprocess and combine text columns
    
    Args:
        df (pd.DataFrame): Dataset
        
    Returns:
        pd.DataFrame: Dataset with cleaned text
    """
    print("\n" + "=" * 80)
    print("STEP 1: TEXT CLEANING & NORMALIZATION")
    print("=" * 80)
    
    # Create combined text from all relevant fields
    print("\n🧹 Cleaning text fields...")
    
    df['morphology_clean'] = df['MORPHOLOGY'].apply(clean_text)
    df['lesions_clean'] = df['LESIONS'].apply(clean_text)
    df['distribution_clean'] = df['DISTRIBUTION'].apply(clean_text)
    
    # Combine all text features
    df['combined_text_clean'] = (
        df['morphology_clean'] + ' ' +
        df['lesions_clean'] + ' ' +
        df['distribution_clean']
    )
    
    print("✅ Text cleaning completed")
    
    print("\n👀 Sample cleaned text:")
    for i in range(2):
        print(f"\n   Sample {i+1}:")
        print(f"   Original: {df['DISTRIBUTION'].iloc[i][:100]}...")
        print(f"   Cleaned: {df['distribution_clean'].iloc[i][:100]}...")
    
    return df


# ==============================================================================
# 3. FEATURE EXTRACTION FROM STRUCTURED FIELDS
# ==============================================================================

def extract_structured_features(df):
    """
    Extract additional features from parsed dictionary fields
    
    Args:
        df (pd.DataFrame): Dataset
        
    Returns:
        pd.DataFrame: Dataset with additional features
    """
    print("\n" + "=" * 80)
    print("STEP 2: STRUCTURED FEATURE EXTRACTION")
    print("=" * 80)
    
    print("\n🔧 Extracting binary and categorical features...")
    
    # Severity encoding
    severity_map = {'Low': 0, 'Medium': 1, 'High': 2}
    df['severity_encoded'] = df['SEVERITY'].map(severity_map)
    
    # Extract keywords from edge info
    df['has_irregular_edges'] = df['edge_info'].str.contains('irregular', case=False, na=False).astype(int)
    df['has_necrotic_edges'] = df['edge_info'].str.contains('necrotic', case=False, na=False).astype(int)
    df['has_curled_edges'] = df['edge_info'].str.contains('curl', case=False, na=False).astype(int)
    
    # Extract keywords from lesion color
    df['has_brown_lesions'] = df['lesion_color'].str.contains('brown', case=False, na=False).astype(int)
    df['has_yellow_lesions'] = df['lesion_color'].str.contains('yellow', case=False, na=False).astype(int)
    df['has_dark_lesions'] = df['lesion_color'].str.contains('dark|black', case=False, na=False).astype(int)
    
    # Extract keywords from lesion shape
    df['has_irregular_lesions'] = df['lesion_shape'].str.contains('irregular', case=False, na=False).astype(int)
    df['has_circular_lesions'] = df['lesion_shape'].str.contains('circular', case=False, na=False).astype(int)
    df['has_coalescing_lesions'] = df['lesion_shape'].str.contains('coalescing', case=False, na=False).astype(int)
    
    # Extract keywords from distribution
    df['has_marginal_dist'] = df['DISTRIBUTION'].str.contains('marginal|margin', case=False, na=False).astype(int)
    df['has_interveinal_dist'] = df['DISTRIBUTION'].str.contains('interveinal', case=False, na=False).astype(int)
    df['has_widespread_dist'] = df['DISTRIBUTION'].str.contains('widespread', case=False, na=False).astype(int)
    
    # Text statistics
    df['text_length'] = df['combined_text_clean'].str.len()
    df['word_count'] = df['combined_text_clean'].str.split().str.len()
    
    structured_features = [
        'severity_encoded', 'has_irregular_edges', 'has_necrotic_edges', 
        'has_curled_edges', 'has_brown_lesions', 'has_yellow_lesions',
        'has_dark_lesions', 'has_irregular_lesions', 'has_circular_lesions',
        'has_coalescing_lesions', 'has_marginal_dist', 'has_interveinal_dist',
        'has_widespread_dist', 'text_length', 'word_count'
    ]
    
    print(f"✅ Extracted {len(structured_features)} structured features")
    print(f"\n📋 Feature list:")
    for i, feat in enumerate(structured_features, 1):
        print(f"   {i:2d}. {feat}")
    
    return df, structured_features


# ==============================================================================
# 4. TF-IDF VECTORIZATION (FOR BASELINE MODELS)
# ==============================================================================

def create_tfidf_features(X_train_text, X_val_text, X_test_text, max_features=5000):
    """
    Create TF-IDF features for baseline models
    
    Args:
        X_train_text (pd.Series): Training text
        X_val_text (pd.Series): Validation text
        X_test_text (pd.Series): Test text
        max_features (int): Maximum number of features
        
    Returns:
        Tuple of TF-IDF matrices and vectorizer
    """
    print("\n" + "=" * 80)
    print("STEP 3: TF-IDF VECTORIZATION (Baseline Models)")
    print("=" * 80)
    
    print(f"\n🔢 Creating TF-IDF features (max_features={max_features})...")
    
    # Initialize TF-IDF vectorizer
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),  # Unigrams and bigrams
        min_df=2,  # Ignore terms that appear in less than 2 documents
        max_df=0.95,  # Ignore terms that appear in more than 95% of documents
        sublinear_tf=True  # Apply sublinear tf scaling
    )
    
    # Fit on training data and transform all sets
    X_train_tfidf = tfidf.fit_transform(X_train_text)
    X_val_tfidf = tfidf.transform(X_val_text)
    X_test_tfidf = tfidf.transform(X_test_text)
    
    print(f"✅ TF-IDF vectorization completed")
    print(f"   • Vocabulary size: {len(tfidf.vocabulary_):,}")
    print(f"   • Train shape: {X_train_tfidf.shape}")
    print(f"   • Validation shape: {X_val_tfidf.shape}")
    print(f"   • Test shape: {X_test_tfidf.shape}")
    
    # Show top features
    feature_names = tfidf.get_feature_names_out()
    print(f"\n🔝 Sample TF-IDF features:")
    for i in range(min(10, len(feature_names))):
        print(f"   {i+1:2d}. {feature_names[i]}")
    
    return X_train_tfidf, X_val_tfidf, X_test_tfidf, tfidf


# ==============================================================================
# 5. BERT TOKENIZATION (FOR TRANSFORMER MODELS)
# ==============================================================================

def create_bert_encodings(X_train_text, X_val_text, X_test_text, max_length=256):
    """
    Create BERT tokenized encodings for transformer models
    
    Args:
        X_train_text (pd.Series): Training text
        X_val_text (pd.Series): Validation text
        X_test_text (pd.Series): Test text
        max_length (int): Maximum sequence length
        
    Returns:
        Tuple of BERT encodings and tokenizer
    """
    print("\n" + "=" * 80)
    print("STEP 4: BERT TOKENIZATION (Transformer Models)")
    print("=" * 80)
    
    print(f"\n🤖 Loading BioBERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract')
    
    print(f"\n🔤 Tokenizing text (max_length={max_length})...")
    
    # Tokenize training data
    train_encodings = tokenizer(
        X_train_text.tolist(),
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    # Tokenize validation data
    val_encodings = tokenizer(
        X_val_text.tolist(),
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    # Tokenize test data
    test_encodings = tokenizer(
        X_test_text.tolist(),
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    print(f"✅ BERT tokenization completed")
    print(f"   • Train shape: {train_encodings['input_ids'].shape}")
    print(f"   • Validation shape: {val_encodings['input_ids'].shape}")
    print(f"   • Test shape: {test_encodings['input_ids'].shape}")
    
    # Show sample tokenization
    print(f"\n👀 Sample tokenization:")
    sample_text = X_train_text.iloc[0][:100]
    sample_tokens = tokenizer.tokenize(sample_text)
    print(f"   Original: {sample_text}...")
    print(f"   Tokens: {' | '.join(sample_tokens[:15])}...")
    
    return train_encodings, val_encodings, test_encodings, tokenizer


# ==============================================================================
# 6. TRAIN-VALIDATION-TEST SPLIT
# ==============================================================================

def split_data(df, test_size=0.15, val_size=0.15, random_state=42, augment=True):
    """
    Split data into train, validation, and test sets (stratified)
    with optional minority class augmentation for training set.
    """
    print("\n" + "=" * 80)
    print("STEP 5: TRAIN-VALIDATION-TEST SPLIT (Stratified)")
    print("=" * 80)
    
    # Prepare features and labels
    # We split the full dataframe first to keep all columns for augmentation
    print(f"\n📊 Split configuration:")
    print(f"   • Test size: {test_size*100:.0f}%")
    print(f"   • Validation size: {val_size*100:.0f}%")
    print(f"   • Training size: {(1-test_size-val_size)*100:.0f}%")
    
    # First split: train+val vs test
    train_val_df, test_df = train_test_split(
        df, 
        test_size=test_size,
        stratify=df['LABEL'],
        random_state=random_state
    )
    
    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size_adjusted,
        stratify=train_val_df['LABEL'],
        random_state=random_state
    )
    
    # --- AUGMENTATION (Training set only) ---
    if augment:
        train_df = augment_minority_classes(train_df, 'combined_text_clean', 'LABEL', min_samples=300)
    
    X_train = train_df['combined_text_clean']
    y_train = train_df['LABEL']
    X_val = val_df['combined_text_clean']
    y_val = val_df['LABEL']
    X_test = test_df['combined_text_clean']
    y_test = test_df['LABEL']
    
    print(f"\n✅ Data split completed:")
    print(f"   • Training samples: {len(X_train):,} (Original + Augmented)")
    print(f"   • Validation samples: {len(X_val):,}")
    print(f"   • Test samples: {len(X_test):,}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


# ==============================================================================
# 7. SAVE PREPROCESSED DATA
# ==============================================================================

def save_preprocessed_data(data_dict, filename='preprocessed_data.pkl'):
    """
    Save preprocessed data to pickle file
    
    Args:
        data_dict (dict): Dictionary containing all preprocessed data
        filename (str): Output filename
    """
    print("\n" + "=" * 80)
    print("SAVING PREPROCESSED DATA")
    print("=" * 80)
    
    with open(filename, 'wb') as f:
        pickle.dump(data_dict, f)
    
    print(f"\n💾 Preprocessed data saved: {filename}")
    print(f"   • File size: {os.path.getsize(filename) / 1024**2:.2f} MB")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """Main execution function"""
    
    import os
    
    print("\n" + "=" * 80)
    print(" " * 20 + "PLANT DISEASE PREDICTION")
    print(" " * 18 + "Preprocessing & Feature Engineering")
    print("=" * 80)
    
    # Check CUDA
    device = check_cuda()
    
    # Load processed data from EDA step
    input_file = 'processed_data_with_features.csv'
    print(f"\n📂 Loading data from: {input_file}")
    df = pd.read_csv(input_file)
    print(f"✅ Loaded {len(df):,} samples")
    
    # Preprocessing steps
    df = preprocess_text_columns(df)
    df, structured_features = extract_structured_features(df)
    
    # Split data (with augmentation)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, augment=True)
    
    # Create TF-IDF features
    X_train_tfidf, X_val_tfidf, X_test_tfidf, tfidf = create_tfidf_features(
        X_train, X_val, X_test, max_features=5000
    )
    
    # Create BERT encodings
    train_encodings, val_encodings, test_encodings, tokenizer = create_bert_encodings(
        X_train, X_val, X_test, max_length=256
    )
    
    # Prepare structured features
    # Note: For simplicity in the ensemble, we'll keep structured features flat or re-derive
    # for augmented samples if necessary. For now, we'll use empty fillers for augmented samples
    # to maintain shape, as text is the primary focus.
    
    def get_structured_matrix(texts, original_df, features):
        indices = texts.index
        # For augmented samples (index not in original_df), use zero vector
        matrix = []
        for idx in indices:
            if idx in original_df.index:
                matrix.append(original_df.loc[idx, features].values)
            else:
                matrix.append(np.zeros(len(features)))
        return np.array(matrix)

    X_train_structured = get_structured_matrix(X_train, df, structured_features)
    X_val_structured = get_structured_matrix(X_val, df, structured_features)
    X_test_structured = get_structured_matrix(X_test, df, structured_features)
    
    # Encode labels
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)
    y_test_encoded = label_encoder.transform(y_test)
    
    print(f"\n🏷️  Label encoding:")
    print(f"   • Number of classes: {len(label_encoder.classes_)}")
    print(f"   • Classes: {label_encoder.classes_[:5]}... (showing first 5)")
    
    # Package all data
    preprocessed_data = {
        # Text data
        'X_train_text': X_train,
        'X_val_text': X_val,
        'X_test_text': X_test,
        
        # TF-IDF features
        'X_train_tfidf': X_train_tfidf,
        'X_val_tfidf': X_val_tfidf,
        'X_test_tfidf': X_test_tfidf,
        'tfidf_vectorizer': tfidf,
        
        # BERT encodings
        'train_encodings': train_encodings,
        'val_encodings': val_encodings,
        'test_encodings': test_encodings,
        'bert_tokenizer': tokenizer,
        
        # Structured features
        'X_train_structured': X_train_structured,
        'X_val_structured': X_val_structured,
        'X_test_structured': X_test_structured,
        'structured_feature_names': structured_features,
        
        # Labels
        'y_train': y_train_encoded,
        'y_val': y_val_encoded,
        'y_test': y_test_encoded,
        'y_train_original': y_train,
        'y_val_original': y_val,
        'y_test_original': y_test,
        'label_encoder': label_encoder,
        
        # Device info
        'device': device
    }
    
    # Save
    save_preprocessed_data(preprocessed_data, 'preprocessed_data.pkl')
    
    print("\n" + "=" * 80)
    print("✅ Preprocessing completed successfully!")
    print("=" * 80)
    print("\n🎯 Next step: Run model training script")
    print("   • Baseline models: 02_train_baseline_models.py")
    print("   • Transformer model: 03_train_bert_model.py")


if __name__ == "__main__":
    main()
