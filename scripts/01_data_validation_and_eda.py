"""
Plant Disease Prediction - Data Validation & Exploratory Data Analysis
========================================================================

This script performs:
1. Data loading and validation
2. Missing value analysis
3. Class distribution analysis
4. Dictionary field parsing
5. Comprehensive visualizations (9 types)
6. Text statistics

Author: NLP Project Team
Dataset: PlantVillage Multimodal Plant Data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import json
import ast
import warnings
from collections import Counter
import re

warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ==============================================================================
# 1. DATA LOADING & INITIAL VALIDATION
# ==============================================================================

def load_and_validate_data(file_path):
    """
    Load dataset and perform initial validation
    
    Args:
        file_path (str): Path to CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    print("=" * 80)
    print("STEP 1: DATA LOADING & VALIDATION")
    print("=" * 80)
    
    # Load data
    print(f"\n📂 Loading data from: {file_path}")
    df = pd.read_csv(file_path)
    
    # Basic info
    print(f"\n✅ Data loaded successfully!")
    print(f"   • Total samples: {len(df):,}")
    print(f"   • Total features: {len(df.columns)}")
    print(f"   • Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Display columns
    print(f"\n📋 Columns:")
    for i, col in enumerate(df.columns, 1):
        print(f"   {i}. {col} ({df[col].dtype})")
    
    # First few rows
    print(f"\n👀 First 3 samples:")
    print(df.head(3).to_string())
    
    return df


def check_missing_values(df):
    """
    Comprehensive missing value analysis
    
    Args:
        df (pd.DataFrame): Dataset
    """
    print("\n" + "=" * 80)
    print("STEP 2: MISSING VALUE ANALYSIS")
    print("=" * 80)
    
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': missing.values,
        'Missing_Percentage': missing_pct.values
    })
    
    missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values(
        'Missing_Count', ascending=False
    )
    
    if len(missing_df) == 0:
        print("\n✅ No missing values found! Dataset is complete.")
    else:
        print(f"\n⚠️  Found missing values in {len(missing_df)} columns:")
        print(missing_df.to_string(index=False))
        
        # Visualize
        if len(missing_df) > 0:
            plt.figure(figsize=(10, 6))
            plt.barh(missing_df['Column'], missing_df['Missing_Percentage'])
            plt.xlabel('Missing Percentage (%)')
            plt.title('Missing Values Analysis')
            plt.tight_layout()
            plt.savefig('outputs/01_missing_values.png', dpi=300, bbox_inches='tight')
            print("\n📊 Visualization saved: outputs/01_missing_values.png")
            plt.close()


# ==============================================================================
# 2. CLASS DISTRIBUTION ANALYSIS
# ==============================================================================

def analyze_class_distribution(df):
    """
    Analyze and visualize class distribution
    
    Args:
        df (pd.DataFrame): Dataset
    """
    print("\n" + "=" * 80)
    print("STEP 3: CLASS DISTRIBUTION ANALYSIS")
    print("=" * 80)
    
    # Count classes
    class_counts = df['LABEL'].value_counts()
    
    print(f"\n📊 Total unique classes: {len(class_counts)}")
    print(f"\n🔢 Class distribution:")
    for i, (label, count) in enumerate(class_counts.items(), 1):
        pct = (count / len(df)) * 100
        print(f"   {i:2d}. {label:50s}: {count:5d} ({pct:5.2f}%)")
    
    # Check for imbalance
    max_count = class_counts.max()
    min_count = class_counts.min()
    imbalance_ratio = max_count / min_count
    
    print(f"\n⚖️  Class Balance Analysis:")
    print(f"   • Most common class: {class_counts.idxmax()} ({max_count} samples)")
    print(f"   • Least common class: {class_counts.idxmin()} ({min_count} samples)")
    print(f"   • Imbalance ratio: {imbalance_ratio:.2f}x")
    
    if imbalance_ratio > 3:
        print("   ⚠️  WARNING: High class imbalance detected!")
        print("   💡 Recommendation: Use stratified sampling and weighted metrics")
    else:
        print("   ✅ Classes are relatively balanced")
    
    # Visualizations
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bar chart
    class_counts.plot(kind='barh', ax=axes[0], color='steelblue')
    axes[0].set_xlabel('Number of Samples')
    axes[0].set_ylabel('Disease Class')
    axes[0].set_title('Class Distribution - Count')
    axes[0].grid(axis='x', alpha=0.3)
    
    # Pie chart (top 10 for readability)
    top_10 = class_counts.head(10)
    colors = plt.cm.Set3(range(len(top_10)))
    axes[1].pie(top_10, labels=top_10.index, autopct='%1.1f%%', 
                startangle=90, colors=colors)
    axes[1].set_title('Top 10 Classes - Distribution')
    
    plt.tight_layout()
    plt.savefig('outputs/02_class_distribution.png', dpi=300, bbox_inches='tight')
    print("\n📊 Visualization saved: outputs/02_class_distribution.png")
    plt.close()


def analyze_severity_distribution(df):
    """
    Analyze severity level distribution
    
    Args:
        df (pd.DataFrame): Dataset
    """
    print("\n" + "=" * 80)
    print("STEP 4: SEVERITY DISTRIBUTION ANALYSIS")
    print("=" * 80)
    
    severity_counts = df['SEVERITY'].value_counts()
    
    print(f"\n📊 Severity levels:")
    for severity, count in severity_counts.items():
        pct = (count / len(df)) * 100
        print(f"   • {severity:10s}: {count:5d} ({pct:5.2f}%)")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart
    severity_counts.plot(kind='bar', ax=axes[0], color=['#2ecc71', '#f39c12', '#e74c3c'])
    axes[0].set_xlabel('Severity Level')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Severity Distribution - Bar Chart')
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)
    
    # Pie chart
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    axes[1].pie(severity_counts, labels=severity_counts.index, autopct='%1.1f%%',
                startangle=90, colors=colors[:len(severity_counts)])
    axes[1].set_title('Severity Distribution - Pie Chart')
    
    plt.tight_layout()
    plt.savefig('outputs/03_severity_distribution.png', dpi=300, bbox_inches='tight')
    print("\n📊 Visualization saved: outputs/03_severity_distribution.png")
    plt.close()


# ==============================================================================
# 3. DICTIONARY FIELD PARSING
# ==============================================================================

def parse_dict_field(text):
    """
    Parse dictionary-like string field OR handle plain text
    
    Args:
        text (str): Dictionary string or plain text
        
    Returns:
        dict: Parsed dictionary, {'description': text}, or empty dict
    """
    if pd.isna(text):
        return {}
    
    text_str = str(text).strip()
    
    # Check if it looks like a dictionary
    if text_str.startswith('{') and ':' in text_str:
        try:
            # Try direct JSON parsing
            return json.loads(text_str.replace("'", '"'))
        except:
            try:
                # Try ast.literal_eval
                return ast.literal_eval(text_str)
            except:
                pass
    
    # If not a dict or parsing failed, treat as plain text
    return {'description': text_str}


def extract_features_from_dicts(df):
    """
    Extract features from MORPHOLOGY and LESIONS fields
    Handles both dictionary format and plain text
    
    Args:
        df (pd.DataFrame): Dataset
        
    Returns:
        pd.DataFrame: Dataset with extracted features
    """
    print("\n" + "=" * 80)
    print("STEP 5: PARSING DICTIONARY FIELDS")
    print("=" * 80)
    
    print("\n🔧 Parsing MORPHOLOGY field...")
    df['morphology_dict'] = df['MORPHOLOGY'].apply(parse_dict_field)
    df['edge_info'] = df['morphology_dict'].apply(
        lambda x: x.get('edges', '') if isinstance(x, dict) else ''
    )
    df['vein_info'] = df['morphology_dict'].apply(
        lambda x: x.get('veins', '') if isinstance(x, dict) else ''
    )
    
    print("✅ Extracted: edge_info, vein_info")
    
    print("\n🔧 Parsing LESIONS field...")
    df['lesions_dict'] = df['LESIONS'].apply(parse_dict_field)
    df['lesion_color'] = df['lesions_dict'].apply(
        lambda x: x.get('color', '') if isinstance(x, dict) else ''
    )
    df['lesion_shape'] = df['lesions_dict'].apply(
        lambda x: x.get('shape', '') if isinstance(x, dict) else ''
    )
    
    print("✅ Extracted: lesion_color, lesion_shape")
    
    # Sample output
    print("\n👀 Sample extracted features:")
    sample = df[['edge_info', 'vein_info', 'lesion_color', 'lesion_shape']].head(3)
    print(sample.to_string())
    
    return df


# ==============================================================================
# 4. TEXT STATISTICS
# ==============================================================================

def analyze_text_statistics(df):
    """
    Analyze text length and statistics
    
    Args:
        df (pd.DataFrame): Dataset
    """
    print("\n" + "=" * 80)
    print("STEP 6: TEXT STATISTICS ANALYSIS")
    print("=" * 80)
    
    # Combine all text fields
    df['combined_text'] = (
        df['MORPHOLOGY'].fillna('') + ' ' +
        df['LESIONS'].fillna('') + ' ' +
        df['DISTRIBUTION'].fillna('')
    )
    
    # Calculate statistics
    df['text_length'] = df['combined_text'].str.len()
    df['word_count'] = df['combined_text'].str.split().str.len()
    
    print("\n📏 Text Length Statistics:")
    print(f"   • Mean length: {df['text_length'].mean():.1f} characters")
    print(f"   • Median length: {df['text_length'].median():.1f} characters")
    print(f"   • Min length: {df['text_length'].min():.0f} characters")
    print(f"   • Max length: {df['text_length'].max():.0f} characters")
    print(f"   • Std deviation: {df['text_length'].std():.1f} characters")
    
    print("\n📝 Word Count Statistics:")
    print(f"   • Mean words: {df['word_count'].mean():.1f}")
    print(f"   • Median words: {df['word_count'].median():.1f}")
    print(f"   • Min words: {df['word_count'].min():.0f}")
    print(f"   • Max words: {df['word_count'].max():.0f}")
    
    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Text length distribution
    axes[0, 0].hist(df['text_length'], bins=50, color='skyblue', edgecolor='black')
    axes[0, 0].set_xlabel('Text Length (characters)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Text Length Distribution')
    axes[0, 0].axvline(df['text_length'].mean(), color='red', 
                       linestyle='--', label=f'Mean: {df["text_length"].mean():.0f}')
    axes[0, 0].legend()
    
    # Word count distribution
    axes[0, 1].hist(df['word_count'], bins=50, color='lightcoral', edgecolor='black')
    axes[0, 1].set_xlabel('Word Count')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Word Count Distribution')
    axes[0, 1].axvline(df['word_count'].mean(), color='red',
                       linestyle='--', label=f'Mean: {df["word_count"].mean():.0f}')
    axes[0, 1].legend()
    
    # Box plot - text length by severity
    severity_order = ['Low', 'Medium', 'High']
    df_sorted = df.copy()
    df_sorted['SEVERITY'] = pd.Categorical(df_sorted['SEVERITY'], 
                                           categories=severity_order, ordered=True)
    df_sorted = df_sorted.sort_values('SEVERITY')
    
    sns.boxplot(data=df_sorted, x='SEVERITY', y='text_length', ax=axes[1, 0])
    axes[1, 0].set_title('Text Length vs Severity Level')
    axes[1, 0].set_ylabel('Text Length (characters)')
    
    # Violin plot
    sns.violinplot(data=df_sorted, x='SEVERITY', y='word_count', ax=axes[1, 1])
    axes[1, 1].set_title('Word Count vs Severity Level')
    axes[1, 1].set_ylabel('Word Count')
    
    plt.tight_layout()
    plt.savefig('outputs/04_text_statistics.png', dpi=300, bbox_inches='tight')
    print("\n📊 Visualization saved: outputs/04_text_statistics.png")
    plt.close()
    
    return df


# ==============================================================================
# 5. WORD CLOUD GENERATION
# ==============================================================================

def generate_wordclouds(df, top_n_classes=5):
    """
    Generate word clouds for top N disease classes
    
    Args:
        df (pd.DataFrame): Dataset
        top_n_classes (int): Number of top classes to visualize
    """
    print("\n" + "=" * 80)
    print(f"STEP 7: WORD CLOUD GENERATION (Top {top_n_classes} Classes)")
    print("=" * 80)
    
    # Get top classes
    top_classes = df['LABEL'].value_counts().head(top_n_classes).index
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, disease_class in enumerate(top_classes):
        print(f"\n🌥️  Generating word cloud for: {disease_class}")
        
        # Filter data
        class_data = df[df['LABEL'] == disease_class]
        text = ' '.join(class_data['combined_text'].values)
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=800, height=400,
            background_color='white',
            colormap='viridis',
            max_words=100,
            relative_scaling=0.5,
            min_font_size=10
        ).generate(text)
        
        # Plot
        axes[idx].imshow(wordcloud, interpolation='bilinear')
        axes[idx].set_title(f'{disease_class}\n({len(class_data)} samples)', 
                           fontsize=10, fontweight='bold')
        axes[idx].axis('off')
    
    # Hide extra subplot
    axes[5].axis('off')
    
    plt.tight_layout()
    plt.savefig('outputs/05_wordclouds.png', dpi=300, bbox_inches='tight')
    print("\n📊 Visualization saved: outputs/05_wordclouds.png")
    plt.close()


# ==============================================================================
# 6. TOP KEYWORDS EXTRACTION
# ==============================================================================

def extract_top_keywords(df, top_n=20):
    """
    Extract and visualize top keywords
    
    Args:
        df (pd.DataFrame): Dataset
        top_n (int): Number of top keywords to show
    """
    print("\n" + "=" * 80)
    print(f"STEP 8: TOP KEYWORDS EXTRACTION (Top {top_n})")
    print("=" * 80)
    
    # Combine all text
    all_text = ' '.join(df['combined_text'].values)
    
    # Simple tokenization (remove punctuation, convert to lowercase)
    words = re.findall(r'\b[a-z]{3,}\b', all_text.lower())
    
    # Remove common stop words
    stop_words = {'the', 'and', 'with', 'some', 'areas', 'are', 'but', 'not',
                  'appear', 'present', 'visible', 'along', 'near', 'from'}
    words = [w for w in words if w not in stop_words]
    
    # Count frequencies
    word_freq = Counter(words)
    top_words = word_freq.most_common(top_n)
    
    print(f"\n🔑 Top {top_n} keywords:")
    for i, (word, count) in enumerate(top_words, 1):
        print(f"   {i:2d}. {word:20s}: {count:5d} occurrences")
    
    # Visualization
    words_list, counts_list = zip(*top_words)
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(words_list)), counts_list, color='teal')
    plt.yticks(range(len(words_list)), words_list)
    plt.xlabel('Frequency')
    plt.title(f'Top {top_n} Keywords in Plant Disease Descriptions')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/06_top_keywords.png', dpi=300, bbox_inches='tight')
    print("\n📊 Visualization saved: outputs/06_top_keywords.png")
    plt.close()


# ==============================================================================
# 7. CORRELATION ANALYSIS
# ==============================================================================

def analyze_correlations(df):
    """
    Analyze correlations between features
    
    Args:
        df (pd.DataFrame): Dataset
    """
    print("\n" + "=" * 80)
    print("STEP 9: FEATURE CORRELATION ANALYSIS")
    print("=" * 80)
    
    # Create numerical features
    severity_map = {'Low': 1, 'Medium': 2, 'High': 3}
    df['severity_numeric'] = df['SEVERITY'].map(severity_map)
    
    # Select numerical columns
    numerical_cols = ['text_length', 'word_count', 'severity_numeric']
    corr_df = df[numerical_cols].corr()
    
    print("\n📊 Correlation Matrix:")
    print(corr_df.to_string())
    
    # Visualization
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_df, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('outputs/07_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("\n📊 Visualization saved: outputs/07_correlation_heatmap.png")
    plt.close()


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """Main execution function"""
    
    # Create output directory
    import os
    os.makedirs('outputs', exist_ok=True)
    
    print("\n" + "=" * 80)
    print(" " * 20 + "PLANT DISEASE PREDICTION")
    print(" " * 15 + "Data Validation & Exploratory Analysis")
    print("=" * 80)
    
    # File path
    file_path = 'multimodal_plant_data.csv'
    
    # Execute all steps
    df = load_and_validate_data(file_path)
    check_missing_values(df)
    analyze_class_distribution(df)
    analyze_severity_distribution(df)
    df = extract_features_from_dicts(df)
    df = analyze_text_statistics(df)
    generate_wordclouds(df, top_n_classes=5)
    extract_top_keywords(df, top_n=20)
    analyze_correlations(df)
    
    # Save processed data
    output_file = 'processed_data_with_features.csv'
    df.to_csv(output_file, index=False)
    print("\n" + "=" * 80)
    print(f"💾 Processed data saved: {output_file}")
    print("=" * 80)
    
    print("\n✅ Data validation and EDA completed successfully!")
    print(f"📁 All visualizations saved in 'outputs/' directory")
    print("\n🎯 Next step: Run preprocessing and model training script")


if __name__ == "__main__":
    main()
