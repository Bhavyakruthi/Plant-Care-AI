import os
import sys
import pandas as pd

# Add root to path
sys.path.append(os.getcwd())

try:
    from backend.utils.nlp_pipeline import preprocess_text, get_cleaned_diagnostic_string
    print("✅ NLP Pipeline imported successfully")
except ImportError as e:
    print(f"❌ Error importing NLP Pipeline: {e}")
    # Try alternative path
    sys.path.append(os.path.join(os.getcwd(), 'backend'))
    from utils.nlp_pipeline import preprocess_text, get_cleaned_diagnostic_string
    print("✅ NLP Pipeline imported via fallback path")

def verify_preprocessing():
    DATA_PATH = 'data/multimodal_plant_data.csv'
    if not os.path.exists(DATA_PATH):
        print(f"❌ Dataset not found at {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    print(f"Total samples: {len(df)}")
    
    # Take a few samples
    samples = df.sample(5, random_state=42)
    
    print("\n" + "="*80)
    print("PREPROCESSING VERIFICATION SAMPLES")
    print("="*80)
    
    for i, (_, row) in enumerate(samples.iterrows()):
        print(f"\n--- SAMPLE {i+1} (Label: {row['LABEL']}) ---")
        
        # Original morphology
        raw_morph = row['MORPHOLOGY']
        print(f"RAW MORPHOLOGY: {raw_morph}")
        
        # Cleaned morphology
        clean_morph = preprocess_text(str(raw_morph))
        print(f"CLEANED MORPH:  {clean_morph}")
        
        # Full Combined & Cleaned Diagnostic String
        full_diagnostic = get_cleaned_diagnostic_string(row)
        print(f"FULL CLEANED:   {full_diagnostic}")
        
    print("\n" + "="*80)

if __name__ == "__main__":
    verify_preprocessing()
