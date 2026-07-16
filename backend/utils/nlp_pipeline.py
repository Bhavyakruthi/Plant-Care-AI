"""
Sophisticated NLP Preprocessing Pipeline
========================================

Implements a multi-stage cleaning process for plant disease symptoms:
1. JSON Strip: Remove braces, quotes, and structural keys.
2. Noise Removal: Strip [NA], None, and Generic placeholders.
3. Stop Word Removal: Filter out common non-diagnostic English words.
4. Lemmatization: Map words to their dictionary base form (using SpaCy).
5. Normalization: Lowercasing and punctuation cleanup.

Author: NLP Project Team
"""

import re
import json
import spacy
import nltk
import pandas as pd
from nltk.corpus import stopwords

# Ensure NLTK stopwords are available
try:
    STOP_WORDS = set(stopwords.words('english'))
except:
    nltk.download('stopwords')
    STOP_WORDS = set(stopwords.words('english'))

# Load SpaCy for lemmatization
try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except:
    # If not found, fall back to basic cleaning
    nlp = None

def preprocess_text(text, remove_stops=True, lemmatize=True):
    """
    Complete NLP preprocessing pipeline.
    """
    if not isinstance(text, str) or not text.strip():
        return "Not available"

    # 1. JSON Cleaning (similar to previous utility)
    if text.strip().startswith('{') and text.strip().endswith('}'):
        try:
            data = json.loads(text)
            text = " ".join([f"{k} {v}" for k, v in data.items() if v and str(v).lower() not in ["[na]", "none", "null"]])
        except:
            pass
            
    # 2. Basic Noise Stripping
    text = re.sub(r'[\{\}\"\']', '', text)
    text = re.sub(r'\[NA\]|\[NONE\]|None|null', '', text, flags=re.IGNORECASE)
    
    # 3. Normalization
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text) # Keep only letters
    
    # 4. SpaCy processing (Lemmatization + Tokenization)
    if nlp and (remove_stops or lemmatize):
        doc = nlp(text)
        tokens = []
        for token in doc:
            if remove_stops and token.text in STOP_WORDS:
                continue
            if lemmatize:
                tokens.append(token.lemma_)
            else:
                tokens.append(token.text)
        text = " ".join(tokens)
    else:
        # Fallback to basic whitespace cleanup
        tokens = text.split()
        if remove_stops:
            tokens = [t for t in tokens if t not in STOP_WORDS]
        text = " ".join(tokens)
        
    # 5. Final Cleanup
    text = re.sub(r'\s+', ' ', text).strip()
    return text if text else "Not available"

def get_cleaned_diagnostic_string(row):
    """
    Specific logic for the plant multimodal dataset rows.
    """
    components = []
    for col in ['MORPHOLOGY', 'LESIONS', 'DISTRIBUTION', 'SEVERITY']:
        if col in row and pd.notna(row[col]):
            cleaned = preprocess_text(str(row[col]))
            if cleaned != "Not available":
                components.append(cleaned)
    
    res = ". ".join(components)
    return res if res.strip() else "Plant disease symptoms description not available."

if __name__ == "__main__":
    import pandas as pd # Just for testing local call
    test_text = '{"edges": "Irregular, necrotic, and curled downwards.", "veins": "Visible but yellowing"}'
    print(f"Original: {test_text}")
    print(f"Cleaned:  {preprocess_text(test_text)}")
