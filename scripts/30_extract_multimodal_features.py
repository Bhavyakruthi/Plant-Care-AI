
import os
import torch
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from transformers import BertTokenizer, BertModel
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.models.image import ImageModel
from backend.models.text import HybridBioBERTClassifier

def clean_text(text):
    if pd.isna(text): return ""
    import re
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s.\-\']', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_features():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")

    # 1. Load Data
    print("📂 Loading datasets...")
    df = pd.read_csv('data/multimodal_plant_data.csv')
    with open('data/image_mappings.pkl', 'rb') as f:
        image_mapping = pickle.load(f)
    with open('data/preprocessed_data.pkl', 'rb') as f:
        prep_data = pickle.load(f)
    
    label_encoder = prep_data['label_encoder']
    num_classes = len(label_encoder.classes_)
    
    # 2. Initialize Models
    print("🤖 Initializing models...")
    
    # Image Feature Extractor (ResNet18 backbone)
    # We don't need the QNN part for extraction, just the ResNet backbone
    img_model = ImageModel() 
    feature_extractor = img_model.feature_extractor.to(device)
    img_transform = img_model.transform
    
    # Text Feature Extractor (BioBERT)
    tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract')
    # Use the same weights as our trained hybrid model if available, 
    # but the base model is also fine for feature extraction if we haven't fine-tuned BERT heavily yet.
    # Actually, we SHOULD use our trained weights if we want the best features.
    bert_path = 'models/text/best_hybrid_bert_model.pth' # Check if this path is correct
    if not os.path.exists(bert_path):
        bert_path = 'best_hybrid_bert_model.pth' # Alternative root path
        
    num_struct = prep_data['X_train_structured'].shape[1]
    text_model = HybridBioBERTClassifier(num_classes, num_struct).to(device)
    
    if os.path.exists(bert_path):
        text_model.load_state_dict(torch.load(bert_path, map_location=device))
        print(f"✅ Loaded trained BioBERT weights from {bert_path}")
    else:
        print("⚠️  Trained BioBERT weights not found. Using pre-trained weights.")
    
    bert_backbone = text_model.bert.to(device)
    bert_backbone.eval()
    feature_extractor.eval()

    # 3. Process Paired Data
    processed_data = []
    
    print(f"⚡ Extracting features for {len(df)} samples...")
    
    batch_size = 32
    for i in tqdm(range(0, len(df), batch_size)):
        batch_df = df.iloc[i : i + batch_size]
        
        # --- Text Features ---
        texts = []
        for _, row in batch_df.iterrows():
            combined = clean_text(row['MORPHOLOGY']) + " " + \
                       clean_text(row['LESIONS']) + " " + \
                       clean_text(row['DISTRIBUTION'])
            texts.append(combined)
            
        inputs = tokenizer(texts, return_tensors='pt', truncation=True, padding=True, max_length=128).to(device)
        with torch.no_grad():
            bert_outputs = bert_backbone(**inputs)
            text_feats = bert_outputs.pooler_output.cpu().numpy()
            
        # --- Image Features ---
        img_feats_batch = []
        valid_batch_indices = []
        
        for idx_in_batch, (_, row) in enumerate(batch_df.iterrows()):
            # Construct path directly: dataset/Image Dataset/PlantVillage/{LABEL}/{FILENAME}
            # Note: We need to handle potential naming mismatches if they exist, but the folder list looked standard.
            label = row['LABEL']
            filename = row['FILENAME']
            
            # Base data path
            base_path = os.path.join("dataset", "Image Dataset", "PlantVillage", label, filename)
            
            # Absolute path
            img_path = os.path.abspath(base_path)
            
            if os.path.exists(img_path):
                try:
                    with Image.open(img_path) as img:
                        img_t = img_transform(img.convert('RGB')).unsqueeze(0).to(device)
                        with torch.no_grad():
                            feat = feature_extractor(img_t).squeeze(-1).squeeze(-1)
                            img_feats_batch.append(feat.cpu().numpy()[0])
                            valid_batch_indices.append(idx_in_batch)
                except Exception as e:
                    print(f"Error processing image {img_path}: {e}")
            else:
                if i == 0 and idx_in_batch == 0:
                    print(f"⚠️  Image not found at: {img_path}")
        
        # --- Store Sync'd Pairs ---
        labels = label_encoder.transform(batch_df['LABEL'].iloc[valid_batch_indices])
        
        for v_idx, img_feat in zip(valid_batch_indices, img_feats_batch):
            processed_data.append({
                'text_feat': text_feats[v_idx],
                'image_feat': img_feat,
                'label': labels[valid_batch_indices.index(v_idx)]
            })

    # 4. Save
    print(f"✅ Extracted paired features for {len(processed_data)} samples.")
    output_path = 'data/multimodal_features_attention.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(processed_data, f)
    print(f"💾 Saved to {output_path}")

if __name__ == "__main__":
    extract_features()
