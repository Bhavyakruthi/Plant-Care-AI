"""
BioBERT Model Overhaul - High Accuracy Re-training
==================================================

Implements a proper training methodology for the text-based diagnostic model:
1. Data Preprocessing: Using the new NLP pipeline (Lemmatization, No Stop Words).
2. Model Architecture: HybridBioBERTClassifier (Text + Structured features).
3. Optimization: AdamW with weight decay and learning rate scheduling.
4. Evaluation: Validation-based checkpointing (F1-Score).

Author: NLP Project Team
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
import pickle

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from backend.models.text import HybridBioBERTClassifier
from backend.utils.nlp_pipeline import preprocess_text

class PlantTextDataset(Dataset):
    def __init__(self, texts, structured_features, labels, tokenizer, max_len=128):
        self.texts = texts
        self.structured_features = structured_features
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
        struct = self.structured_features[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'structured_features': torch.tensor(struct, dtype=torch.float),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train_epoch(model, data_loader, optimizer, device, scheduler):
    model.train()
    losses = []
    correct_predictions = 0

    for d in tqdm(data_loader, desc="Training"):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        struct_features = d["structured_features"].to(device)
        labels = d["labels"].to(device)

        outputs = model(input_ids, attention_mask, struct_features)
        _, preds = torch.max(outputs, dim=1)
        loss = nn.CrossEntropyLoss()(outputs, labels)

        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)

def eval_model(model, data_loader, device):
    model.eval()
    losses = []
    correct_predictions = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            struct_features = d["structured_features"].to(device)
            labels = d["labels"].to(device)

            outputs = model(input_ids, attention_mask, struct_features)
            _, preds = torch.max(outputs, dim=1)
            loss = nn.CrossEntropyLoss()(outputs, labels)

            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses), all_preds, all_labels

def main():
    # 1. Config
    SAVE_DIR = 'd:/COLLEGE FILES/ALL SUBJECTS/SEMESTER 6/Natural Languge Processing/LANGUAGE_MODEL_PROJECT/models/text/overhaul'
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 16
    EPOCHS = 20  # Total target epochs (e.g. initial 10 + continue 10)
    RESUME = True # Set to True to continue from best checkpoint
    LEARNING_RATE = 1e-5 if RESUME else 2e-5 # Lower LR for fine-tuning
    MAX_LEN = 128
    
    # 2. Load Data
    CLEANED_DATA_PATH = 'd:/COLLEGE FILES/ALL SUBJECTS/SEMESTER 6/Natural Languge Processing/LANGUAGE_MODEL_PROJECT/data/cleaned_multimodal_plant_data.csv'
    
    if not os.path.exists(CLEANED_DATA_PATH):
        print(f"❌ Cleaned dataset not found at {CLEANED_DATA_PATH}")
        print("Please run scripts/44_persist_cleaned_dataset.py first.")
        return

    print(f"Loading cleaned dataset from {CLEANED_DATA_PATH}...")
    df = pd.read_csv(CLEANED_DATA_PATH)
    
    # Use the pre-cleaned text
    df['cleaned_text'] = df['cleaned_text'].fillna("Not available")
    
    # Extract real structured features
    print("Extracting structured features...")
    from backend.utils.feature_engineering import extract_structured_features
    struct_feats = extract_structured_features(df)
    print(f"✅ Extracted structured features with shape: {struct_feats.shape}")
    
    # Label encoding
    print("Encoding labels...")
    le = LabelEncoder()
    df['label_idx'] = le.fit_transform(df['LABEL'])
    
    # Split
    df_train, df_val = train_test_split(df, test_size=0.1, random_state=42, stratify=df['label_idx'])
    
    # 3. Data Loaders
    tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract')
    
    train_dataset = PlantTextDataset(
        texts=df_train.cleaned_text.to_numpy(),
        structured_features=struct_feats[df_train.index],
        labels=df_train.label_idx.to_numpy(),
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )
    
    val_dataset = PlantTextDataset(
        texts=df_val.cleaned_text.to_numpy(),
        structured_features=struct_feats[df_val.index],
        labels=df_val.label_idx.to_numpy(),
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # 4. Initialize Model
    model = HybridBioBERTClassifier(num_classes=len(le.classes_), num_structured_features=struct_feats.shape[1])
    
    if RESUME:
        best_model_path = os.path.join(SAVE_DIR, 'best_biobert_overhaul.pth')
        if os.path.exists(best_model_path):
            print(f"🔄 Resuming from checkpoint: {best_model_path}")
            model.load_state_dict(torch.load(best_model_path, map_location='cpu'))
        else:
            print(f"⚠️ Checkpoint {best_model_path} not found. Starting from scratch.")
            
    model = model.to(DEVICE)
    
    # Differential Learning Rates & Weight Decay to prevent overfitting
    # BERT needs a very small LR, the Head can take a larger one
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.bert.named_parameters()], 'lr': 1e-6, 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if 'bert' not in n], 'lr': 1e-4, 'weight_decay': 0.01}
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps//10, num_training_steps=total_steps)
    
    # 5. Training Loop
    best_accuracy = 0
    patience = 3
    no_improve = 0
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print("-" * 10)
        
        train_acc, train_loss = train_epoch(model, train_loader, optimizer, DEVICE, scheduler)
        print(f"Train loss {train_loss:.4f} accuracy {train_acc:.4f}")
        
        val_acc, val_loss, preds, actuals = eval_model(model, val_loader, DEVICE)
        print(f"Val loss   {val_loss:.4f} accuracy {val_acc:.4f}")
        
        if val_acc > best_accuracy:
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'best_biobert_overhaul.pth'))
            best_accuracy = val_acc
            no_improve = 0
            print(f"✅ New best model saved with accuracy: {val_acc:.4f}")
        else:
            no_improve += 1
            print(f"ℹ️ No improvement for {no_improve} epoch(s).")
            
        if no_improve >= patience:
            print(f"🛑 Early stopping triggered at epoch {epoch + 1}")
            break
            
    # Save label mapping and metadata
    metadata = {
        'label_encoder': le,
        'num_struct': struct_feats.shape[1],
        'best_acc': best_accuracy
    }
    with open(os.path.join(SAVE_DIR, 'overhaul_metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
        
    print("\nTraining complete!")
    print(f"Best Val Accuracy: {best_accuracy:.4f}")

if __name__ == "__main__":
    main()
