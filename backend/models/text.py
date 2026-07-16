import torch
import torch.nn as nn
from transformers import BertModel
import os
import pickle
import sys

# Ensure backend directory is in path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class HybridBioBERTClassifier(nn.Module):
    def __init__(self, num_classes, num_structured_features, dropout=0.3):
        super(HybridBioBERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract')
        bert_hidden_size = self.bert.config.hidden_size
        
        # Fusion Layer
        total_input_size = bert_hidden_size + num_structured_features
        
        self.fc1 = nn.Linear(total_input_size, 512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, input_ids, attention_mask, structured_features, return_attentions=False):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_attentions=return_attentions)
        bert_output = outputs.pooler_output
        combined = torch.cat((bert_output, structured_features), dim=1)
        
        x = self.fc1(combined)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        logits = self.classifier(x)
        
        if return_attentions:
            return logits, outputs.attentions
        return logits

class TextModel:
    def __init__(self, model_path, data_path, device=None):
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Determine if loading from overhaul metadata or legacy pkl
        if 'overhaul_metadata' in data_path:
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
            self.label_encoder = data['label_encoder']
            self.num_classes = len(self.label_encoder.classes_)
            self.num_struct = data['num_struct']
        else:
            # Legacy Path
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
            self.label_encoder = data['label_encoder']
            self.num_classes = len(self.label_encoder.classes_)
            self.num_struct = data['X_train_structured'].shape[1]
        
        # Initialize model
        self.model = HybridBioBERTClassifier(self.num_classes, self.num_struct).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        from transformers import BertTokenizer
        self.tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract')

    def predict(self, text: str):
        # Apply the same preprocessing as used in training
        from utils.nlp_pipeline import preprocess_text
        cleaned_text = preprocess_text(text)
        
        # We need structured features for the hybrid model.
        # Placeholder: using zeros for structured features during pure-text inference.
        inputs = self.tokenizer(cleaned_text, return_tensors='pt', truncation=True, padding='max_length', max_length=128).to(self.device)
        struct_feats = torch.zeros((1, self.num_struct)).to(self.device).float()
        
        with torch.no_grad():
            logits = self.model(inputs['input_ids'], inputs['attention_mask'], struct_feats)
            probs = torch.softmax(logits, dim=1)
            
        return probs.cpu().numpy()
