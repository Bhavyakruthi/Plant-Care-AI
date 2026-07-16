import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAttention(nn.Module):
    """
    Implements Cross-Modal Attention where Text features query Image features.
    """
    def __init__(self, embed_dim, n_heads=8):
        super(CrossModalAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, text_feat, image_feat):
        # text_feat: (batch, seq_len, embed_dim) - or (batch, 1, embed_dim)
        # image_feat: (batch, seq_len, embed_dim)
        
        # Query: Text, Key/Value: Image
        attn_output, _ = self.multihead_attn(text_feat, image_feat, image_feat)
        return self.norm(attn_output + text_feat)

class MultimodalCrossAttentionModel(nn.Module):
    def __init__(self, num_classes, text_dim=768, image_dim=512, hidden_dim=512, num_structured=15):
        super(MultimodalCrossAttentionModel, self).__init__()
        
        # Projections to common space
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        
        # Input Normalization (Crucial for ResNet features)
        self.ln_text = nn.LayerNorm(text_dim)
        self.ln_image = nn.LayerNorm(image_dim)
        
        # Cross Attention (Text attends to Image)
        self.cross_attn = CrossModalAttention(hidden_dim)
        
        # Final Classifier
        # Inputs: [Attended_Text (hidden), Image (hidden), Structured (num_structured)]
        fusion_dim = (hidden_dim * 2) + num_structured
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, text_embedding, image_features, structured_features):
        # 1. Input Normalization & Projection
        t_norm = self.ln_text(text_embedding)
        i_norm = self.ln_image(image_features)
        
        t = self.text_proj(t_norm).unsqueeze(1) # (batch, 1, hidden_dim)
        i = self.image_proj(i_norm).unsqueeze(1) # (batch, 1, hidden_dim)
        
        # 2. Cross-Modal Attention
        # Text queries Image
        fused_text = self.cross_attn(t, i) # (batch, 1, hidden_dim)
        
        # 3. Concatenate and Classify
        # Combine: Attended Text + Original Image Projection + Structured Features
        combined = torch.cat((
            fused_text.squeeze(1), 
            i.squeeze(1), 
            structured_features
        ), dim=1) 
        
        return self.classifier(combined)

if __name__ == "__main__":
    # Test the architecture
    model = MultimodalCrossAttentionModel(num_classes=15)
    mock_text = torch.randn(2, 768)
    mock_image = torch.randn(2, 512)
    output = model(mock_text, mock_image)
    print(f"✅ Full Cross-Modal Output Shape: {output.shape}")
