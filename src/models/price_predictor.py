import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Injects sequence order into the Transformer."""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) 
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class PriceTransformer(nn.Module):
    def __init__(self, input_size=56, d_model=256, nhead=8, num_layers=3, dropout=0.25):
        super(PriceTransformer, self).__init__()
        
        # 1. Feature Projection
        self.input_proj = nn.Linear(input_size, d_model)
        
        # 2. Input LayerNorm
        self.norm_input = nn.LayerNorm(d_model)
        
        # 3. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # 4. Transformer (Shallower to prevent overfitting)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            dropout=dropout,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers,
            enable_nested_tensor=False
        )

        # 5. Classifier
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(d_model // 2, 3) # 0: Down, 1: Neutral, 2: Up
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.norm_input(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x[:, -1, :] # Final context vector
        return self.fc(x)