import torch
import torch.nn as nn

class PriceTransformer(nn.Module):
    def __init__(self, input_size, d_model=128, nhead=4, num_layers=2):
        super(PriceTransformer, self).__init__()

        # Projects your raw feature size into the higher-dimensional Transformer space
        self.input_proj = nn.Linear(input_size, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            dropout=0.1
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Output layer for 3 classes: 0 (Down), 1 (Neutral), 2 (Up)
        self.fc = nn.Linear(d_model, 3)

    def forward(self, x):
        x = self.input_proj(x)              # Shape: (Batch, Seq_len, d_model)
        x = self.transformer(x)             # Shape: (Batch, Seq_len, d_model)
        x = x[:, -1, :]                     # Isolate the very last timestep
        x = self.fc(x)                      # Shape: (Batch, 3)
        return torch.softmax(x, dim=1)