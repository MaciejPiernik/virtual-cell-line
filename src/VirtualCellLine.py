import torch
import torch.nn as nn

from BaseSimulator import BaseSimulator

class VirtualCellLine(BaseSimulator):
    def __init__(self, num_genes, embedding_dim=128, num_heads=8, num_transformer_layers=6, dropout_rate=0.1):
        super(VirtualCellLine, self).__init__()
        
        self.num_genes = num_genes
        
        # Initial embedding layer
        self.embedding = nn.Linear(num_genes, embedding_dim)
        
        # Transformer layers
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout_rate,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_transformer_layers)
        
        # Output layer
        self.output_layer = nn.Linear(embedding_dim, num_genes)
        
    def forward(self, x, mask=None):
        # x: [batch_size, num_genes]
        # mask: [batch_size, num_genes] binary mask where 1 indicates masked (unknown) values
        
        if mask is not None:
            x = x * (1 - mask)  # Apply mask
        
        # Embed
        x = self.embedding(x)

        # Pass through transformer
        x = self.transformer(x)
        
        # Output layer
        output = self.output_layer(x)
        
        return output
