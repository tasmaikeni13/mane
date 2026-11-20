"""A ~38M-parameter decoder-only Transformer language model.

Although it is built from ``nn.TransformerEncoder`` layers, a causal
(subsequent) mask makes it autoregressive, so it behaves as a decoder-only LM
(the same construction as nanoGPT-style models). Input/output embeddings are
tied. Defaults: vocab 50257 (GPT-2), d_model 512, 8 heads, 4 layers.
"""

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding, shape (max_len, 1, d_model)."""

    def __init__(self, d_model, max_len=1024):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: SeqLen x Batch x d_model
        x = x + self.pe[:x.size(0)]
        return x

class LanguageModel(nn.Module):
    def __init__(self, vocab_size=50257, d_model=512, nhead=8, num_layers=4, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        # Tie weights
        self.fc_out.weight = self.embedding.weight
        
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()

    def generate_square_subsequent_mask(self, sz):
        # Additive attention mask: position i may attend to positions j <= i
        # (0.0), future positions (j > i) are masked out with -inf.
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        # src: Batch x SeqLen
        src = src.transpose(0, 1) # SeqLen x Batch
        mask = self.generate_square_subsequent_mask(src.size(0)).to(src.device)
        
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer(src, mask)
        output = self.fc_out(output)
        
        return output.transpose(0, 1) # Batch x SeqLen x Vocab
