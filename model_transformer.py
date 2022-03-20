from typing import Dict

import torch
from torch.nn import Embedding
import math

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class SeqSlotTransformerClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        num_class: int,
        n_head: int = 5
    ) -> None:
        super(SeqSlotTransformerClassifier, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        # TODO: model architecture
        self.embedding_dim = embeddings.size(1) # 300
        
        self.pos_encoder = PositionalEncoding(
            d_model=self.embedding_dim, 
            dropout=dropout
            )

        self.encoder = torch.nn.Transformer(
            d_model=self.embedding_dim, 
            nhead=n_head,
            dim_feedforward=hidden_size,
            dropout=dropout,
            #batch_first=True
            )

        self.encoder_out_dim = self.embedding_dim
        
        hidden_size2 = int(hidden_size/2)
        self.fc_layers = torch.nn.Sequential(
            torch.nn.Linear(self.encoder_out_dim, hidden_size),
            torch.nn.Dropout(dropout),
            torch.nn.Sigmoid(),

            torch.nn.Linear(hidden_size, hidden_size2),
            torch.nn.Dropout(dropout),
            torch.nn.Sigmoid(),

            torch.nn.Linear(hidden_size2, num_class),
        )

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        batch = self.embed(batch)
        print("bos encoding")
        print(batch[0][0]) 
        # print("First word encoding")
        # print(batch[0][1]) 
        # Positional Encoding
        batch = batch.transpose(0,1)
        batch = self.pos_encoder(batch)

        #tgt = batch[0:, 1:, 0:]
        tgt = batch[1:, 0:, 0:]
        batch = self.encoder(batch, tgt)
        # [batch_size, seq_len -1, embedding_dim]
        batch = batch.transpose(0,1)
        
        batch = self.fc_layers(batch)
        # [batch_size, seq_len, num_class]
        return batch
        # raise NotImplementedError