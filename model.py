from typing import Dict

import torch
from torch.nn import Embedding


class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        # TODO: model architecture
        self.D = 2 if bidirectional==True else 1
        self.embedding_dim = embeddings.size(1)
        # self.rnn = torch.nn.RNN(self.embedding_dim, hidden_size, num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.lstm = torch.nn.LSTM(self.embedding_dim, hidden_size, num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.fc_layers = torch.nn.Sequential(
            torch.nn.Linear(self.D * hidden_size, 256),
            torch.nn.Sigmoid(),
            torch.nn.Linear(256, num_class),
        )

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        raise NotImplementedError

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        batch = self.embed(batch)
        batch, _ = self.lstm(batch, None)
        batch = batch[:,-1,:]
        batch = self.fc_layers(batch)
        return batch
        # raise NotImplementedError
