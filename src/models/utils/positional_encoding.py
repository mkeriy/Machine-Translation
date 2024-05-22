import torch
from torch.nn import nn
from math import log


class PositionalEncoding(torch.nn.Module):
    def __init__(self, emb_size: int, dropout: float = 0.1, maxlen: int = 5000):
        """
        emb_size - размер эмбеддингов
        maxlen - длинна контекста
        """
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(maxlen, emb_size)

        k = torch.arange(0, maxlen).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, emb_size, 2) * -(log(10000.0) / emb_size))

        pe[:, 0::2] = torch.sin(k * div_term)
        pe[:, 1::2] = torch.cos(k * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)
