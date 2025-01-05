"""Feed forward module part of the transformer architecture."""
import torch.nn as nn

class FeedForward(nn.Module):
    """Simple feed forward network part of the transformer architecture

    Parameters
    ----------
    n_embed : int
        Model latent dimension
    dropout : float
        Dropout rate
    """
    def __init__(self, n_embed: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, n_embed*4),
            nn.ReLU(),
            nn.Linear(4*n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
