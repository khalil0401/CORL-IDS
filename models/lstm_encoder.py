"""
lstm_encoder.py

Module 1: LSTM Feature Encoder

Architecture:
    Input  : (batch, seq_len, input_dim)
    LSTM   : hidden_size=64, num_layers=1, batch_first=True
    Output : final hidden state h_T → Linear(64 → 32) → z_t  (R^32)
"""

import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    """
    Temporal representation encoder for network traffic sequences.

    Parameters
    ----------
    input_dim   : number of input features per timestep
    hidden_size : LSTM hidden dimension (default 64)
    latent_dim  : output latent dimension (default 32)
    dropout     : dropout on LSTM output (default 0.0)
    """

    def __init__(self,
                 input_dim: int,
                 hidden_size: int = 64,
                 latent_dim: int = 32,
                 dropout: float = 0.0,
                 use_attention: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.latent_dim  = latent_dim
        self.use_attention = use_attention

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=dropout,
        )

        if self.use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=4,
                dropout=dropout,
                batch_first=True
            )

        self.projection = nn.Sequential(
            nn.Linear(hidden_size, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor (batch, seq_len, input_dim)

        Returns
        -------
        z : Tensor (batch, latent_dim)   — latent representation z_t
        """
        # lstm_out : (batch, seq_len, hidden_size)
        # h_n      : (num_layers, batch, hidden_size)
        lstm_out, (h_n, _) = self.lstm(x)

        if self.use_attention:
            # Query, Key, Value are all the LSTM output sequence
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            # Pool the attended sequence (mean pooling across sequence length)
            h_final = attn_out.mean(dim=1)
        else:
            # Take final layer hidden state → (batch, hidden_size)
            h_final = h_n[-1]

        # Project to latent space
        z = self.projection(h_final)   # (batch, latent_dim)
        return z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for forward, usable in inference context."""
        return self.forward(x)


def build_encoder(input_dim: int,
                  hidden_size: int = 64,
                  latent_dim: int = 32,
                  device: str = "cpu") -> LSTMEncoder:
    enc = LSTMEncoder(input_dim, hidden_size, latent_dim)
    return enc.to(device)
