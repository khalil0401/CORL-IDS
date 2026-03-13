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
                 dropout: float = 0.0):
        super().__init__()
        # Rule 5: LSTM Architecture
        # single-layer LSTM, 64 hidden units
        self.hidden_size = 64
        self.latent_dim  = latent_dim

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )

        # Rule 2: Remove extras (LayerNorm/Attention)
        self.projection = nn.Linear(64, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor (batch, seq_len, input_dim)

        Returns
        -------
        z : Tensor (batch, latent_dim)
        """
        # Take final hidden state h_T as representation
        _, (h_n, _) = self.lstm(x)
        h_final = h_n[-1] # (batch, 64)

        # Linear projection to latent space
        z = self.projection(h_final)
        return z


def build_encoder(input_dim: int,
                  hidden_size: int = 64,
                  latent_dim: int = 32,
                  device: str = "cpu") -> LSTMEncoder:
    enc = LSTMEncoder(input_dim, hidden_size, latent_dim)
    return enc.to(device)
