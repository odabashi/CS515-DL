import torch
import torch.nn as nn
from models.moving_avg_feat import MovingAverageFeatures


class StockGRU(nn.Module):
    """
    Stacked GRU -> Dropout -> Fully-connected output layer.

    GRU is a lighter alternative to LSTM: it merges the forget and input gates into a single "update gate" and 
    removes the separate cell state. This gives fewer parameters and often trains faster.

    GRU equations (from the assignment):
      z_t = σ(W_z [h_{t-1}, x_t] + b_z)                # update gate
      r_t = σ(W_r [h_{t-1}, x_t] + b_r)                # reset gate
      h̃_t = tanh(W_h [r_t ⊙ h_{t-1}, x_t] + b_h)       # candidate hidden
      h_t = (1 − z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t            # hidden state update

    The architecture mirrors StockLSTM exactly so we can do a fair comparison.
    """

    def __init__(self, input_size: int = 4, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2,
                 output_size: int = 5, use_ma_features: bool = True):
        super().__init__()

        self.use_ma_features = use_ma_features

        if use_ma_features:
            self.ma_block = MovingAverageFeatures(input_size, out_channels=4)
            gru_input_dim = input_size + 4
        else:
            gru_input_dim = input_size

        # Stacked GRU
        self.gru = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:
            (batch, T, F)
        Output:
            (batch, D)
        """
        if self.use_ma_features:
            x = self.ma_block(x)                        # (batch, T, F+4)

        # gru_out shape: (batch, T, hidden_size)
        # h_n shape: (num_layers, batch, hidden_size)
        gru_out, h_n = self.gru(x)

        # We only care about the hidden state of the LAST layer at the LAST time step. This is the model's
        # compressed summary of the whole window.
        last_hidden = h_n[-1]                           # (batch, hidden_size)

        # Dropout for regularization, then project to D output values.
        out = self.dropout(last_hidden)
        out = self.fc(out)                              # (batch, D)
        return out
