import torch
import torch.nn as nn
from models.moving_avg_feat import MovingAverageFeatures


class StockLSTM(nn.Module):
    """
    Stacked LSTM -> Dropout -> Fully-connected output layer.

    Architecture
    ------------
    Input:      (batch, T, F_hat), T time steps, F_hat features
                If use_ma_features=True, F_hat = F + out_channels (default 8)
                Otherwise F_hat = F = 4

    LSTM:       num_layers stacked LSTM cells, each with hidden_size units.
                batch_first=True means the batch dimension comes first and this is more natural and matches how we 
                build our DataLoader.

    Dropout:    applied BETWEEN layers only (PyTorch ignores it on the last layer automatically when num_layers > 1, 
                but we add an explicit one here too so it also applies after the final recurrent layer).

    Output :    Linear(hidden_size -> D) gives one return prediction per horizon.

    LSTM recap (from the assignment):
      f_t = σ(W_f [h_{t-1}, x_t] + b_f)         # forget gate
      i_t = σ(W_i [h_{t-1}, x_t] + b_i)         # input gate
      c̃_t = tanh(W_c [h_{t-1}, x_t] + b_c)      # candidate cell
      c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t           # cell state update
      o_t = σ(W_o [h_{t-1}, x_t] + b_o)         # output gate
      h_t = o_t ⊙ tanh(c_t)                     # hidden state
    """

    def __init__(self, input_size: int = 4, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2,
                 output_size: int = 5, use_ma_features: bool = True):
        super().__init__()

        self.use_ma_features = use_ma_features

        # Auxiliary moving-average feature extractor (optional)
        if use_ma_features:
            self.ma_block = MovingAverageFeatures(input_size, out_channels=4)
            lstm_input_dim = input_size + 4
        else:
            lstm_input_dim = input_size

        # Stacked LSTM
        # dropout = inside nn.LSTM applies between layers; it is ignored for a single-layer model
        # (safe to pass regardless).
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,        # input/output: (batch, seq, feature)
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Post-LSTM dropout + projection
        # We apply an explicit dropout here so regularization also acts after the final recurrent layer
        # (nn.LSTM only drops between layers).
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, T, F)

        Flow:
            1. Optionally augment features with moving-average channels.
            2. Run through stacked LSTM — we only need the last hidden state.
            3. Apply dropout for regularization.
            4. Project to D output values.

        Returns: (batch, D)
        """
        # 1. Optional MA features
        if self.use_ma_features:
            x = self.ma_block(x)                # (batch, T, F+4)

        # 2. LSTM
        # lstm_out shape: (batch, T, hidden_size)
        # h_n shape: (num_layers, batch, hidden_size)  # hidden states at t=T
        # c_n shape: (num_layers, batch, hidden_size)  # cell states at t=T
        lstm_out, (h_n, c_n) = self.lstm(x)

        # We only care about the hidden state of the LAST layer at the LAST time step. This is the model's
        # compressed summary of the whole window.
        last_hidden = h_n[-1]                           # (batch, hidden_size)

        # 3. Dropout
        out = self.dropout(last_hidden)                 # (batch, hidden_size)

        # 4. Linear projection
        out = self.fc(out)                              # (batch, D)
        return out
