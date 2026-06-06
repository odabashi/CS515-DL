import torch
import torch.nn as nn
from models.moving_avg_feat import MovingAverageFeatures


class BiDirectionalStockModel(nn.Module):
    """
    Bidirectional LSTM or GRU for turning-point / buy-signal detection.

    Why bidirectional for classification?
    --------------------------------------
    In a standard (unidirectional) RNN the hidden state at time t only summarizes the past [0 ... t]. 
    A bidirectional RNN runs TWO passes:
      - Forward pass:  h_t^ summarizes [0 ... t]
      - Backward pass: h_t^ summarizes [T ... t]
    The final representation concatenates both directions, giving the model richer context about the whole window 
    which is useful for detecting whether the window represents a turning point rather than just predicting what 
    comes next.

    NOTE: Bidirectional is only valid here because our 'window' is a fixed historical lookback. We are NOT peeking 
    into the future at test time, the backward pass still only sees what is inside the window.

    Output heads
    ------------
    This model has TWO output heads (multi-task learning):
      1. Regression head:  Linear -> (batch, D)                     # same return ratios as before
      2. Classification head: Linear -> Sigmoid -> (batch, 1)       # P(buy signal)

    The bidirectional LSTM doubles the effective hidden dimension, so the FC layers receive tensors of
    size 2 x hidden_size.

    Training note: we use BCELoss (Binary Cross Entropy) for the binary head and MSELoss (Mean Squared Error) for the
    regression head, combined as (loss = MSE + lambda x BCE).
    """
    def __init__(self, input_size: int = 4, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2,
                 output_size: int = 5, cell_type: str = "LSTM", use_ma_features: bool = True):
        super().__init__()

        assert cell_type in {"LSTM", "GRU"}, f"cell_type must be 'LSTM' or 'GRU', got '{cell_type}'"

        self.cell_type = cell_type
        self.use_ma_features = use_ma_features
        self.hidden_size = hidden_size

        if use_ma_features:
            self.ma_block = MovingAverageFeatures(input_size, out_channels=4)
            rnn_input_dim = input_size + 4
        else:
            rnn_input_dim = input_size

        # Bidirectional recurrent layer(s)
        rnn_cls = nn.LSTM if cell_type == "LSTM" else nn.GRU
        self.rnn = rnn_cls(
            input_size=rnn_input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,                                 # the key flag
        )

        self.dropout = nn.Dropout(dropout)

        # Head 1: regression: predicts D return ratios
        # Because bidirectional doubles the output, both FC heads take 2 * hidden_size as input.
        bidir_dim = 2 * hidden_size
        self.fc_reg = nn.Linear(bidir_dim, output_size)

        # Head 2: classification: predicts P(buy signal)
        # Sigmoid squashes the output to [0, 1] so it can be interpreted as a probability.
        # A threshold of 0.5 gives the binary buy/pass decision.
        self.fc_cls = nn.Linear(bidir_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (batch, T, F)

        Returns:
            Tuple containing:
                reg_out: (batch, D), predicted return ratios
                cls_out: (batch, 1), P(buy signal) in [0, 1]
        """
        if self.use_ma_features:
            x = self.ma_block(x)                        # (batch, T, F+4)

        if self.cell_type == "LSTM":
            rnn_out, (h_n, _) = self.rnn(x)
            # h_n shape: (num_layers * 2, batch, hidden_size)
            # The forward direction:    h_n[-2]
            # The backward direction:   h_n[-1]
            last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # (batch, 2*H)
        else:
            rnn_out, h_n = self.rnn(x)
            last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # (batch, 2*H)

        feat = self.dropout(last_hidden)               # (batch, 2*H)

        reg_out = self.fc_reg(feat)                    # (batch, D)
        cls_out = self.sigmoid(self.fc_cls(feat))      # (batch, 1)

        return reg_out, cls_out
