import torch
import torch.nn as nn


class MovingAverageFeatures(nn.Module):
    """
    Optional auxiliary feature block (mentioned in the assignment footnote).

    Applies a 1-D convolution with a learned kernel along the time axis to produce a smoothed (moving-average-like) 
    version of the Close price.

    Why this can help
    -----------------
    LSTMs already learn temporal patterns, but providing an explicit short-term trend signal as an extra input 
    channel can reduce the burden on the recurrent layers and speed up convergence.

    Input  shape: (batch, T, F)
    Output shape: (batch, T, F + out_channels) original features + MA features
    """

    def __init__(self, in_channels: int = 4, out_channels: int = 4, kernel_size: int = 5):
        super().__init__()
        # Conv1d expects (batch, channels, length) so we transpose inside forward().
        # padding='same' keeps the time dimension length unchanged.
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,   # manual 'same' padding
            bias=False
        )
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, T, F)
        x_t = x.permute(0, 2, 1)            # -> (batch, F, T)              Conv1d wants (B, C, L)
        ma = self.conv(x_t)                 # -> (batch, out_channels, T)
        ma = ma.permute(0, 2, 1)            # -> (batch, T, out_channels)
        ma = self.norm(ma)
        return torch.cat([x, ma], dim=-1)  # -> (batch, T, F + out_channels)
