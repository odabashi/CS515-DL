import torch.nn as nn
import torch.nn.functional as F


def get_activation(name: str) -> nn.Module:

    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "leaky_relu":
        return nn.LeakyReLU()
    elif name == "elu":
        return nn.ELU()
    elif name == "tanh":
        return nn.Tanh()
    else:
        raise ValueError("Unknown activation")


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, hidden_activation, num_classes,
                 enable_dropout, dropout, enable_batch_norm):
        super().__init__()
        layers = []
        in_dim = input_size
        for h in hidden_sizes:
            hidden_layer: list = [nn.Linear(in_dim, h)]
            if enable_batch_norm:
                hidden_layer.append(nn.BatchNorm1d(h))
            hidden_layer.append(get_activation(hidden_activation))
            if enable_dropout:
                hidden_layer.append(nn.Dropout(dropout))

            layers += hidden_layer
            in_dim = h

        # Add Output Layer
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)   # flatten (B, 1, 28, 28) → (B, 784)
        return self.net(x)
