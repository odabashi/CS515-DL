import torch.nn as nn
from models.bidirectional_stock import BiDirectionalStockModel
from models.stock_gru import StockGRU
from models.stock_lstm import StockLSTM
from models.moving_avg_feat import MovingAverageFeatures


__all__ = [
    "BiDirectionalStockModel",
    "StockGRU",
    "StockLSTM",
    "MovingAverageFeatures",
    "build_model",
]


def build_model(name: str, **kwargs) -> nn.Module:
    """
    Instantiate a model by name.

    Parameters:
        name : one of {"lstm", "gru", "bilstm", "bigru"}
        **kwargs : forwarded to the model constructor (override any default)

    Returns:
        nn.Module: the instantiated model

    Raises:
        ValueError: if an unknown model name is provided
    """
    name = name.lower()
    if name == "lstm":
        return StockLSTM(**kwargs)
    elif name == "gru":
        return StockGRU(**kwargs)
    elif name == "bilstm":
        return BiDirectionalStockModel(cell_type="LSTM", **kwargs)
    elif name == "bigru":
        return BiDirectionalStockModel(cell_type="GRU", **kwargs)
    else:
        raise ValueError(f"Unknown model name '{name}'. Choose from: lstm, gru, bilstm, bigru")
