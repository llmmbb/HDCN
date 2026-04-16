import torch.nn as nn

from .models.hdcn_model import HDCNModel
from .losses.hdcn_loss import HDCNLoss


MODEL_HYPER_PARAMS = {
    "d_model": 512,
    "d_ff": 2048,
    "n_heads": 8,
    "e_layers": 3,
    "dropout": 0.1,

    "patch_len": 16,
    "stride": 8,

    "seq_len": 96,
    "pred_len": 24,

    "series_dim": 1,

    "wavelet_levels": 2,
    "use_wavelet": True,

    "loss": "MAE",

    "alpha": 0.5,   
    "beta": 0.3,    
    "gamma": 0.1,   
    "delta": 0.1,  

    "lr": 0.001,
    "batch_size": 64,
    "num_epochs": 50,
}


class HDCN(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.model = HDCNModel(config)
        self.criterion = HDCNLoss(config)

    def forward(self, x, y=None):
        outputs = self.model(x)

        if self.training and y is not None:
            loss_dict = self.criterion(outputs, y)
            outputs.update(loss_dict)

        return outputs