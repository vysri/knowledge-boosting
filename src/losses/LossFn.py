import torch
import torch.nn as nn
from asteroid.losses.sdr import SingleSrcNegSDR

from src.losses.sisdr_with_pit import SISDR_with_PIT_Loss



class LossFn(nn.Module):
    def __init__(self, name='snr', **kwargs) -> None:
        super().__init__()        
        if name == 'sisdr':
            self.loss_fn = SingleSrcNegSDR('sisdr')
        elif name == 'snr':
            self.loss_fn = SingleSrcNegSDR('snr')
        elif name == 'sdsdr':
            self.loss_fn = SingleSrcNegSDR('sdsdr')
        else:
            assert 0, f"Invalid loss function used: Loss {self.loss_fn} not found"

    def forward(self, est: torch.Tensor, gt: torch.Tensor, **kwargs):
        """
        est: (B, C, T)
        gt: (B, C, T)
        """
        if est is not None:
            B, C, T = est.shape
        if len(kwargs) > 0:
            if est is not None:
                est = est.reshape(-1, T)
            if gt is not None:
                gt = gt.reshape(-1, T)
            batch_losses = self.loss_fn(est, gt, **kwargs)
            if 'gt1' in kwargs:
                batch_losses, out1, out2 = batch_losses
                return torch.mean(batch_losses), out1, out2
        else:
            batch_losses = self.loss_fn(est.reshape(-1, T), gt.reshape(-1, T))
        return torch.mean(batch_losses)
