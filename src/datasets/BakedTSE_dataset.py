"""
Torch dataset object for synthetically rendered spatial data.
"""

import os, glob
from pathlib import Path
import random
import logging
import warnings

import pandas as pd
import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as AT
import scaper
import numpy as np

import json


# Ignore scaper normalization warnings
warnings.filterwarnings(
    "ignore", message="Soundscape audio is clipping!")
warnings.filterwarnings(
    "ignore", message="Peak normalization applied")
warnings.filterwarnings(
    "ignore", message="Scale factor for peak normalization is extreme")

class PRASimulator():
    def __init__(self):
        pass

    def simulate():
        pass

class BakedTSE_dataset(Dataset):
    def __init__(self, mixtures_dir, dset, num_samples,
                 chunks_delay=0, ms_chunk=1, sr=None, resample_rate=None,
                 rt60_min=None, rt60_max=None) -> None:
        super().__init__()
        assert dset in ['train', 'val', 'test'], \
            "`dset` must be one of ['train', 'val', 'test']"

        self.dset = dset
        self.mixtures_dir = os.path.join(mixtures_dir, dset)
        
        self.sr = sr
        self.resample_rate = resample_rate
        self.chunks_delay = chunks_delay
        self.ms_chunk = ms_chunk

        logging.info(f"Loading dataset: {dset} {sr=} {resample_rate=} ...")
        logging.info(f"- Mixtures directory: {mixtures_dir}")

        self.samples = sorted(list(Path(self.mixtures_dir).glob('[0-9]*')))
        self.samples = self.samples[0:num_samples]
                     
        if resample_rate is not None:
            self.resampler = AT.Resample(sr, resample_rate)
            self.sr = resample_rate
        else:
            self.resampler = lambda a: a
            self.sr = sr

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_dir = self.samples[idx].__str__()

        # Load Audio
        src1_path = os.path.join(sample_dir, 'source00.wav')
        # print("SOURCE 1", src1_path, os.path.exists(src1_path))
        target1, sample_rate = torchaudio.load(src1_path)
        embed1_path = os.path.join(sample_dir, 'spk_id00.pt')
        spk_id1 = torch.load(embed1_path)

        src2_path = os.path.join(sample_dir, 'source01.wav')
        target2, sample_rate = torchaudio.load(src2_path)
        embed2_path = os.path.join(sample_dir, 'spk_id01.pt')
        spk_id2 = torch.load(embed2_path)

        # Create mixture by summing the two
        mixture = target1 + target2

        if np.random.rand() < 0.5:
            target = self.resampler(target1)
            spk_id = spk_id1
        else:
            target = self.resampler(target2)
            spk_id = spk_id2
        # Resample
        mixture = self.resampler(mixture)

        _mixture = torch.roll(mixture, self.ms_chunk*self.chunks_delay, 1)
        _mixture[..., 0:self.ms_chunk*self.chunks_delay] = 0
        # print("spk_id: ", spk_id.shape)
        inputs = {
            'mixture': mixture,
            '_mixture': _mixture,
            'sample_dir': sample_dir,
            'spk_id': spk_id[0]
        }

        targets = {
            'target': target,
        }

        return inputs, targets
