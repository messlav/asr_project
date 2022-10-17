from torch import nn
from torch.nn import Sequential
import torch.nn.functional as F

from hw_asr.base import BaseModel


class DeepSpeech2(BaseModel):
    def __init__(self, n_feats, n_class, fc_hidden=512, **batch):
        super().__init__(n_feats, n_class, **batch)
        self.cnn1 = nn.Conv2d(1, 32, (11, 41), stride=(2, 2))
        self.layernorm1 = nn.LayerNorm(n_feats)
        self.cnn2 = nn.Conv2d(32, 32, (11, 21), stride=(1, 2))
        self.layernorm2 = nn.LayerNorm(n_feats)
        self.cnn3 = nn.Conv2d(32, 64, (11, 21), stride=(1, 2))
        self.mlp1 = nn.Linear(n_feats, 100)

    def forward(self, spectrogram, **batch):
        shp = spectrogram.shape
        spectrogram = spectrogram.view(shp[0], 1, shp[1], shp[2])
        print(spectrogram.shape)
        x = self.cnn1(spectrogram)
        print(x.shape)
        x = x.transpose(2, 3)
        x = self.layernorm1(x)
        x = x.transpose(2, 3)
        print(x.shape)
        x = F.gelu(x)
        x = self.cnn2(x)
        x = self.layernorm2(x)
        x = F.gelu(x)
        x = self.cnn3(x)
        shp = x.shape
        x = x.view(shp[0], shp[1] * shp[2], shp[3]).transpose(1, 2)
        print('here', x.shape)
        return {"logits": self.net(spectrogram.transpose(1, 2))}

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here
