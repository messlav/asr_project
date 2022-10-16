from torch import nn
from torch.nn import Sequential

from hw_asr.base import BaseModel


class FirstModel(BaseModel):
    def __init__(self, n_feats, n_class, fc_hidden=512, **batch):
        super().__init__(n_feats, n_class, **batch)
        self.lstm = nn.LSTM(input_size=n_feats, hidden_size=fc_hidden, num_layers=2, batch_first=True)
        self.net = Sequential(
            nn.Linear(in_features=fc_hidden, out_features=fc_hidden//2),
            nn.ReLU(),
            nn.Linear(in_features=fc_hidden//2, out_features=fc_hidden),
            nn.ReLU(),
            nn.Linear(in_features=fc_hidden, out_features=n_class)
        )

    def forward(self, spectrogram, **batch):
        spectrogram, _ = self.lstm(spectrogram.transpose(1, 2))
        return {"logits": self.net(spectrogram)}

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here
