from torch import Tensor, clamp
from torch import distributions

from hw_asr.augmentations.base import AugmentationBase


class GaussianNoise(AugmentationBase):
    def __init__(self, *args, **kwargs):
        noiser = distributions.Normal(*args, **kwargs)
        self._aug = lambda wav: clamp(wav + noiser.sample(wav.size()), 0, 1)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)
