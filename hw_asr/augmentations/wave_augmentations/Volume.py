from torch import Tensor
from torchaudio.transforms import Vol

from hw_asr.augmentations.base import AugmentationBase


class Volume(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self._aug = Vol(*args, **kwargs)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)
