from torch import Tensor
from librosa.effects import pitch_shift

from hw_asr.augmentations.base import AugmentationBase


class PitchShifting(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self._aug = pitch_shift(*args, **kwargs)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)
