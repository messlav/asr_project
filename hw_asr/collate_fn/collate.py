import logging
from typing import List
import torch

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    # TODO: actually librspeach has sort items by audio and spec length
    audios, spectrograms, durations, texts, text_encodeds, audio_paths = [], [], [], [], [], []
    audios_length, spectrograms_length, text_encodeds_length = [], [], []
    for elem in dataset_items:
        audios += [elem['audio']]
        spectrograms += [elem['spectrogram']]
        durations += [elem['duration']]
        texts += [elem['text']]
        text_encodeds += [elem['text_encoded']]
        audio_paths += [elem['audio_path']]

        audios_length += [elem['audio'].shape[-1]]
        spectrograms_length += [elem['spectrogram'].shape[-1]]
        text_encodeds_length += [elem['text_encoded'].shape[-1]]

    # pad audio
    batch_audios = torch.zeros(len(dataset_items), max(audios_length))
    for i, audio in enumerate(audios):
        batch_audios[i, :audios_length[i]] = audio

    # pad spectogram
    batch_spectrograms = torch.zeros(len(dataset_items), spectrograms[0].shape[1], max(spectrograms_length))
    for i, spec in enumerate(spectrograms):
        batch_spectrograms[i, :, :spectrograms_length[i]] = spec

    # pad encoded text
    batch_text_encodeds = torch.zeros(len(dataset_items), max(text_encodeds_length))
    for i, text_encoded in enumerate(text_encodeds):
        batch_text_encodeds[i, :text_encodeds_length[i]] = text_encoded

    result_batch = {
        'audio': batch_audios,
        'spectrogram': batch_spectrograms,
        'spectrogram_length': torch.tensor(spectrograms_length),
        'text_encoded': batch_text_encodeds,
        'text_encoded_length': torch.tensor(text_encodeds_length),
        'text': texts,
        'durations': torch.tensor(durations),
        'audio_path': audio_paths
    }
    return result_batch