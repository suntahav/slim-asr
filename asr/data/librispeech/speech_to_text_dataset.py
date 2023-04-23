import logging
import os
import random

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


from .load_audio import load_audio
from .specaugment import SpecAugment

logger = logging.getLogger(__name__)


class SpeechToTextDataset(Dataset):
    r"""
    Dataset for audio & transcript matching
    Note:
        Do not use this class directly, use one of the sub classes.
    Args:
        dataset_path (str): path of librispeech dataset
        audio_paths (list): list of audio path
        transcripts (list): list of transript
        sos_id (int): identification of <startofsentence>
        eos_id (int): identification of <endofsentence>
        del_silence (bool): flag indication whether to apply delete silence or not
        apply_spec_augment (bool): flag indication whether to apply spec augment or not
        apply_noise_augment (bool): flag indication whether to apply noise augment or not
        apply_time_stretch_augment (bool): flag indication whether to apply time stretch augment or not
        apply_joining_augment (bool): flag indication whether to apply audio joining augment or not
    """
    NONE_AUGMENT = 0
    SPEC_AUGMENT = 1
    NOISE_AUGMENT = 2
    TIME_STRETCH = 3
    AUDIO_JOINING = 4

    def __init__(
        self,
        dataset_path: str,
        audio_paths: list,
        transcripts: list,
        sos_id: int = 1,
        eos_id: int = 2,
        del_silence: bool = False,
        apply_spec_augment: bool = False,
        apply_noise_augment: bool = False,
        apply_time_stretch_augment: bool = False,
        apply_joining_augment: bool = False,
    ) -> None:
        super(SpeechToTextDataset, self).__init__()
        self.dataset_path = dataset_path
        self.audio_paths = list(audio_paths)
        self.transcripts = list(transcripts)
        self.augments = [self.NONE_AUGMENT] * len(self.audio_paths)
        self.dataset_size = len(self.audio_paths)
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.sample_rate = 16000
        self.num_mels = 128
        self.del_silence = del_silence
        self.apply_spec_augment = apply_spec_augment
        self.apply_noise_augment = apply_noise_augment
        self.apply_time_stretch_augment = apply_time_stretch_augment
        self.apply_joining_augment = apply_joining_augment
        self._load_audio = load_audio

        if self.apply_spec_augment:
            self._spec_augment = SpecAugment(
                freq_mask_para=27,
                freq_mask_num=2,
                time_mask_num=4,
            )
            for idx in range(self.dataset_size):
                self.audio_paths.append(self.audio_paths[idx])
                self.transcripts.append(self.transcripts[idx])
                self.augments.append(self.SPEC_AUGMENT)
        self.total_size = len(self.audio_paths)

        tmp = list(zip(self.audio_paths, self.transcripts, self.augments))
        random.shuffle(tmp)

        for i, x in enumerate(tmp):
            self.audio_paths[i] = x[0]
            self.transcripts[i] = x[1]
            self.augments[i] = x[2]

    def _parse_audio(self, audio_path: str, augment: int = None, joining_idx: int = 0) -> Tensor:
        """
        Parses audio.
        Args:
            audio_path (str): path of audio file
            augment (int): augmentation identification
        Returns:
            feature (np.ndarray): feature extract by sub-class
        """
        signal = self._load_audio(audio_path, sample_rate=self.sample_rate, del_silence=self.del_silence)

        if signal is None:
            logger.warning(f"{audio_path} is not Valid!!")
            return torch.zeros(1000, self.num_mels)

        if augment == self.AUDIO_JOINING:
            joining_signal = self._load_audio(self.audio_paths[joining_idx], sample_rate=self.sample_rate)
            signal = self._joining_augment((signal, joining_signal))

        if augment == self.TIME_STRETCH:
            signal = self._time_stretch_augment(signal)

        if augment == self.NOISE_AUGMENT:
            signal = self._noise_injector(signal)

        feature = self.transforms(signal)

        feature -= feature.mean()
        feature /= np.std(feature)

        feature = torch.FloatTensor(feature).transpose(0, 1)

        if augment == self.SPEC_AUGMENT:
            feature = self._spec_augment(feature)

        return feature

    def _parse_transcript(self, transcript: str) -> list:
        """
        Parses transcript
        Args:
            transcript (str): transcript of audio file
        Returns
            transcript (list): transcript that added <sos> and <eos> tokens
        """
        tokens = transcript.split(" ")
        transcript = list()

        transcript.append(int(self.sos_id))
        for token in tokens:
            transcript.append(int(token))
        transcript.append(int(self.eos_id))

        return transcript

    def __getitem__(self, idx):
        """Provides paif of audio & transcript"""
        audio_path = os.path.join(self.dataset_path, self.audio_paths[idx])

        if self.augments[idx] == self.AUDIO_JOINING:
            joining_idx = random.randint(0, self.total_size)
            feature = self._parse_audio(audio_path, self.augments[idx], joining_idx)
            transcript = self._parse_transcript(f"{self.transcripts[idx]} {self.transcripts[joining_idx]}")

        else:
            feature = self._parse_audio(audio_path, self.augments[idx])
            transcript = self._parse_transcript(self.transcripts[idx])

        return feature, transcript

    def __len__(self):
        return len(self.audio_paths)

    def count(self):
        return len(self.audio_paths)