import os
import logging
import shutil
import tarfile
from typing import Optional, Tuple
import wget

import pytorch_lightning as pl
from ..audio_dataloader import AudioDataLoader
from ..sampler import RandomSampler
#RandomSampler
from .subword import generate_manifest_files
from .speech_to_text_dataset import SpeechToTextDataset
class LightningLibriSpeechDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning Data Module for LibriSpeech Dataset. LibriSpeech is a corpus of approximately 1000 hours of read
    English speech with sampling rate of 16 kHz, prepared by Vassil Panayotov with the assistance of Daniel Povey.
    The data is derived from read audiobooks from the LibriVox project, and has been carefully segmented and aligned.

    """

    LIBRISPEECH_TRAIN_NUM = 28539
    LIBRISPEECH_VALID_NUM = 2703
    LIBRISPEECH_TEST_NUM = 2620
    LIBRISPEECH_PARTS = [
        "train-clean-100",
        "dev-clean",
        "test-clean"
    ]

    def __init__(self) -> None:
        super(LightningLibriSpeechDataModule, self).__init__()
        self.dataset_path = "../Librispeech/"
        self.manifest_file_path = "../Librispeech/manifest.txt"

        self.dataset = dict()
        self.logger = logging.getLogger(__name__)

    def _parse_manifest_file(self, manifest_file_path: str) -> Tuple[list, list]:
        """Parsing manifest file"""
        audio_paths = list()
        transcripts = list()

        with open(manifest_file_path) as f:
            for idx, line in enumerate(f.readlines()):
                audio_path, _, transcript = line.split("\t")
                transcript = transcript.replace("\n", "")

                audio_paths.append(audio_path)
                transcripts.append(transcript)

        return audio_paths, transcripts

    def _download_dataset(self) -> None:
        """
        Download librispeech dataset.
            - train-960(train-clean-100, train-clean-360, train-other-500)
            - dev-clean
            - dev-other
            - test-clean
            - test-other
        """
        base_url = "http://www.openslr.org/resources/12"
        train_dir = "LibriSpeech/train-960"
        dataset_path = self.dataset_path
        if not os.path.exists(dataset_path):
            os.mkdir(dataset_path)

        for part in self.LIBRISPEECH_PARTS:
            self.logger.info(f"Librispeech-{part} download..")
            url = f"{base_url}/{part}.tar.gz"
            wget.download(url)
            shutil.move(f"{part}.tar.gz", os.path.join(dataset_path, f"{part}.tar.gz"))

            self.logger.info(f"Un-tarring archive {dataset_path}/{part}.tar.gz")
            tar = tarfile.open(f"{dataset_path}/{part}.tar.gz", mode="r:gz")
            tar.extractall(dataset_path)
            tar.close()
            os.remove(f"{dataset_path}/{part}.tar.gz")

        self.logger.info("Merge all train packs into one")

        if not os.path.exists(dataset_path):
            os.mkdir(dataset_path)
        if not os.path.exists(os.path.join(dataset_path, train_dir)):
            os.mkdir(os.path.join(dataset_path, train_dir))

        for part in self.LIBRISPEECH_PARTS[-3:]:  # train
            path = os.path.join(dataset_path, "LibriSpeech", part)
            subfolders = os.listdir(path)
            for subfolder in subfolders:
                shutil.move(
                    os.path.join(path, subfolder),
                    os.path.join(dataset_path, train_dir, subfolder),
                )

    def prepare_data(self,data_download=False) -> None:
        """
        Prepare librispeech data
        Returns:
            tokenizer (Tokenizer): tokenizer is in charge of preparing the inputs for a model.
        """
        dataset_path = self.dataset_path


        if data_download:
            self._download_dataset()

        if not os.path.exists(self.manifest_file_path):
            self.logger.info("Manifest file is not exists !!\n" "Generate manifest files..")

            vocab_path = "../Librispeech/"
            generate_manifest_files(
                dataset_path=dataset_path,
                manifest_file_path=self.manifest_file_path,
                vocab_path=vocab_path,
                vocab_size=4096,
            )

    def setup(self, stage: Optional[str] = None) -> None:
        r"""Split dataset into train, valid, and test."""
        valid_end_idx = self.LIBRISPEECH_TRAIN_NUM + self.LIBRISPEECH_VALID_NUM
        audio_paths, transcripts = self._parse_manifest_file(self.manifest_file_path)

        audio_paths = {
            "train": audio_paths[: self.LIBRISPEECH_TRAIN_NUM],
            "valid": audio_paths[self.LIBRISPEECH_TRAIN_NUM : valid_end_idx],
            "test": audio_paths[valid_end_idx:],
        }
        transcripts = {
            "train": transcripts[: self.LIBRISPEECH_TRAIN_NUM],
            "valid": transcripts[self.LIBRISPEECH_TRAIN_NUM : valid_end_idx],
            "test": transcripts[valid_end_idx:],
        }
        dataset_path = self.dataset_path
        for stage in audio_paths.keys():
            self.dataset[stage] = SpeechToTextDataset(
                dataset_path=os.path.join(dataset_path, "LibriSpeech"),
                audio_paths=audio_paths[stage],
                transcripts=transcripts[stage],
                apply_spec_augment=True if stage == "train" else False,
                )

    def train_dataloader(self) -> AudioDataLoader:
        sampler = RandomSampler
        train_sampler = sampler(data_source=self.dataset["train"], batch_size=32)
        return AudioDataLoader(
            dataset=self.dataset["train"],
            num_workers=12,
            batch_sampler=train_sampler,
        )

    def val_dataloader(self) -> AudioDataLoader:
        sampler = RandomSampler
        valid_sampler = sampler(self.dataset["valid"], batch_size=32)
        return AudioDataLoader(
            dataset=self.dataset["valid"],
            num_workers=12,
            batch_sampler=valid_sampler,
        )

    def test_dataloader(self) -> AudioDataLoader:
        sampler = RandomSampler
        test_sampler = sampler(self.dataset["test"], batch_size=32)
        return AudioDataLoader(
            dataset=self.dataset["test"],
            num_workers=12,
            batch_sampler=test_sampler,
        )