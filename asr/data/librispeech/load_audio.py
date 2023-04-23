import logging

import librosa
import numpy as np

logger = logging.getLogger(__name__)


def load_audio(audio_path: str, sample_rate: int, del_silence: bool = False) -> np.ndarray:
    """
    Load audio file (PCM) to sound. if del_silence is True, Eliminate all sounds below 30dB.
    If exception occurs in numpy.memmap(), return None.
    """
    try:
        if audio_path.endswith("pcm"):
            signal = np.memmap(audio_path, dtype="h", mode="r").astype("float32")

            if del_silence:
                non_silence_indices = librosa.effects.split(signal, top_db=30)
                signal = np.concatenate([signal[start:end] for start, end in non_silence_indices])

            return signal / 32767  # normalize audio

        elif audio_path.endswith("wav") or audio_path.endswith("flac"):
            signal, _ = librosa.load(audio_path, sr=sample_rate)
            return signal

    except ValueError:
        logger.warning("ValueError in {0}".format(audio_path))
        return None
    except RuntimeError:
        logger.warning("RuntimeError in {0}".format(audio_path))
        return None
    except IOError:
        logger.warning("IOError in {0}".format(audio_path))
        return None