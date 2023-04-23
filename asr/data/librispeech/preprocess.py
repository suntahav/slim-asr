import os

LIBRI_SPEECH_DATASETS = [
    "train-clean-100",
    "dev-clean",
    "test-clean"
]


def collect_transcripts(dataset_path):
    """Collect librispeech transcripts"""
    dataset_path = os.path.join(dataset_path, "LibriSpeech")
    transcripts_collection = list()

    for dataset in LIBRI_SPEECH_DATASETS:
        dataset_transcripts = list()

        for subfolder1 in os.listdir(os.path.join(dataset_path, dataset)):
            for subfolder2 in os.listdir(os.path.join(dataset_path, dataset, subfolder1)):
                for file in os.listdir(os.path.join(dataset_path, dataset, subfolder1, subfolder2)):
                    if file.endswith("txt"):
                        with open(os.path.join(dataset_path, dataset, subfolder1, subfolder2, file)) as f:
                            for line in f.readlines():
                                tokens = line.split()
                                audio_path = os.path.join(dataset, subfolder1, subfolder2, tokens[0])
                                audio_path = f"{audio_path}.flac"
                                transcript = " ".join(tokens[1:])
                                dataset_transcripts.append("%s|%s" % (audio_path, transcript))

                    else:
                        continue

        transcripts_collection.append(dataset_transcripts)

    return transcripts_collection