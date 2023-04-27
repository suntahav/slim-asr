from torchaudio import datasets
import os

if __name__ == "__main__":
    librispeech_cls = datasets.LIBRISPEECH
    data_dir = "./data"
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
    train_data = datasets.LIBRISPEECH(root=data_dir, url='train-clean-100', download=True)
    valid_data = datasets.LIBRISPEECH(data_dir, url='dev-clean', download=True)
    test_data = datasets.LIBRISPEECH(data_dir, url='test-clean', download=True)
