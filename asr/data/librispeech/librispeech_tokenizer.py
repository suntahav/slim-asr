import os
import torch.nn as nn

from .subword import SENTENCEPIECE_MODEL_NAME



class LibriSpeechSubwordTokenizer(nn.Module):
    """
    Tokenizer class in Subword-units for LibriSpeech.

    """

    def __init__(self):
        super(LibriSpeechSubwordTokenizer, self).__init__()
        try:
            import sentencepiece as spm
        except ImportError:
            raise ImportError("Please install sentencepiece to use LibriSpeechSubwordTokenizer.")
        vocab_path = "../Librispeech/"
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(os.path.join(vocab_path, f"{SENTENCEPIECE_MODEL_NAME}.model"))
        self.pad_id = self.sp.PieceToId("<pad>")
        self.sos_id = self.sp.PieceToId("<s>")
        self.eos_id = self.sp.PieceToId("</s>")
        #self.blank_id = self.sp.PieceToId("<blank>")
        self.vocab_size = 4096

    def __len__(self):
        return self.vocab_size

    def decode(self, labels):
        if len(labels.shape) == 1:
            return self.sp.DecodeIds([l.item() for l in labels])

        elif len(labels.shape) == 2:
            sentences = list()

            for label in labels:
                sentence = self.sp.DecodeIds([l.item() for l in label])
                sentences.append(sentence)
            return sentences
        else:
            raise ValueError("Unsupported label's shape")

    def encode(self, sentence):
        text = " ".join(self.sp.EncodeAsPieces(sentence))
        label = " ".join([str(self.sp.PieceToId(token)) for token in text])
        return label