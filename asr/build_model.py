import torch
from encoder.audio_encoder import TransformerTransducerAudioEncoder
from encoder.label_encoder import TransformerTransducerLabelEncoder
from model import TransformerTransducer
def build_transformer_transducer(
        device: torch.device,
        num_vocabs: int,
        input_size: int = 80,
        model_dim: int = 512,
        ff_dim: int = 2048,
        num_audio_layers: int = 18,
        num_label_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.2,
        max_len: int = 4096,
        pad_id: int = 0,
        sos_id: int = 1,
        eos_id: int = 2,
) -> TransformerTransducer:
    encoder = build_audio_encoder(
        device,
        input_size,
        model_dim,
        ff_dim,
        num_audio_layers,
        num_heads,
        dropout,
        max_len,
    )
    decoder = build_label_encoder(
        device,
        num_vocabs,
        model_dim,
        ff_dim,
        num_label_layers,
        num_heads,
        dropout,
        max_len,
        pad_id,
        sos_id,
        eos_id,
    )
    return TransformerTransducer(encoder, decoder, num_vocabs, model_dim << 1, model_dim).to(device)


def build_audio_encoder(
        device: torch.device,
        input_size: int = 80,
        model_dim: int = 512,
        ff_dim: int = 2048,
        num_audio_layers: int = 18,
        num_heads: int = 8,
        dropout: float = 0.3,
        max_len: int = 4096,
) -> TransformerTransducerAudioEncoder:
    return TransformerTransducerAudioEncoder(
        device,
        input_size,
        model_dim,
        ff_dim,
        num_audio_layers,
        num_heads,
        dropout,
        max_len,
    )


def build_label_encoder(
        device: torch.device,
        num_vocabs: int,
        model_dim: int = 512,
        ff_dim: int = 2048,
        num_label_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.3,
        max_len: int = 4096,
        pad_id: int = 0,
        sos_id: int = 1,
        eos_id: int = 2,
) -> TransformerTransducerLabelEncoder:
    return TransformerTransducerLabelEncoder(
        device,
        num_vocabs,
        model_dim,
        ff_dim,
        num_label_layers,
        num_heads,
        dropout,
        max_len,
        pad_id,
        sos_id,
        eos_id,
    )