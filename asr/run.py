import os
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.callbacks import LearningRateMonitor
from callbacks import CheckpointEveryNSteps
from data.librispeech.librispeech import LightningLibriSpeechDataModule
from data.librispeech.librispeech_tokenizer import LibriSpeechSubwordTokenizer
from build_model import build_transformer_transducer
from pytorch_lightning.loggers import TensorBoardLogger
def get_pl_trainer(
    num_devices: int, logger: TensorBoardLogger
) -> pl.Trainer:    

    trainer = pl.Trainer(
        accelerator='gpu',
        gpus=num_devices,
        accumulate_grad_batches=1,
        check_val_every_n_epoch=1,
        gradient_clip_val=5,
        logger=logger,
        max_steps=200000,
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            CheckpointEveryNSteps(100),
        ],
    )


    return trainer


def hydra_main() -> None:
    pl.seed_everything(32)
    num_devices = 4
    logger = TensorBoardLogger("logs/")
    data_module = LightningLibriSpeechDataModule()
    data_module.prepare_data(data_download=False)
    tokenizer = LibriSpeechSubwordTokenizer()
    data_module.setup()
    model = build_transformer_transducer(pad_id=tokenizer.pad_id, eos_id=tokenizer.eos_id,sos_id=tokenizer.sos_id, vocab_size=tokenizer.vocab_size)
    trainer = get_pl_trainer(num_devices, logger)
    trainer.fit(model, data_module)
    trainer.test(model, data_module)


if __name__ == "__main__":
    hydra_main()