import sys
import os
sys.path.insert(1, os.getcwd())

from src.models.t5 import Seq2SeqT5
import torch
import yaml
from src.trainer import Trainer
from src.data.datamodule import DataManager
from txt_logger import TXTLogger

from transformers.optimization import Adafactor

if __name__ == "__main__":
    if torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"

    data_config = yaml.load(open("configs/data_config.yaml", "r"), Loader=yaml.Loader)
    dm = DataManager(data_config, DEVICE)
    train_dataloader, dev_dataloader = dm.prepare_data()

    model_config = yaml.load(open("configs/t5_config.yaml", "r"), Loader=yaml.Loader)

    model = Seq2SeqT5(
        model_name=model_config["model_name"],
        tokenizer=dm.tokenizer,
        optimizer=Adafactor,
        scheduler_step_size=model_config["scheduler_step_size"],
        lr=model_config["learning_rate"],
        loss_fn=torch.nn.CrossEntropyLoss(),
        device=DEVICE,
    )
    
    logger = TXTLogger("training_logs")
    trainer_cls = Trainer(model=model, model_config=model_config, logger=logger)

    if model_config["try_one_batch"]:
        train_dataloader = [list(train_dataloader)[0]]
        dev_dataloader = [list(train_dataloader)[0]]

    trainer_cls.train(train_dataloader, dev_dataloader)
