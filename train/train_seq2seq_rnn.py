import sys
import os
sys.path.insert(1, os.getcwd())

from src.models.rnn import Seq2SeqRNN
import torch
import yaml
from src.trainer import Trainer
from src.data.datamodule import DataManager
from src.txt_logger import TXTLogger

if __name__ == "__main__":
    if torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = 'cpu'

    data_config = yaml.load(open("configs/data_config.yaml", 'r'),   Loader=yaml.Loader)
    dm = DataManager(data_config, DEVICE)
    train_dataloader, dev_dataloader = dm.prepare_data()

    model_config = yaml.load(open("configs/model_config.yaml", 'r'),   Loader=yaml.Loader)

    model = Seq2SeqRNN(
        encoder_vocab_size=len(dm.source_tokenizer.index2word),
        encoder_embedding_size=model_config['embedding_size'],
        decoder_embedding_size=model_config['embedding_size'],
        encoder_hidden_size=model_config['hidden_size'],
        decoder_hidden_size=model_config['hidden_size'],
        decoder_vocab_size=len(dm.target_tokenizer.index2word),
        lr=model_config['learning_rate'],
        device=DEVICE,
        target_tokenizer=dm.target_tokenizer
    )

    logger = TXTLogger('training_logs')
    trainer_cls = Trainer(model=model, model_config=model_config, logger=logger)

    if model_config['try_one_batch']:
        train_dataloader = [list(train_dataloader)[0]]
        dev_dataloader = [list(train_dataloader)[0]]

    trainer_cls.train(train_dataloader, dev_dataloader)




