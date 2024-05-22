import sys
import os
sys.path.insert(1, os.getcwd())

import torch
import yaml
from src.trainer import Trainer
from src.data.datamodule import DataManager
from src.txt_logger import TXTLogger
from src.models.transformer import Seq2SeqTransformer

if __name__ == "__main__":
    if torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"

    data_config = yaml.load(open("configs/data_config.yaml", "r"), Loader=yaml.Loader)
    dm = DataManager(data_config, DEVICE)
    train_dataloader, dev_dataloader = dm.prepare_data()
    
    in_vocab_size = dm.source_tokenizer.tokenizer.get_vocab_size()
    out_vocab_size = dm.target_tokenizer.tokenizer.get_vocab_size()
    idxs = {}
    idxs["PAD_IDX"] = dm.source_tokenizer.tokenizer.token_to_id("[PAD]")
    idxs["BOS_IDX"] = dm.source_tokenizer.tokenizer.token_to_id("[BOS]")
    idxs["EOS_IDX"] = dm.source_tokenizer.tokenizer.token_to_id("[EOS]")

    model_config = yaml.load(open("configs/model_config.yaml", "r"), Loader=yaml.Loader)

    emb_size = model_config["embedding_size"]
    hidden_size = model_config["hidden_size"]
    num_enc = model_config["num_encoders"]
    num_dec = model_config["num_decoders"]
    nhead = model_config["n_heads"]
    
    model = Seq2SeqTransformer(
        num_encoders=num_enc,
        num_decoders=num_dec,
        emb_size=emb_size,
        nhead=nhead,
        inpt_vocab_size=in_vocab_size,
        out_vocab_size=out_vocab_size,
        idxs=idxs,
        dim_feedforward=hidden_size,
        device=DEVICE,
    )

    logger = TXTLogger("training_logs")
    trainer_cls = Trainer(model=model, model_config=model_config, logger=logger)

    if model_config["try_one_batch"]:
        train_dataloader = [list(train_dataloader)[0]]
        dev_dataloader = [list(train_dataloader)[0]]

    trainer_cls.train(train_dataloader, dev_dataloader)
