import torch
from torch.nn import nn

import src.metrics.bleu_score as bleu_scorer
from src.models.utils.positional_encoding import PositionalEncoding
from src.models.utils.token_embeddings import TokenEmbedding


class Seq2SeqTransformer(torch.nn.Module):
    def __init__(
        self,
        num_encoders: int,
        num_decoders: int,
        emb_size: int,
        nhead: int,
        inpt_vocab_size: int,
        out_vocab_size: int,
        idxs: dict[str, str],
        dropout: float = 0.1,
        dim_feedforward: int = 512,
        device: str = "cpu",
    ):
        super(Seq2SeqTransformer, self).__init__()
        self.config = idxs
        self.device = device
        self.transformer = nn.Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoders,
            num_decoder_layers=num_decoders,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.generator = nn.Linear(emb_size, out_vocab_size)
        self.src_tok_emb = TokenEmbedding(inpt_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(out_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(self, input_tensor: torch.Tensor):
        batch_size = input_tensor.shape[0]
        src = input_tensor.transpose(1, 0)
        tgt_input = torch.tensor(
            [self.config["BOS_IDX"] * batch_size], dtype=torch.long, device=self.device
        ).view(1, batch_size)
        for _ in range(self.config["max_len"]):
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.create_mask(
                src, tgt_input
            )
            logits = self.forward_pass(
                src,
                tgt_input,
                src_mask,
                tgt_mask,
                src_padding_mask,
                tgt_padding_mask,
                src_padding_mask,
            )
            _, next_items = logits[-1].topk(1)
            next_items = next_items.squeeze().view(1, batch_size)
            tgt_input = torch.cat((tgt_input, next_items), dim=0)
        return tgt_input, None

    def forward_pass(
        self,
        src: torch.Tensor,
        tgt_input: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
        src_padding_mask: torch.Tensor,
        tgt_padding_mask: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
    ):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt_input))
        outs = self.transformer(
            src_emb,
            tgt_emb,
            src_mask,
            tgt_mask,
            None,
            src_padding_mask,
            tgt_padding_mask,
            memory_key_padding_mask,
        )
        return self.generator(outs)

    def training_step(self, batch):
        self.optimizer.zero_grad()
        src, tgt = batch
        src = src.transpose(1, 0)
        tgt = tgt.transpose(1, 0)
        tgt_input = tgt[:-1, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.create_mask(
            src, tgt_input
        )
        logits = self.forward_pass(
            src,
            tgt_input,
            src_mask,
            tgt_mask,
            src_padding_mask,
            tgt_padding_mask,
            src_padding_mask,
        )
        tgt_out = tgt[1:, :]
        loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return loss

    def validation_step(self, batch):
        with torch.no_grad():
            src, tgt = batch
            src = src.transpose(1, 0)
            tgt = tgt.transpose(1, 0)
            tgt_input = tgt[:-1, :]
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
                src, tgt_input
            )
            logits = self.forward_pass(
                src,
                tgt_input,
                src_mask,
                tgt_mask,
                src_padding_mask,
                tgt_padding_mask,
                src_padding_mask,
            )
            tgt_out = tgt[1:, :]
            loss = self.loss_fn(
                logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1)
            )

        return loss

    def eval_bleu(self, predicted_ids_list, target_tensor):
        predicted = torch.stack(predicted_ids_list)
        predicted = predicted.squeeze().detach().cpu().numpy().swapaxes(0, 1)[:, 1:]
        actuals = target_tensor.squeeze().detach().cpu().numpy()[:, 1:]
        bleu_score, actual_sentences, predicted_sentences = bleu_scorer(
            predicted=predicted, actual=actuals, target_tokenizer=self.target_tokenizer
        )
        return bleu_score, actual_sentences, predicted_sentences

    def create_mask(self, src: torch.Tensor, tgt: torch.Tensor):
        src_seq_len = src.shape[0]

        tgt_seq_len = tgt.shape[0]

        mask = (torch.triu(torch.ones((tgt_seq_len, tgt_seq_len), device=self.device)) == 1).transpose(
            0, 1
        )
        tgt_mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        src_mask = torch.zeros((src_seq_len, src_seq_len), device=self.device).type(
            torch.bool
        )

        src_padding_mask = (src == self.config["PAD_IDX"]).transpose(0, 1)
        tgt_padding_mask = (tgt == self.config["PAD_IDX"]).transpose(0, 1)
        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
