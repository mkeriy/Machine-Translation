import torch

import metrics.bleu_score as bleu_scorer
from transformers import T5ForConditionalGeneration



class Seq2SeqT5(torch.nn.Module):
    def __init__(
        self,
        model_name,
        tokenizer,
        optimizer,
        scheduler_step_size,
        lr,
        loss_fn,
        device,
    ):
        super(Seq2SeqT5, self).__init__()
        self.device = device
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.resize_token_embeddings(len(tokenizer))
        self.optimizer = optimizer(self.model.parameters(), lr=lr, relative_step=False)
        self.loss_fn = loss_fn
        self.model.to(self.device)
        self.tokenizer = tokenizer
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=scheduler_step_size, gamma=0.99
        )

    def forward(self, input_tensor: torch.Tensor):
        generated_ids = self.model.generate(
            input_ids=input_tensor[0],
            attention_mask=input_tensor[1],
            max_length=15,
            num_beams=2,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True,
        )

        preds = [
            self.tokenizer.decode(
                g, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            for g in generated_ids
        ]
        target = [
            self.tokenizer.decode(
                t, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            for t in input_tensor[2]
        ]

        return preds, target

    def training_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()

        loss = self.model(
            input_ids=batch[0], attention_mask=batch[1], labels=batch[2]
        ).loss

        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        return loss

    def validation_step(self, batch):
        self.model.eval()
        with torch.no_grad():
            loss = self.model(
                input_ids=batch[0], attention_mask=batch[1], labels=batch[2]
            ).loss

        return loss

    def eval_bleu(self, predicted_ids_list, target_tensor):
        predicted = torch.stack(predicted_ids_list)
        predicted = predicted.squeeze().detach().cpu().numpy().swapaxes(0, 1)[:, 1:]
        actuals = target_tensor.squeeze().detach().cpu().numpy()[:, 1:]
        bleu_score, actual_sentences, predicted_sentences = bleu_scorer(
            predicted=predicted, actual=actuals, target_tokenizer=self.target_tokenizer
        )
        return bleu_score, actual_sentences, predicted_sentences
