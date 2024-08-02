from tokenizers import Tokenizer
from typing import *
import numpy as np
import evaluate


class TranslationEvaluation:
    def __init__(
        self,
        tokenizer: Tokenizer,
        decoder: Union[Callable, None] = None,
        metric=evaluate.load("sacrebleu"),
    ):

        self.tokenizer = tokenizer

        self.decoder = decoder

        self.metric = metric

    def postprocess_text(self, preds, labels):

        preds = [pred.strip() for pred in preds]

        for label in labels:

            print(label)

        labels = [[label.strip()] for label in labels]

        return preds, labels

    def compute_metrics(self, eval_preds):

        preds, labels = eval_preds

        if isinstance(preds, tuple):

            preds = preds[0]

        decoded_preds = (
            self.tokenizer.batch_decode(preds, skip_special_tokens=True)
            if not self.decoder
            else self.decoder(preds)
        )

        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)

        decoded_labels = (
            self.tokenizer.batch_decode(labels, skip_special_tokens=True)
            if not self.decoder
            else self.decoder(labels)
        )

        decoded_preds, decoded_labels = self.postprocess_text(
            decoded_preds, decoded_labels
        )

        result = self.metric.compute(
            predictions=decoded_preds, references=decoded_labels
        )

        result = {"bleu": result["score"]}

        prediction_lens = [
            np.count_nonzero(np.array(pred) != self.tokenizer.pad_token_id)
            for pred in preds
        ]

        result["gen_len"] = np.mean(prediction_lens)

        result = {k: round(v, 4) for k, v in result.items()}

        return result
