from tokenizers import Tokenizer
from typing import *
import numpy as np
import evaluate


class TranslationEvaluation:
    def __init__(
        self,
        tokenizer: Tokenizer,
        decoder: Union[Callable, None] = None,
        next_gen: bool = False,
    ):

        self.tokenizer = tokenizer

        self.decoder = decoder

        self.bleu = evaluate.load("sacrebleu")

        self.rouge = evaluate.load("rouge")

        self.accuracy = evaluate.load("accuracy")

        self.next_gen = next_gen

    def postprocess_text(self, preds, labels):

        for i in range(len(labels)):

            pred = preds[i].strip()

            label = labels[i].strip()

            if self.next_gen:

                new_pred = ""

                new_label = ""

                for j in range(len(labels[i])):

                    if pred[:j] != labels[i][:j]:

                        new_pred = pred[j:]

                        new_label = label[j:]

                        break

                preds[i] = new_pred

                labels[i] = new_label

            else:

                preds[i] = pred

                labels[i] = [label]

        return preds, labels

    def postprocess_codes(self, preds: np.ndarray, labels: np.ndarray):

        label_weights = (labels != 0).astype(float).tolist()

        preds = preds.tolist()

        labels = labels.tolist()

        return preds, labels, label_weights

    def compute_metrics(
        self, eval_preds, rouge: bool = True, bleu: bool = True, accuracy: bool = False
    ):

        preds, labels = eval_preds

        if isinstance(preds, tuple):

            preds = preds[0]

        decoded_preds = (
            self.tokenizer.batch_decode(preds, skip_special_tokens=True)
            if not self.decoder
            else self.decoder(preds)
        )

        decoded_labels = (
            self.tokenizer.batch_decode(labels, skip_special_tokens=True)
            if not self.decoder
            else self.decoder(labels)
        )

        result = {}

        if accuracy:

            pred_codes, label_codes, sample_weight = self.postprocess_codes(
                preds, labels
            )

            accuracy_result = np.mean(
                [
                    self.accuracy.compute(
                        predictions=pred_codes[i],
                        references=label_codes[i],
                        sample_weight=sample_weight[i],
                    )["accuracy"]
                    for i in range(len(pred_codes))
                ]
            )

            result["accuracy"] = accuracy_result

        if bleu or rouge:

            decoded_preds, decoded_labels = self.postprocess_text(
                decoded_preds, decoded_labels
            )

        if bleu:

            bleu_result = self.bleu.compute(
                predictions=decoded_preds, references=decoded_labels
            )

            result["bleu"] = bleu_result["score"]

        if rouge:

            rouge_result = self.rouge.compute(
                predictions=decoded_preds, references=decoded_labels
            )

            result.update(rouge_result)

        prediction_lens = [
            np.count_nonzero(np.array(pred) != self.tokenizer.pad_token_id)
            for pred in preds
        ]

        result["gen_len"] = np.mean(prediction_lens)

        result = {k: round(v, 4) for k, v in result.items()}

        return result
