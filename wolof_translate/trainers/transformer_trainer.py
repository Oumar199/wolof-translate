"""Nouvelle classe d'entraînement. On la fournit un modèle et des hyperparamètres en entrée.
Nous allons créer des classes supplémentaire qui vont supporter la classe d'entraînement
"""

from wolof_translate.utils.evaluation import TranslationEvaluation
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from tqdm import tqdm, trange
from torch.nn import utils
from torch import optim
from typing import *
from torch import nn
import pandas as pd
import numpy as np
import string
import torch
import json
import copy
import os

# choose letters for random words
letters = string.ascii_lowercase


class PredictionError(Exception):
    def __init__(self, error: Union[str, None] = None):

        if not error is None:

            print(error)

        else:

            print(
                "You cannot with this type of data! Provide a list of tensors, a list of numpy arrays, a numpy array or a torch tensor."
            )


class LossError(Exception):
    def __init__(self, error: Union[str, None] = None):

        if not error is None:

            print(error)

        else:

            print("A list of losses is provided for multiple outputs.")


class ModelRunner:
    def __init__(
        self,
        model: nn.Module,
        optimizer=optim.AdamW,
        seed: Union[int, None] = None,
        evaluation: Union[TranslationEvaluation, None] = None,
        version: int = 1,
    ):

        # Initialisation de la graine du générateur
        self.seed = seed

        # Initialisation de la version
        self.version = version

        # Recuperate the evaluation metric
        self.evaluation = evaluation

        # Initialisation du générateur
        if self.seed:
            torch.manual_seed(self.seed)

        # Le modèle à utiliser pour les différents entraînements
        self.orig_model = model

        # L'optimiseur à utiliser pour les différentes mises à jour du modèle
        self.orig_optimizer = optimizer

        # Récupération du type de 'device'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.compilation = False

    # ------------------------------ Training staffs (Partie entraînement et compilation) --------------------------

    def batch_train(
        self,
        input_: torch.Tensor,
        input_mask: torch.Tensor,
        labels: torch.Tensor,
        labels_mask: torch.Tensor,
        pad_token_id: int = 3,
    ):
        if (
            self.hugging_face
        ):  # Nous allons utilise un modèle text to text de hugging face (but only for fine-tuning)

            # concatenate the input and the label

            # effectuons un passage vers l'avant
            outputs = self.model(
                input_ids=input_, attention_mask=input_mask, labels=labels
            )

            # recuperate the predictions and the loss
            preds, loss = outputs.logits, outputs.loss

        else:

            # effectuons un passage vers l'avant
            outputs = self.model(
                input_, input_mask, labels, labels_mask, pad_token_id=pad_token_id
            )

            # recuperate the predictions and the loss
            preds, loss = outputs["preds"], outputs["loss"]

        # effectuons un passage vers l'arrière
        loss.backward()

        # forcons les valeurs des gradients à se tenir dans un certain interval si nécessaire
        if not self.clipping_value is None:

            utils.clip_grad_value_(
                self.model.parameters(), clip_value=self.clipping_value
            )

        # mettons à jour les paramètres
        self.optimizer.step()

        # Réduction du taux d'apprentissage à chaque itération si nécessaire
        if not self.lr_scheduling is None:

            self.lr_scheduling.step()

        # reinitialisation des gradients
        self.optimizer.zero_grad()

        return preds, loss

    def batch_eval(
        self,
        input_: torch.Tensor,
        input_mask: torch.Tensor,
        labels: torch.Tensor,
        labels_mask: torch.Tensor,
        pad_token_id: int = 3,
    ):

        if (
            self.hugging_face
        ):  # Nous allons utilise un modèle text to text de hugging face (but only for fine-tuning)

            # effectuons un passage vers l'avant
            outputs = self.model(
                input_ids=input_, attention_mask=input_mask, labels=labels
            )
            # recuperate the predictions and the loss
            preds, loss = outputs.logits, outputs.loss

        else:

            # effectuons un passage vers l'avant
            outputs = self.model(
                input_, input_mask, labels, labels_mask, pad_token_id=pad_token_id
            )

            # recuperate the predictions and the loss
            preds, loss = outputs["preds"], outputs["loss"]

        return preds, loss

    # On a décidé d'ajouter quelques paramètres qui ont été utiles au niveau des enciennes classes d'entraînement
    def compile(
        self,
        train_dataset: Dataset,
        test_dataset: Union[Dataset, None] = None,
        tokenizer: Union[Tokenizer, None] = None,
        train_loader_kwargs: dict = {"batch_size": 16},
        test_loader_kwargs: dict = {"batch_size": 16},
        optimizer_kwargs: dict = {"lr": 1e-4, "weight_decay": 0.4},
        model_kwargs: dict = {
            "class_criterion": nn.CrossEntropyLoss(label_smoothing=0.1)
        },
        lr_scheduler_kwargs: dict = {"d_model": 512, "lr_warmup_step": 100},
        lr_scheduler=None,
        gradient_clipping_value: Union[float, torch.Tensor, None] = None,
        predict_with_generate: bool = False,
        logging_dir: Union[str, None] = None,
        hugging_face: bool = False,
        decoder_only: bool = False,
    ):

        if self.seed:
            torch.manual_seed(self.seed)

        # On devra utiliser la méthode 'spread' car on ne connait pas les paramètres du modèle
        if isinstance(
            self.orig_model, nn.Module
        ):  # si c'est une instance d'un modèle alors pas de paramètres requis

            self.model = copy.deepcopy(self.orig_model).to(self.device)

        else:  # sinon on fournit les paramètres

            self.model = copy.deepcopy(self.orig_model(**model_kwargs)).to(self.device)

        # Initialisation des paramètres de l'optimiseur
        self.optimizer = self.orig_optimizer(
            self.model.parameters(), **optimizer_kwargs
        )

        # On ajoute un réducteur de taux d'apprentissage si nécessaire
        self.lr_scheduling = None

        if not lr_scheduler is None and self.lr_scheduling is None:

            self.lr_scheduling = lr_scheduler(self.optimizer, **lr_scheduler_kwargs)

        # initialize the datasets and the loaders
        self.train_set = train_dataset
        self.test_set = test_dataset

        self.train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            **train_loader_kwargs,
        )

        if test_dataset:
            self.test_loader = DataLoader(
                test_dataset,
                shuffle=False,
                **test_loader_kwargs,
            )

        else:
            self.test_loader = None

        # Let us initialize the clipping value to make gradient clipping
        self.clipping_value = gradient_clipping_value

        # Other parameters for step tracking and metrics
        self.compilation = True

        self.current_epoch = None

        self.best_score = None

        self.best_epoch = self.current_epoch

        # Recuperate some boolean attributes
        self.predict_with_generate = predict_with_generate

        # Recuperate tokenizer
        self.tokenizer = tokenizer

        # Recuperate the logging directory
        self.logging_dir = logging_dir

        # Initialize the metrics
        self.metrics = {}

        # Initialize the attribute which indicate if the model is from huggingface
        self.hugging_face = hugging_face

        # Initialize the hugging face model type
        self.decoder_only = decoder_only

    def train(
        self,
        epochs: int = 100,
        auto_save: bool = False,
        log_step: Union[int, None] = None,
        saving_directory: str = "data/checkpoints/last_checkpoints",
        file_name: str = "checkpoints",
        save_best: bool = True,
        metric_for_best_model: str = "test_loss",
        metric_objective: str = "minimize",
    ):
        """Entraînement du modèle

        Args:
            epochs (int, optional): Le nombre d'itérations. Defaults to 100.
            auto_save (bool, optional): Auto-sauvegarde du modèle. Defaults to False.
            log_step (int, optional): Le nombre d'itération avant d'afficher les performances. Defaults to 1.
            saving_directory (str, optional): Le dossier de sauvegarde du modèle. Defaults to "inception_package/storage".
            file_name (str, optional): Le nom du fichier de sauvegarde. Defaults to "checkpoints".
            save_best (bool): Une varible booléenne indiquant si l'on souhaite sauvegarder le meilleur modèle. Defaults to True.
            metric_for_best_model (str): Le nom de la métrique qui permet de choisir le meilleur modèle. Defaults to 'eval_loss'.
            metric_objective (str): Indique si la métrique doit être maximisée 'maximize' ou minimisée 'minimize'. Defaults to 'minimize'.

        Raises:
            Exception: L'entraînement implique d'avoir déja initialisé les paramètres
        """

        # the file name cannot be "best_checkpoints"
        assert file_name != "best_checkpoints"

        ##################### Error Handling ##################################################
        if not self.compilation:
            raise Exception(
                "You must initialize datasets and\
                            parameters with `compile` method. Make sure you don't forget any of them before \n \
                                training the model"
            )

        ##################### Initializations #################################################

        if metric_objective in ["maximize", "minimize"]:

            best_score = (
                float("-inf") if metric_objective == "maximize" else float("inf")
            )

        else:

            raise ValueError(
                "The metric objective can only between 'maximize' or minimize!"
            )

        if not self.best_score is None:

            best_score = self.best_score

        start_epoch = self.current_epoch if not self.current_epoch is None else 0

        ##################### Training ########################################################

        modes = ["train", "test"]

        if self.test_loader is None:
            modes = ["train"]

        for epoch in tqdm(range(start_epoch, start_epoch + epochs)):

            # Print the actual learning rate
            print(f"For epoch {epoch + 1}: ")

            if self.lr_scheduling:
                print(f"{{Learning rate: {self.lr_scheduling.get_lr()}}}")

            self.metrics = {}

            for mode in modes:

                with torch.set_grad_enabled(mode == "train"):

                    # Initialize the loss of the current mode
                    self.metrics[f"{mode}_loss"] = 0

                    # Let us initialize the predictions
                    predictions_ = []

                    # Let us initialize the labels
                    labels_ = []

                    if mode == "train":

                        self.model.train()

                        loader = list(iter(self.train_loader))

                    else:

                        self.model.eval()

                        loader = list(iter(self.test_loader))

                    with trange(
                        len(loader), unit="batches", position=0, leave=True
                    ) as pbar:
                        # for i in tqdm(range(len(loader))):
                        for i in pbar:

                            pbar.set_description(
                                f"{mode[0].upper() + mode[1:]} batch number {i + 1}"
                            )

                            data = loader[i]

                            input_ = data[0].long().to(self.device)

                            # let us initialize a fake input
                            # input__ = None

                            input_mask = data[1].to(self.device)

                            # let us initialize a fake input mask
                            # input_mask_ = None

                            labels = data[2].long().to(self.device)

                            if self.hugging_face:

                                # concatenate the input with the labels and the two attention masks if we only use a decoder
                                # if self.decoder_only:

                                #     # let us modify the fake input to the first sentence
                                #     input__ = copy.deepcopy(input_)

                                #     input_ = torch.concat((input_, labels), dim=1)

                                #     # the new labels are equal to the inputs
                                #     labels = copy.deepcopy(input_)

                                #     # let us modify the fake input mask to mask of the first sentence
                                #     input_mask_ = copy.deepcopy(input_mask)

                                #     input_mask = torch.concat((input_mask, data[3].to(self.device)), dim=1)

                                labels[labels == self.tokenizer.pad_token_id] == -100

                            labels_mask = data[3].to(self.device)

                            # Récupération de identifiant token du padding (par défaut = 3)
                            pad_token_id = (
                                3
                                if self.tokenizer is None
                                else self.tokenizer.pad_token_id
                            )

                            preds, loss = (
                                self.batch_train(
                                    input_,
                                    input_mask,
                                    labels,
                                    labels_mask,
                                    pad_token_id,
                                )
                                if mode == "train"
                                else self.batch_eval(
                                    input_,
                                    input_mask,
                                    labels,
                                    labels_mask,
                                    pad_token_id,
                                )
                            )

                            self.metrics[f"{mode}_loss"] += loss.item()

                            # let us add the predictions and labels in the list of predictions and labels after their determinations
                            if mode == "test":

                                if self.predict_with_generate:

                                    if self.hugging_face:

                                        # preds = self.model.generate(input_ if not self.decoder_only else input__,
                                        #  attention_mask = input_mask if not self.decoder_only else input_mask_,
                                        #   max_new_tokens = self.train_set.max_len, pad_token_id = self.test_set.tokenizer.eos_token_id)

                                        preds = self.model.generate(
                                            input_,
                                            attention_mask=input_mask,
                                            max_length=self.train_set.max_len,
                                        )

                                    else:

                                        preds = self.model.generate(
                                            input_,
                                            input_mask,
                                            pad_token_id=pad_token_id,
                                        )

                                else:

                                    if self.hugging_face:

                                        preds = torch.argmax(preds, dim=-1)

                                predictions_.extend(preds.detach().cpu().tolist())

                                labels_.extend(labels.detach().cpu().tolist())

            if not self.evaluation is None and mode == "test":

                self.metrics.update(
                    self.evaluation.compute_metrics(
                        (np.array(predictions_), np.array(labels_))
                    )
                )

            self.metrics[f"train_loss"] = self.metrics[f"train_loss"] / len(
                self.train_loader
            )

            if not self.test_loader is None:

                self.metrics[f"test_loss"] = self.metrics[f"test_loss"] / len(
                    self.test_loader
                )

            # for metric in self.metrics:

            #    if metric != 'train_loss':

            #     self.metrics[metric] = self.metrics[metric] / len(self.test_loader)

            # Affichage des métriques
            if not log_step is None and (epoch + 1) % log_step == 0:

                print(f"\nMetrics: {self.metrics}")

                if not self.logging_dir is None:

                    with SummaryWriter(
                        os.path.join(self.logging_dir, f"version_{self.version}")
                    ) as writer:

                        for metric in self.metrics:

                            writer.add_scalar(
                                metric, self.metrics[metric], global_step=epoch
                            )

                            writer.add_scalar("global_step", epoch)

            print("\n=============================\n")

            ##################### Model saving #########################################################

            # Save the model in the end of the current epoch. Sauvegarde du modèle à la fin d'une itération
            if auto_save:

                self.current_epoch = epoch + 1

                if save_best:

                    # verify if the current score is best and recuperate it if yes
                    if metric_objective == "maximize":

                        last_score = best_score < self.metrics[metric_for_best_model]

                    elif metric_objective == "minimize":

                        last_score = best_score > self.metrics[metric_for_best_model]

                    else:

                        raise ValueError(
                            "The metric objective can only be in ['maximize', 'minimize'] !"
                        )

                    # recuperate the best score
                    if last_score:

                        best_score = self.metrics[metric_for_best_model]

                        self.best_epoch = self.current_epoch + 1

                        self.best_score = best_score

                        self.save(saving_directory, "best_checkpoints")

                self.save(saving_directory, file_name)

    # Pour la méthode nous allons nous inspirer sur la méthode save de l'agent ddpg (RL) que l'on avait créée
    def save(
        self,
        directory: str = "data/checkpoints/last_checkpoints",
        file_name: str = "checkpoints",
    ):

        if not os.path.exists(directory):
            os.makedirs(directory)

        file_path = os.path.join(directory, f"{file_name}.pth")

        checkpoints = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "current_epoch": self.current_epoch,
            "metrics": self.metrics,
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "lr_scheduler_state_dict": self.lr_scheduling.state_dict()
            if not self.lr_scheduling is None
            else None,
        }

        torch.save(checkpoints, file_path)

        # update metrics and the best score dict
        self.metrics["current_epoch"] = (
            self.current_epoch + 1
            if not self.current_epoch is None
            else self.current_epoch
        )

        best_score_dict = {"best_score": self.best_score, "best_epoch": self.best_epoch}

        # save the metrics as json file
        metrics = json.dumps(
            {"metrics": self.metrics, "best_performance": best_score_dict}, indent=4
        )

        with open(os.path.join(directory, f"{file_name}.json"), "w") as f:

            f.write(metrics)

    # Ainsi que pour la méthode load
    def load(
        self,
        directory: str = "data/checkpoints/last_checkpoints",
        file_name: str = "checkpoints",
        load_best: bool = False,
    ):

        if load_best:
            file_name = "best_checkpoints"

        file_path = os.path.join(directory, f"{file_name}.pth")

        if os.path.exists(file_path):

            checkpoints = (
                torch.load(file_path)
                if torch.device == torch.device("cuda")
                else torch.load(file_path, map_location="cpu")
            )

            self.model.load_state_dict(checkpoints["model_state_dict"])

            self.optimizer.load_state_dict(checkpoints["optimizer_state_dict"])

            self.current_epoch = checkpoints["current_epoch"]

            self.best_score = checkpoints["best_score"]

            self.best_epoch = checkpoints["best_epoch"]

            if not self.lr_scheduling is None:

                self.lr_scheduling.load_state_dict(
                    checkpoints["lr_scheduler_state_dict"]
                )

        else:

            raise OSError(
                f"Le fichier {file_path} est introuvable. Vérifiez si le chemin fourni est correct!"
            )

    def evaluate(self, test_dataset, batch_size: int = 16, loader_kwargs: dict = {}):

        self.model.eval()

        test_loader = list(
            iter(
                DataLoader(
                    test_dataset,
                    batch_size,
                    shuffle=False,
                    **loader_kwargs,
                )
            )
        )

        # Let us initialize the predictions
        predictions_ = []

        # Let us initialize the labels
        labels_ = []

        metrics = {"test_loss": 0.0}

        results = {"original_sentences": [], "translations": [], "predictions": []}

        with torch.no_grad():

            with trange(
                len(test_loader), unit="batches", position=0, leave=True
            ) as pbar:
                # for i in tqdm(range(len(test_loader))):
                for i in pbar:

                    pbar.set_description(f"Evaluation batch number {i + 1}")

                    data = test_loader[i]

                    input_ = data[0].long().to(self.device)

                    input_mask = data[1].to(self.device)

                    labels = data[2].long().to(self.device)

                    if self.hugging_face:

                        # concatenate the input with the labels and the two attention masks if we only use a decoder
                        # if self.decoder_only:

                        #     labels = torch.concat((input_, labels))

                        labels[labels == test_dataset.tokenizer.pad_token_id] == -100

                    labels_mask = data[3].to(self.device)

                    preds, loss = self.batch_eval(
                        input_,
                        input_mask,
                        labels,
                        labels_mask,
                        test_dataset.tokenizer.pad_token_id,
                    )

                    metrics[f"test_loss"] += loss.item()

                    if self.hugging_face:

                        # preds = self.model.generate(input_, attention_mask = input_mask, max_new_tokens = self.train_set.max_len * 2, pad_token_id = test_dataset.tokenizer.eos_token_id)

                        preds = self.model.generate(
                            input_,
                            attention_mask=input_mask,
                            max_length=self.train_set.max_len,
                        )

                    else:

                        preds = self.model.generate(
                            input_,
                            input_mask,
                            pad_token_id=test_dataset.tokenizer.pad_token_id,
                        )

                    labels_.extend(labels.detach().cpu().tolist())

                    predictions_.extend(preds.detach().cpu().tolist())

                    # let us recuperate the original sentences
                    results["original_sentences"].extend(
                        test_dataset.tokenizer.batch_decode(
                            input_, skip_special_tokens=True
                        )
                    )

                    results["translations"].extend(
                        test_dataset.tokenizer.batch_decode(
                            labels, skip_special_tokens=True
                        )
                    )

                    results["predictions"].extend(test_dataset.decode(preds))

            if not self.evaluation is None:

                metrics.update(
                    self.evaluation.compute_metrics(
                        (np.array(predictions_), np.array(labels_))
                    )
                )

            metrics["test_loss"] = metrics["test_loss"] / len(test_loader)

            return metrics, pd.DataFrame(results)
