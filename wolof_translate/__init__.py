"""Script containing importation
================================
"""

# let us import all necessary libraries
from transformers import T5Model, T5ForConditionalGeneration, Seq2SeqTrainer, T5TokenizerFast, set_seed, AdamW, get_linear_schedule_with_warmup,\
                          get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup, \
                              Adafactor
from wolof_translate.utils.sent_transformers import TransformerSequences
from wolof_translate.utils.improvements.end_marks import add_end_mark # added
from torch.nn import TransformerEncoderLayer, TransformerDecoderLayer
from torch.utils.data import Dataset, DataLoader, random_split
from wolof_translate.data.dataset_v4 import SentenceDataset # v2 -> v3 -> v4
from wolof_translate.utils.sent_corrections import *
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import _LRScheduler
from torch.nn.utils.rnn import pad_sequence
from plotly.subplots import make_subplots
from nlpaug.augmenter import char as nac
from torch.utils.data import DataLoader
from torch.nn import functional as F
import plotly.graph_objects as go
from tokenizers import Tokenizer
import torch.distributed as dist
import matplotlib.pyplot as plt
import pytorch_lightning as lt
from tqdm import tqdm, trange
from functools import partial
from torch.nn import utils
from copy import deepcopy
from torch import optim
from typing import *
from torch import nn
import pandas as pd
import numpy as np
import itertools
import evaluate
import random
import string
import shutil
import wandb
import torch
import json
import copy
import os

###-----------------------------------------------
# Libraries imported from wolof translate
from wolof_translate.utils.bucket_iterator import SequenceLengthBatchSampler, BucketSampler, collate_fn, collate_fn_trunc
from wolof_translate.trainers.transformer_trainer_custom import ModelRunner as CustomModelRunner
from wolof_translate.models.transformers.optimization import TransformerScheduler
from wolof_translate.utils.recuperate_datasets import recuperate_datasets
from wolof_translate.trainers.transformer_trainer_ml_ import ModelRunner
from wolof_translate.utils.evaluate_custom import TranslationEvaluation
from wolof_translate.models.transformers.main import Transformer
from wolof_translate.utils.split_with_valid import split_data
