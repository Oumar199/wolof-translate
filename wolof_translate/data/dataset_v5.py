
from wolof_translate.utils.sent_transformers import TransformerSequences
from wolof_translate.data.dataset_v4 import T5SentenceDataset
from transformers import PreTrainedTokenizerFast
from torch.utils.data import Dataset
from typing import *
import pandas as pd
import torch
import re

class SentenceDataset(T5SentenceDataset):

    def __init__(
        self,
        data_path: str, 
        tokenizer: PreTrainedTokenizerFast,
        corpus_1: str = "french",
        corpus_2: str = "wolof",
        file_sep: str = ",",
        cp1_transformer: Union[TransformerSequences, None] = None,
        cp2_transformer: Union[TransformerSequences, None] = None,
        **kwargs):
        
        super().__init__(data_path, 
                        tokenizer,
                        corpus_1,
                        corpus_2,
                        0,
                        False,
                        file_sep,
                        cp1_transformer,
                        cp2_transformer
                        **kwargs)
        
    def __getitem__(self, index):
        """Recuperate ids and attention masks of sentences at index

        Args:
            index (int): The index of the sentences to recuperate

        Returns:
            tuple: The `sentence to translate' ids`, `the attention mask of the sentence to translate`
            `the labels' ids`
        """
        sentence_1 = self.sentences_1[index]
        
        sentence_2 = self.sentences_2[index]
        
        # apply transformers if necessary
        if not self.cp1_transformer is None:
            
            sentence_1 = self.cp1_transformer(sentence_1)[0] 
        
        if not self.cp2_transformer is None:
            
            sentence_2 = self.cp2_transformer(sentence_2)[0]
        
        # let us encode the sentences (we provide the second sentence as labels to the tokenizer)
        data = self.tokenizer(
            sentence_1
        )
        
        # let us encode the sentences (we provide the second sentence as labels to the tokenizer)
        labels = self.tokenizer(
            sentence_2
        )
        
        return (data.input_ids.squeeze(0), 
                labels.input_ids.squeeze(0))
    
