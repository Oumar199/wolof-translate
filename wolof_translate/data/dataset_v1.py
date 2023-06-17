from wolof_translate.utils.sent_transformers import TransformerSequences
from transformers import PreTrainedTokenizerFast
from torch.utils.data import Dataset
from tokenizers import Tokenizer
from typing import *
import pandas as pd
import torch
import re

class SentenceDataset(Dataset):
 
    def __init__(self,
                 file_path: str, 
                 corpus_1: str = "french_corpus",
                 corpus_2: str = "wolof_corpus",
                 tokenizer_path: str = "wolof-translate/wolof_translate/tokenizers/tokenizer_v1.json",
                 max_len: int = 379,
                 truncation: bool = False,
                 file_sep: str = ",", 
                 cls_token: str = "<|endoftext|>",
                 sep_token: str = "<|translateto|>",
                 pad_token: str = "<|pad|>",
                 cp1_transformer: Union[TransformerSequences, None] = None,
                 cp2_transformer: Union[TransformerSequences, None] = None,
                 **kwargs):
        
        # let us recuperate the data frame
        self.__sentences = pd.read_csv(file_path, sep=file_sep, **kwargs)
        
        # let us recuperate the tokenizer
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=tokenizer_path,
            bos_token=cls_token,
            eos_token=cls_token,
            pad_token=pad_token
            )
        
        # recuperate the first corpus' sentences
        self.__sentences_1 = self.__sentences[corpus_1].to_list()
        
        # recuperate the second corpus' sentences
        self.__sentences_2 = self.__sentences[corpus_2].to_list()
        
        # recuperate the special tokens
        self.cls_token = cls_token
        
        self.sep_token = sep_token
        
        self.pad_token = pad_token
        
        # recuperate the length
        self.__length = len(self.__sentences_1)
        
        # recuperate the max id
        self.max_id = len(self.tokenizer) - 1
        
        # let us recuperate the max len
        self.max_len = max_len
        
        # let us recuperate the truncate argument
        self.truncation = truncation
        
        # let us initialize the transformer
        self.cp1_transformer = cp1_transformer
        
        self.cp2_transformer = cp2_transformer
        
    def __getitem__(self, index):
        
        sentence_1 = self.__sentences_1[index]
        
        sentence_2 = self.__sentences_2[index]
        
        # apply transformers if necessary
        if not self.cp1_transformer is None:
            
            sentence_1 = self.cp1_transformer(sentence_1) 
        
        if not self.cp2_transformer is None:
            
            sentence_2 = self.cp2_transformer(sentence_2)
        
        # let us create the sentence with special tokens
        sentence = f"{self.cls_token}{sentence_1}{self.sep_token}{sentence_2}{self.cls_token}"
        
        # let us encode the sentence
        encoding = self.tokenizer(sentence, truncation=self.truncation, max_length=self.max_len, padding='max_length', return_tensors="pt")
        
        return encoding.input_ids.squeeze(0), encoding.attention_mask.squeeze(0)
        
    def __len__(self):
        
        return self.__length
    
    def decode(self, ids: torch.Tensor, for_prediction: bool = False):
        
        if ids.ndim < 2:
            
            ids = ids.unsqueeze(0)
        
        ids = ids.tolist()
        
        for id in ids:
            
            sentence = self.tokenizer.decode(id)

            if not for_prediction:
            
                sentence = sentence.split(f"{self.sep_token}")
            
            else:
                
                try:
                    
                    while self.sep_token in sentence:
                        
                        sentence = re.findall(f"{self.sep_token}(.*)", sentence)[-1]
                    
                except:
                    
                    sentence = "None"
            
            if for_prediction:
                
                yield sentence.replace(f'{self.cls_token}', '').replace(f'{self.pad_token}', '')
            
            else:
                
                sents = []
                
                for sent in sentence:
                    
                    sents.append(sent.replace(f'{self.cls_token}', '').replace(f'{self.pad_token}', ''))
                    
                yield sents
