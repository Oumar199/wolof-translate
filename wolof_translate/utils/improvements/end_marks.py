
import pandas as pd
from typing import *

def add_end_mark(sentences: Union[list, str], end_mark: str = ".", end_mark_to_remove: Union[str, None] = None, 
                 replace: bool = False, poss_end_marks: list = ['!', '?', '.', '...']):
    
    if isinstance(sentences, str): sentences = [sentences]
    
    if replace and end_mark_to_remove is None:
        
        raise ValueError("You must provide a end mark to remove if you want to make replacement !")
    
    new_sentences = []
    
    # if replace is chosen we will replace the end to remove by the end mark if not only add end mark to each sentence and remove the end mark to remove
    for sentence in sentences:
        
        if replace:
            
            if sentence[-1] == end_mark_to_remove:
                
                sentence = sentence[:-1].strip() + end_mark
            
        else:
            
            if not end_mark_to_remove is None and sentence[-1] == end_mark_to_remove:
                
                sentence = sentence[:-1]
            
            if not sentence[-1] in poss_end_marks: 
                
                sentence = sentence.strip() + end_mark
        
        new_sentences.append(sentence)
    
    return new_sentences
    
