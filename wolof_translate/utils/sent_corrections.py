from typing import *

def add_guillemet_space(sentences: Union[list, str]):
    """Adding space between a letter and guillemet in a sentence

    Args:
        sentence (Union[list, str]): The sentence that will be changed

    Returns:
        str: The modified sentence
    """
    
    if type(sentences) is str: sentences = [sentences]
    
    for s in range(len(sentences)):
        
        sentence = sentences[s]
            
        if "«" in sentence:
            
            sentence = sentence.split()
            
            for i in range(len(sentence)):
                
                word = sentence[i]
                
                if "«" in word and word != "«":
                    
                    word = word.split("«")
                    
                    word = "« ".join(word)
                
                if "»" in word and word != "»":
                    
                    word = word.split("»")
                    
                    word = " »".join(word)
                            
                sentence[i] = word
            
            sentence = " ".join(sentence)
        
        sentences[s] = sentence
        
    return sentences

def delete_guillemet_space(sentences: Union[list, str]):
    
    if type(sentences) is str: sentences = [sentences]
    
    for s in range(len(sentences)):
        
        sentence = sentences[s]
        
        letters = [sentence[0]]
        
        for i in range(1, len(sentence)):
            
            if sentence[i] == "”":
                
                j = i-1
                
                while letters[j] == " ":
                    
                    letters[j] = ""

                    j -= 1
                
                letters.append(sentence[i])
            
            elif letters[-1] == "“" and sentence[i] == " ":
                    
                    letters.append("")
            
            else:
                
                letters.append(sentence[i])
        
        sentences[s] = "".join(letters)
    
    return sentences

def add_mark_space(sentences: Union[list, str], marks: list = ['?', '!', '–', ':', ';']):
    
    if type(sentences) is str: sentences = [sentences]
    
    for s in range(len(sentences)):
        
        sentence = sentences[s]
        
        letters = [sentence[0]]
        
        for i in range(1, len(sentence)):
            
            if sentence[i] in marks and letters[-1] != " ":
                
                letters[-1] = letters[-1] + " " 
                
                letters.append(sentence[i])
            
            elif letters[-1] in marks and sentence[i] != " ":
                
                letters.append(sentence[i] + " ")
            
            else:
                
                letters.append(sentence[i])
        
        sentences[s] = "".join(letters)
    
    return sentences

def remove_mark_space(sentences: Union[list, str], marks: list = ["'", "-"]):
    
    if type(sentences) is str: sentences = [sentences]
    
    for s in range(len(sentences)):
        
        sentence = sentences[s]
        
        letters = [sentence[0]]
        
        for i in range(1, len(sentence)):
            
            if sentence[i] in marks:
                
                j = i-1
                
                while letters[j] == " ":
                    
                    letters[j] = ""

                    j -= 1
                
                letters.append(sentence[i])
            
            elif letters[-1] in marks and sentence[i] == " ":
                    
                    letters.append("")
            
            else:
                
                letters.append(sentence[i])
        
        sentences[s] = "".join(letters)
    
    return sentences
    
def delete_much_space(sentences: Union[list, str]):
    
    if type(sentences) is str: sentences = [sentences]
    
    for i in range(len(sentences)):
        
        sentences[i] = " ".join(sentences[i].split())
    
    return sentences
    


            
