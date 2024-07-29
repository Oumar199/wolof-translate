import spacy
from typing import *


def tokenization(
    nlp=spacy.load("fr_core_news_lg"),
    corpus: Union[List[str], Tuple[str]] = [],
    rm_spaces: bool = True,
):
    """Tokenize the text (keep each of the unique token both in the french and the wolof corpora)

    Args:
        nlp (_type_, optional): A spacy model. Defaults to spacy.load("fr_core_news_lg").
        corpus (Union[List[str], Tuple[str]], optional): The list of documents. Defaults to [].
        rm_spaces (bool, optional): Indicate if the too much spaces will be deleted. Defaults to True.

    Returns:
        List[List[str]]: The list of list of tokens
    """

    # Create a inner function to tokenize a given document
    def transformation(doc):

        tokens = []

        for token in doc:

            if not (rm_spaces and token.is_space):

                tokens.append(token.text)

        return tokens

    # Let's create a pipeline with the nlp object
    docs = nlp.pipe(corpus)

    # Initialize the list of tokenized documents and the list of pos_tags
    tokens = []

    for doc in docs:

        tokens_ = transformation(doc)

        tokens.append(tokens_)

    return tokens
