from typing import *


class TransformerSequences:
    def __init__(self, *args, **kwargs):

        self.transformers = []

        self.transformers.extend(list(args))

        self.transformers.extend(list(kwargs.values()))

    def __call__(self, sentences: Union[List, str]):

        output = sentences

        for transformer in self.transformers:

            if hasattr(transformer, "augment"):

                output = transformer.augment(output)

            else:

                output = transformer(output)

        return output
