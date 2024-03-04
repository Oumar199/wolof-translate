from setuptools import setup

setup(name="wolof_translate", version="0.0.1", author="Oumar Kane", author_email="oumar.kane@univ-thies.sn", 
      description="Contain function and classes to process corpora for making translation between wolof text and other languages.",
      requires=['spacy', 'nltk', 'gensim', 'furo', 'myst-parser', 'Sphinx==6.2.1', 'nbsphinx', 
                'streamlit', 'tokenizers', 'evaluate', 'torch', 'transformers', 'sphinx-autobuild',
                'pandas', 'numpy', 'scikit-learn', 'matplotlib', 'plotly', 'sacrebleu', 'tensorboard',
                'nlpaug', 'wandb', 'pytorch_lightning', 'selenium', 'sentencepiece',])
