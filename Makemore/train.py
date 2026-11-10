from model import *
from sample import *
from pathlib import Path

TRAIN_STEPS = 5000
OUTPUT_STEP = 500

_NAMES_FILE = Path(__file__).with_name('names.txt')
words = import_words(_NAMES_FILE, encoding_method='utf-8')
vocab = build_vocab(words)

train = [vocab.encode(word) for word in words]

"""
Not completed.
"""