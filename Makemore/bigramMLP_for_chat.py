from bigramMLP import NEURONS
from bigramMLP import BLOCK_SIZE
from bigramMLP import DIMENSIONS
from bigramMLP import trainMLP
from data import *
from pathlib import Path
from collections import Counter

_MIN_FREQ = 3

_CLEAN = Path(__file__).parent.parent / "Data" / "clean.txt"
words = import_words(_CLEAN, encoding_method='utf-8')

"""Trimming out rare characters. Copied from 'bigram_for_chat.py'."""
freq = Counter(''.join(words)) # Dict[char] -> occurences
keep = {char for char, i in freq.items() if i >= _MIN_FREQ or 0xE000 <= ord(char) <= 0xF8FF}
spe = '\ue0fe'
words = [
    ''.join(char if char in keep else spe for char in word)
    for word in words
] # Correct implementation.

vocab = build_vocab(words, block_size=BLOCK_SIZE, splitting_char='\x00')

trainMLP(vocab, words)