from bigram import run_model
from data import *
from pathlib import Path
from collections import Counter
import secrets

_SEED = secrets.randbits(32)
_TESTING_RANGE = 120
_SMOOTH = 0.001
_MAX_LEN = 50
_MIN_FREQ = 5
_CLEAN = Path(__file__).parent.parent / "Data" / "clean.txt"
# This 'Data' folder is ignored by git, due to privacy issues.

words = import_words(_CLEAN, encoding_method='utf-8')

"""Below turns low-frequency characters into a specific character."""
freq = Counter(''.join(words)) # Dict[char] -> occurences
keep = {char for char, i in freq.items() if i >= _MIN_FREQ or 0xE000 <= ord(char) <= 0xF8FF}
"""

for word in words:
    for char in word:
        if char not in keep:
            char = '\ue0fe'

Debugging note: This is incorrect because strings cannot be altered this way.       
"""
spe = '\ue0fe'
words = [
    ''.join(char if char in keep else spe for char in word)
    for word in words
] # Correct implementation.


vocab = build_vocab(words, splitting_char='\x00')
train = [vocab.encode(word) for word in words]

size = vocab.size


run_model(train, _TESTING_RANGE, vocab, size, _SMOOTH, _MAX_LEN, _SEED)