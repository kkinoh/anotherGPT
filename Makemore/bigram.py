import torch
from data import *
from special_character_support import *
from pathlib import Path

_NAMES_FILE = Path(__file__).with_name('names.txt')
words = import_words(_NAMES_FILE, encoding_method='utf-8')
words_split = split_words(words)
words_train = words_split[0]
words_val = words_split[1]
words_test = words_split[2]

vocab = build_vocab(words_train)
size = vocab.size
train = [vocab.encode(word) for word in words_train]
val = [vocab.encode(word) for word in words_val]
test = [vocab.encode(word) for word in words_test]

TESTING_RANGE = 10

# Now train is a list containing all encoded words to train.
# '.'s are added.

"""
Below calculates frequencies of character occurances.
"""

"""

freq = {}
for encoded_word in train:
    for code1, code2 in zip(encoded_word, encoded_word[1:]):
        bigram = (code1, code2)
        freq[bigram] = 1 if bigram not in freq else freq[bigram] + 1

        
"""

N = torch.zeros((size, size), dtype=torch.int32)

for encoded_word in train:
    for code1, code2 in zip(encoded_word, encoded_word[1:]):
        N[code1][code2] += 1

g = torch.Generator().manual_seed(70416)

P = (N + 1).float()
P = P / P.sum(1, keepdim=True)

for _ in range(TESTING_RANGE):
    out = []
    ix = 0
    while True:
        p = P[ix]
        # ix is an int representing the respective character.
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(vocab.itos[ix])
        if ix == 0:
            break
    print(''.join(out))