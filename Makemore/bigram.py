import torch
from data import *
from special_character_support import *

words = import_words('names.txt', encoding_method='utf-8')
words_split = split_words(words)
words_train = words_split[0]
words_val = words_split[1]
words_test = words_split[2]

vocab = build_vocab(words_train)
train = [vocab.encode(word) for word in words_train]
val = [vocab.encode(word) for word in words_val]
test = [vocab.encode(word) for word in words_test]

