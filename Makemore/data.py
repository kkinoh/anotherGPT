import random
from dataclasses import dataclass

"""
Vocab is a class that stores stoi and itos.
"""

@dataclass # Generates __init__ method and others.
class Vocab:
    stoi : dict[str, int]
    itos : dict[int, str]
    splitting_char : str
    block_size: int

    @property
    def size(self) -> int:
        return len(self.stoi)

    def encode(self, word: str) -> list[int]:
        # 'result' -> '.result.' in int
        return [self.stoi[self.splitting_char] for _ in range(self.block_size)] + [self.stoi[c] for c in word] + [self.stoi[self.splitting_char]]

    def decode(self, codes: list[int], to_str=False) -> list[str] | str:
        if to_str:
            return ''.join([self.itos[char] for char in codes[self.block_size:-1]])
        # Returns a list if to_str is False, else a string.
        return codes[self.block_size:-1]

def build_vocab(words: list[str], splitting_char='.', block_size=1) -> Vocab:
    chars = sorted(set(''.join(words)))
    if splitting_char in chars:
        raise ValueError("Splitting character must not be in words.")
    stoi = {s: i + 1 for i, s in enumerate(chars)}
    stoi[splitting_char] = 0
    itos = {i: s for s, i in stoi.items()}

    return Vocab(stoi, itos, splitting_char, block_size)

def import_words(path, encoding_method) -> list[str]:
    with open(path, encoding=encoding_method) as f:
        return [line.strip() for line in f if line.strip()]

"""
Split_words creates a tuple containing training, valuing and testing.
"""

def split_words(words: list[str], train=0.8, val=0.1, seed=70416) -> tuple:
    new_words = words[:] # Does not alter original list.
    rng = random.Random(seed)
    rng.shuffle(new_words)
    length = len(new_words)
    train_index = int(train * length)
    val_index = int(val * length) + train_index
    return (
        new_words[:train_index],
        new_words[train_index : val_index],
        new_words[val_index:]
    )

    