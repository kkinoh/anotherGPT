"""

This achieves a similar bigram model to "bigram.py".
But "model.py" is achieved in the neural network way.


"""

import torch
import torch.nn.functional as F
from data import *
from pathlib import Path

_NAMES_FILE = Path(__file__).with_name('names.txt')
words = import_words(_NAMES_FILE, encoding_method='utf-8')
vocab = build_vocab(words)

train = [vocab.encode(word) for word in words]

g = torch.Generator().manual_seed(70416)
W = torch.randn((vocab.size, vocab.size), generator=g, requires_grad=True)

xs = []
ys = []
for code in train:
    for code1, code2 in zip(code, code[1:]):
        xs.append(code1)
        ys.append(code2)

xs = torch.tensor(xs, dtype=torch.long)
ys = torch.tensor(ys, dtype=torch.long)
for step in range(1000):
    xenc = F.one_hot(xs, num_classes=vocab.size).float()
    logits = xenc @ W # Predicts next char, which is y.
    counts = logits.exp() # Softmax.
    probs = counts / counts.sum(1, keepdim=True)
    loss = -probs[torch.arange(xs.nelement()), ys].log().mean()

    W.grad = None
    loss.backward()

    W.data -= 50 * W.grad

    if step % 100 == 0:
        print(loss.item())