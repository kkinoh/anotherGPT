"""

This is a torchified implementation of MLP.


"""

import torch
import torch.nn.functional as F
from data import *
from pathlib import Path

if __name__ == "__main__":
    g = torch.Generator().manual_seed(70416)

class Linear:
    def __init__(self, fan_in: int, fan_out: int, bias: bool=True):
        self.weight = torch.randn((fan_in, fan_out), generator=g)
        self.bias = torch.zeros(fan_out) if bias else None

        return

    def __call__(self, x: torch.Tensor):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])

class BatchNorm1d:
    def __init__(self, dim: int, eps: float=1e-5, momentum: float=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        # Parameters.
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        # Buffers.
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

        return

    def __call__(self, x: torch.Tensor):
        if self.training:
            xmean = x.mean(0, keepdim=True)
            xvar = x.var(0, keepdim=True)
        else:
            xmean = self.running_mean
            xvar = self.running_var

        self.out = self.gamma * (x - xmean) / torch.sqrt(xvar + self.eps) + self.beta

        # Update buffers.
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar

        return self.out

    def parameters(self):
        return [self.gamma, self.beta]

class Tanh:
    def __call__(self, x: torch.Tensor):
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []

if __name__ == "__main__":
    EMBEDDING = 10 # Dimension of embedding vector: turning character (encoded) into vector
    NEURONS = 100
    BLOCK_SIZE = 5

    _NAMES_FILE = Path(__file__).with_name('names.txt')
    words = import_words(_NAMES_FILE, encoding_method='utf-8')
    split = split_words(words)
    vocab = build_vocab(words, block_size=BLOCK_SIZE)
    def _build(_words):
        X, Y = [], []
        for word in _words:
            code = vocab.encode(word)
            for i in range(len(code) - vocab.block_size):
                X.append(code[i: i + vocab.block_size]) # Context.
                Y.append(code[i + vocab.block_size]) # Output.
        return torch.tensor(X), torch.tensor(Y)
    X, Y = _build(split[0])
    Xval, Yval = _build(split[1])
    Xtest, Ytest = _build(split[2])

    C = torch.randn((vocab.size), EMBEDDING)
    layers = [
        Linear(EMBEDDING * BLOCK_SIZE, NEURONS, bias=False), BatchNorm1d(NEURONS), Tanh(),
        Linear(               NEURONS, NEURONS, bias=False), BatchNorm1d(NEURONS), Tanh(),
        Linear(               NEURONS, NEURONS, bias=False), BatchNorm1d(NEURONS), Tanh(),
        Linear(               NEURONS, NEURONS, bias=False), BatchNorm1d(NEURONS), Tanh(),
        Linear(            NEURONS, vocab.size, bias=False), BatchNorm1d(vocab.size)
    ]

    with torch.no_grad():
        layers[-1].gamma *= 0.1 # Make last layer less confident.
        for layer in layers[:-1]:
            if isinstance(layer, Linear):
                layer.weight *= (5 / 3 / (layer.weight.shape[0] ** 0.5))

    parameters = [p for layer in layers for p in layer.parameters()] + [C]
    for p in parameters:
        p.requires_grad = True

    """Start training."""
    STEPS = 20000
    BATCH_SIZE = 32

    for i in range(STEPS + 1):
        ix = torch.randint(0, X.shape[0], (BATCH_SIZE, ), generator=g)
        Xb, Yb = X[ix], Y[ix]

        emb = C[Xb]
        x = emb.view(emb.shape[0], -1)
        for layer in layers:
            x = layer(x)
        loss = F.cross_entropy(x, Yb)

        # Backward.
        for p in parameters:
            p.grad = None
        loss.backward()

        # Update.
        lr = 0.1 if i < STEPS * 3 / 4 else 0.01 # Learning rate.
        for p in parameters:
            p.data -= lr * p.grad

        # Track.
        if i % 5000 == 0:
            print(f'{i: 7d} / {STEPS: 7d} : {loss.item(): .4f}')
        
        