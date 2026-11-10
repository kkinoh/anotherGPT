"""

This is an implementation of bigram in MLP style.


"""


import torch
import torch.nn.functional as F
from pathlib import Path
from data import *
from special_character_support import *

if __name__ == "__main__":
    BLOCK_SIZE = 3
    DIMENSIONS = 3

    _NAMES_FILE = Path(__file__).with_name('names.txt')
    words = import_words(_NAMES_FILE, encoding_method='utf-8')
    vocab = build_vocab(words, block_size=BLOCK_SIZE)
    train = [vocab.encode(word) for word in words]

    g = torch.Generator().manual_seed(70416)
    # W = torch.randn((vocab.size, vocab.size), generator=g, requires_grad=True)

def trainMLP(train: list[list[int]], vocab: Vocab, is_chat: bool=False):
    X, Y = [], []
    for code in train:
        for i in range(len(code) - vocab.block_size):
            X.append(code[i: i + vocab.block_size]) # Context.
            Y.append(code[i + vocab.block_size]) # Output.

    X = torch.tensor(X)
    Y = torch.tensor(Y)

    """Input."""
    W1 = torch.randn((DIMENSIONS * vocab.block_size, 100), generator=g)
    B1 = torch.randn(100, generator=g)
    C = torch.randn((vocab.size, DIMENSIONS), generator=g)
    # C transforms a code (of a character) into a vector.

    
    """emb = C[X]"""
    # emb = torch.cat([emb[:, i, :] for i in range(vocab.block_size)], 1)
    # emb = torch.cat(torch.unbind(emb, 1), 1)
    """emb = emb.view(-1, DIMENSIONS * vocab.block_size)"""
    # Emb is concatonated. This allows multiple-character context.

    

    """Output."""
    W2 = torch.randn((100, vocab.size), generator=g)
    B2 = torch.randn(vocab.size, generator=g)
    
    """
    counts = logits.exp() # Softmax.
    prob = counts / counts.sum(1, keepdim=True) # Note broadcasting.
    loss = -prob[torch.arange(32), Y].log().mean()

    """
    parameters = [W1, W2, B1, B2, C]
    for param in parameters:
        param.requires_grad = True

    """Search for optimal learning rate."""
    # lre = torch.linspace(-3, 0, 1000)
    # lrs = 10 ** lre
    # lri = []
    # lossi = []

    """Back propogation."""
    def _train(steps: int, lr: float, is_chat: bool):
        for i in range(steps):
            """Minibatch construct."""
            ix = torch.randint(0, X.shape[0], (32, ))
            emb = C[X[ix]]
            h = torch.tanh(emb.view(-1, DIMENSIONS * vocab.block_size) @ W1 + B1)
            logits = h @ W2 + B2
            loss = F.cross_entropy(logits, Y[ix])

            """Backward."""
            for param in parameters:
                param.grad = None
            loss.backward()

            """Update."""
            # lr = lrs[i]
            for param in parameters:
                param.data -= lr * param.grad
                # lri.append(lr)
                # lossi.append(lossi)
            if i % 100 == 0:
                print(loss.item())

        print(loss.item())

        """The lower part is for 'bigramMLP_for_chat.py'. 
        It's ugly but this style of implementation will be abandoned shortly.
        So that's not a huge deal."""

        if is_chat:
            1 # Not implemented yet.
        return

    _train(1000, 0.1, is_chat)
    _train(1000, 0.01, is_chat)


if __name__ == '__main__':
    trainMLP(train, vocab)



