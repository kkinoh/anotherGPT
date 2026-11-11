"""

This is an implementation of bigram in MLP style.


"""
BLOCK_SIZE = 5
DIMENSIONS = 6
NEURONS = 300

import torch
import torch.nn.functional as F
from pathlib import Path
from data import *
from special_character_support import *

if __name__ == "__main__":
    _NAMES_FILE = Path(__file__).with_name('names.txt')
    words = import_words(_NAMES_FILE, encoding_method='utf-8')
    vocab = build_vocab(words, block_size=BLOCK_SIZE)
    # W = torch.randn((vocab.size, vocab.size), generator=g, requires_grad=True)

def trainMLP(vocab: Vocab, words: list[list[int]]):
    g = torch.Generator().manual_seed(70416)
    #for code in train:
    #    for i in range(len(code) - vocab.block_size):
    #        X.append(code[i: i + vocab.block_size]) # Context.
    #        Y.append(code[i + vocab.block_size]) # Output.

    def _build(_words):
        X, Y = [], []
        for word in _words:
            code = vocab.encode(word)
            for i in range(len(code) - vocab.block_size):
                X.append(code[i: i + vocab.block_size]) # Context.
                Y.append(code[i + vocab.block_size]) # Output.
        return torch.tensor(X), torch.tensor(Y)

    """Change here for train / no train."""
    split = split_words(words)
    X, Y = _build(split[0])
    Xval, Yval = _build(split[1])
    Xtest, Ytest = _build(split[2])

    """Input."""
    W1 = torch.randn((DIMENSIONS * vocab.block_size, NEURONS), generator=g) * 0.1
    B1 = torch.randn(NEURONS, generator=g)
    C = torch.randn((vocab.size, DIMENSIONS), generator=g) * 0.1
    # C transforms a code (of a character) into a vector.

    
    """emb = C[X]"""
    # emb = torch.cat([emb[:, i, :] for i in range(vocab.block_size)], 1)
    # emb = torch.cat(torch.unbind(emb, 1), 1)
    """emb = emb.view(-1, DIMENSIONS * vocab.block_size)"""
    # Emb is concatonated. This allows multiple-character context.

    

    """Output."""
    W2 = torch.randn((NEURONS, vocab.size), generator=g) * 0.1
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
    def _train(steps: int, lr: float):
        for i in range(steps):
            """Minibatch construct."""
            ix = torch.randint(0, X.shape[0], (32, ), generator=g)
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
            if i % 500 == 0:
                print(loss.item())
            if i % 1000 == 0:
                with torch.no_grad():
                    embval = C[Xval]
                    hval = torch.tanh(embval.view(-1, DIMENSIONS * vocab.block_size) @ W1 + B1)
                    logitsval =  hval @ W2 + B2
                    lossval = F.cross_entropy(logitsval, Yval)
                    print(f'Val loss: {lossval}')

        print(loss.item())
        return

    _train(40000, 0.1)
    _train(40000, 0.01)

    """Calculate loss based on val."""
    

    def _out(testing_range: int, max_len: int=80):
        for _ in range(testing_range):
            with torch.no_grad():
                out = []
                context = [vocab.stoi[vocab.splitting_char]] * vocab.block_size
                for _ in range(max_len):
                    emb = C[torch.tensor([context])]
                    # Size is (1, block size, dimensions)
                    h = torch.tanh(emb.view(1, -1) @ W1 + B1)
                    # Flattens a number of block_size words (each being a vector with dimensions) into one vector.
                    # Ex. (3, 2), (4, 5), (6, 7) -> (3, 2, 4, 5, 6, 7), with
                    # block_size = 3
                    # dimensions = 2
                    logits = h @ W2 + B2 # Size is (1, vocab.size)
                    # 1 is batch size.
                    probs = F.softmax(logits, dim=1)
                    ix = torch.multinomial(probs, num_samples=1, replacement=True, generator=g).item()
                    if ix == 0:
                        break
                    context = context[1:] + [ix]
                    out.append(ix)
                print(decode_specials(''.join(vocab.itos[i] for i in out)).replace('\ue0fe', '<sp>'))

        return

    _out(200)



if __name__ == '__main__':
    trainMLP(vocab, words)



