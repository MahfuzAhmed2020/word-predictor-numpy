import numpy as np
from utils import one_hot

def load_corpus(path="data/corpus.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().lower().splitlines()


def build_vocab(sentences, vocab_size=1000):
    words = set()

    for s in sentences:
        for w in s.split():
            words.add(w)

    words = list(words)

    # pad if needed
    while len(words) < vocab_size:
        words.append(f"word_{len(words)}")
       # words = words[:vocab_size]

    words = words[:vocab_size]

    word_to_index = {w: i for i, w in enumerate(words)}
    index_to_word = {i: w for w, i in word_to_index.items()}

    return words, word_to_index, index_to_word


def build_dataset(sentences, word_to_index, vocab_size, context_size=2):
    X, y = [], []

    for sentence in sentences:
        tokens = sentence.split()

        for i in range(len(tokens) - context_size):
            context = tokens[i:i+context_size]
            target = tokens[i+context_size]

            try:
                x_vec = np.concatenate([
                    one_hot(word_to_index[w], vocab_size)
                    for w in context
                ])

                y_vec = one_hot(word_to_index[target], vocab_size)

                X.append(x_vec)
                y.append(y_vec)

            except KeyError:
                continue

    return np.array(X), np.array(y)