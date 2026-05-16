import numpy as np
from data import load_corpus, build_vocab, build_dataset
from model import NeuralLM


def predict(model, context_vec, index_to_word):
    out = model.forward(context_vec)
    return index_to_word[out.argmax()]


# =========================================================
# GPT-STYLE TEXT GENERATION FUNCTION
# =========================================================
def generate_text(model, word_to_index, index_to_word, vocab_size, seed, max_words=30):

    words = seed.lower().split()

    for _ in range(max_words):

        # take last 2 words as context
        context = words[-2:]

        try:
            vec = np.concatenate([
                np.eye(vocab_size)[word_to_index[w]]
                for w in context
            ]).reshape(1, -1)

        except KeyError:
            print("Unknown word in seed:", context)
            break

        out = model.forward(vec)

        # NEXT WORD (you can replace argmax with randomness later)
        #next_word = index_to_word[np.argmax(out)]
        next_word = index_to_word[np.random.choice(len(out.ravel()), p=out.ravel())]

        words.append(next_word)

    return " ".join(words)


# =========================================================
# MAIN PROGRAM
# =========================================================
def main():

    sentences = load_corpus()

    vocab, word_to_index, index_to_word = build_vocab(sentences)

    X, y = build_dataset(sentences, word_to_index, len(vocab))

    input_size = len(vocab) * 2
    hidden_size = 64
    output_size = len(vocab)

    model = NeuralLM(input_size, hidden_size, output_size)

    epochs = 2000

    # ================= TRAINING =================
    for epoch in range(epochs):

        out = model.forward(X)
        model.backward(X, y)

        if epoch % 200 == 0:
            loss = -((y * (out + 1e-9)).sum(axis=1)).mean()
            print("Epoch:", epoch, "Loss:", loss)

    model.save()

    print("\nTraining Complete!\n")

    # =====================================================
    # GPT-STYLE MODE (THIS IS THE PART YOU WERE MISSING)
    # =====================================================
    print("GPT-style generator ready!\n")

    while True:

        seed = input("Enter starting words (or 'exit'): ").lower()

        if seed == "exit":
            break

        print("\nGenerated Text:\n")

        result = generate_text(
            model,
            word_to_index,
            index_to_word,
            len(vocab),
            seed,
            max_words=25
        )

        print(result)
        print("\n------------------------\n")


if __name__ == "__main__":
    main()