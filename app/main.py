import numpy as np
from data import load_corpus, build_vocab, build_dataset
from model import NeuralLM


def predict(model, context_vec, index_to_word):
    out = model.forward(context_vec)
    return index_to_word[out.argmax()]


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

    # ================= INTERACTIVE MODE =================
    print("Now you can type 2 words to predict next word (type 'exit' to stop)\n")

    while True:

        text = input("Enter 2 words: ").lower()

        if text == "exit":
            break

        words = text.split()

        if len(words) != 2:
            print("Please enter exactly 2 words.")
            continue

        try:
            vec = np.concatenate([
                np.eye(len(vocab))[word_to_index[w]]
                for w in words
            ]).reshape(1, -1)

            print("Prediction:", predict(model, vec, index_to_word))

        except KeyError:
            print("Unknown word in vocabulary.")


if __name__ == "__main__":
    main()