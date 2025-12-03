# glove_loader.py
import numpy as np

def load_glove_embeddings(path, vocab, embedding_dim=100):
    embeddings = np.random.normal(scale=0.6, size=(len(vocab), embedding_dim))

    print("Loading GloVe embeddings...")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            word = parts[0]
            if word in vocab:
                vector = np.array(parts[1:], dtype=float)
                embeddings[vocab[word]] = vector

    print(f"GloVe loaded. Embedding matrix shape = {embeddings.shape}")
    return embeddings
