import numpy as np
import joblib
from glove_loader import load_glove_embeddings

print("Loading vocab...")
vocab = joblib.load("vocab.pkl")

print("Loading GloVe embeddings (exact same as training)...")
embeddings = load_glove_embeddings(
    "embeddings/glove.6B.100d.txt",
    vocab,
    embedding_dim=100
)

print("Saving trained_embeddings.npy ...")
np.save("trained_embeddings.npy", embeddings)

print("Done! Embeddings saved successfully.")
