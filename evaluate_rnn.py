import torch
from torchtext.datasets import IMDB
import numpy as np
import joblib
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report

from dataset_rnn import tokenize, numericalize
from rnn_model import GRUClassifier


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#########################################################
# Load local IMDB dataset
#########################################################

import os

def load_test_set():
    texts = []
    labels = []
    for label, text in IMDB(split="test"):
        texts.append(text)
        labels.append(1 if label == "pos" else 0)
    return texts, torch.tensor(labels)


#########################################################
# Load vocab + embeddings + model
#########################################################

print("Loading vocab...")
vocab = joblib.load("vocab.pkl")
word2idx = vocab
vocab_size = len(vocab)

print("Loading embeddings...")
embedding_matrix = np.load("trained_embeddings.npy")
embed_dim = embedding_matrix.shape[1]

print("Building model...")
model = GRUClassifier(
    vocab_size=vocab_size,
    embed_dim=embed_dim,
    hidden_size=128,
    num_layers=2,
    pretrained_embeddings=embedding_matrix,
    freeze_embeddings=False,
    dropout=0.3
).to(DEVICE)

print("Loading weights...")
state = torch.load("best_rnn_model.pt", map_location=DEVICE)
model.load_state_dict(state)
model.eval()


#########################################################
# Encode test data using SAME pipeline as training
#########################################################

print("Tokenizing...")

texts, labels = load_test_set()

encoded = []
for t in texts:
    tokens = tokenize(t)[:300]
    ids = numericalize(tokens, word2idx)
    encoded.append(ids)

# pad manually
max_len = max(len(x) for x in encoded)
padded = torch.zeros(len(encoded), max_len, dtype=torch.long)
for i, seq in enumerate(encoded):
    padded[i, :len(seq)] = seq

dataset = TensorDataset(padded, labels)
loader = DataLoader(dataset, batch_size=32, shuffle=False)


#########################################################
# Predict
#########################################################

all_preds = []

with torch.no_grad():
    for batch_x, _ in loader:
        batch_x = batch_x.to(DEVICE)
        out = model(batch_x)
        preds = torch.argmax(out, dim=1).cpu().numpy()
        all_preds.extend(preds)

preds = np.array(all_preds)
labels = labels.numpy()


#########################################################
# Metrics
#########################################################

print("\n=== TEST SET RESULTS ===\n")
print(classification_report(
    labels,
    preds,
    labels=[0, 1],
    target_names=["neg", "pos"],
    digits=4
))
