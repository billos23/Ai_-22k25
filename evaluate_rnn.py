import torch
import torch.nn as nn
import joblib
from preprocess import tokenize
from rnn_model import GRUClassifier
from sklearn.metrics import classification_report
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#########################################################
# Load test set
#########################################################

def load_test_set():
    from torchtext.datasets import IMDB
    texts = []
    labels = []
    for label, text in IMDB(split="test"):
        texts.append(text)
        labels.append(1 if label == "pos" else 0)
    return texts, torch.tensor(labels)


#########################################################
# Convert texts to padded tensors
#########################################################

def encode_texts(texts, word2idx, max_len=300):
    encoded = []
    for t in texts:
        tokens = tokenize(t)
        idxs = [word2idx.get(tok, 1) for tok in tokens]   # 1 = <unk>
        idxs = idxs[:max_len]
        if len(idxs) < max_len:
            idxs += [0] * (max_len - len(idxs))          # 0 = <pad>
        encoded.append(torch.tensor(idxs))
    return torch.stack(encoded)


#########################################################
# Load model + vocab
#########################################################

print("Loading vocab and model...")

vocab = joblib.load("vocab.pkl")
word2idx = vocab["word2idx"]

embedding_matrix = np.load("embedding_matrix.npy")
embedding_tensor = torch.tensor(embedding_matrix)

model = GRUClassifier(
    vocab_size=embedding_matrix.shape[0],
    embed_dim=embedding_matrix.shape[1],
    hidden_size=128,
    num_layers=2,
    pretrained_embeddings=embedding_matrix
).to(DEVICE)

model.load_state_dict(torch.load("best_rnn_model.pt", map_location=DEVICE))
model.eval()


#########################################################
# Encode test data
#########################################################

texts, labels = load_test_set()
inputs = encode_texts(texts, word2idx).to(DEVICE)


#########################################################
# Predict
#########################################################

with torch.no_grad():
    outputs = model(inputs)
    preds = torch.argmax(outputs, dim=1).cpu().numpy()

labels = labels.numpy()


#########################################################
# Print report
#########################################################

print("\n=== TEST SET RESULTS ===\n")
print(classification_report(labels, preds, target_names=["neg", "pos"], digits=4))
