# eval_rnn.py

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support

from preprocess import load_dataset, tokenize
from dataset_rnn import TextDataset, collate_fn
from glove_loader import load_glove_embeddings
from rnn_model import GRUClassifier
from joblib import load


def evaluate_rnn():

    # Load test set
    texts, labels = load_dataset("test")
    vocab = load("vocab.pkl")  # same vocabulary

    token_lists = [tokenize(t) for t in texts]
    test_dataset = TextDataset(token_lists, labels, vocab)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # Load GloVe
    emb_matrix = load_glove_embeddings("embeddings/glove.6B.100d.txt", vocab)

    # Build model
    model = GRUClassifier(
        vocab_size=len(vocab),
        embed_dim=100,
        pretrained_embeddings=emb_matrix
    )
    model.load_state_dict(torch.load("best_rnn.pt"))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    preds = []
    trues = []

    with torch.no_grad():
        for padded, lengths, y in test_loader:
            padded, lengths = padded.to(device), lengths.to(device)
            out = model(padded, lengths)
            out = (out > 0.5).long().cpu().numpy()

            preds.extend(out)
            trues.extend(y.numpy())

    p, r, f, support = precision_recall_fscore_support(trues, preds, average=None)
    p_micro, r_micro, f_micro, _ = precision_recall_fscore_support(trues, preds, average="micro")
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(trues, preds, average="macro")

    print("\n=== RNN Test Performance ===")
    print("Class 0 (Negative): P=%.3f R=%.3f F1=%.3f" % (p[0], r[0], f[0]))
    print("Class 1 (Positive): P=%.3f R=%.3f F1=%.3f" % (p[1], r[1], f[1]))
    print("--------------------------------")
    print("Micro Avg: P=%.3f R=%.3f F1=%.3f" % (p_micro, r_micro, f_micro))
    print("Macro Avg: P=%.3f R=%.3f F1=%.3f" % (p_macro, r_macro, f_macro))
