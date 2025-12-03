# dataset_rnn.py
import torch
from torchtext.datasets import IMDB
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import re


def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return text.split()


def build_vocab(tokenized_texts, min_freq=2):
    counter = Counter()
    for tokens in tokenized_texts:
        counter.update(tokens)

    vocab = {"<pad>": 0, "<unk>": 1}
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab


def numericalize(tokens, vocab):
    return torch.tensor([vocab.get(t, vocab["<unk>"]) for t in tokens], dtype=torch.long)


def load_imdb_splits():
    train_iter = IMDB(split="train")
    test_iter  = IMDB(split="test")

    train_texts = []
    train_labels = []
    test_texts = []
    test_labels = []

    for label, text in train_iter:
        train_texts.append(text)
        train_labels.append(1 if label == "pos" else 0)

    for label, text in test_iter:
        test_texts.append(text)
        test_labels.append(1 if label == "pos" else 0)

    return train_texts, train_labels, test_texts, test_labels


def prepare_datasets():
    train_texts, train_labels, test_texts, test_labels = load_imdb_splits()

    tokenized_train = [tokenize(t) for t in train_texts]

    vocab = build_vocab(tokenized_train, min_freq=2)

    X_train = [numericalize(tokens, vocab) for tokens in tokenized_train]
    X_test  = [numericalize(tokenize(t), vocab) for t in test_texts]

    y_train = torch.tensor(train_labels, dtype=torch.long)
    y_test  = torch.tensor(test_labels, dtype=torch.long)

    return X_train, y_train, X_test, y_test, vocab


def collate_fn(batch):
    sequences, labels = zip(*batch)
    sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    return sequences, torch.tensor(labels)
