import os
import numpy as np
import re
from collections import Counter

def load_dataset(split):
 
    base_dir = os.path.dirname(os.path.abspath(__file__))

    imdb_dir = os.path.join(base_dir, "imdb", split)

    pos_dir = os.path.join(imdb_dir, "pos")
    neg_dir = os.path.join(imdb_dir, "neg")

    texts = []
    labels = []
    for fname in os.listdir(neg_dir):
        path = os.path.join(neg_dir, fname)
        with open(path, encoding="utf-8") as f:
            texts.append(f.read())
            labels.append(0)

    for fname in os.listdir(pos_dir):
        path = os.path.join(pos_dir, fname)
        with open(path, encoding="utf-8") as f:
            texts.append(f.read())
            labels.append(1)

    texts = np.array(texts)
    labels = np.array(labels, dtype=np.int32)

    idx = np.random.permutation(len(texts))
    return texts[idx], labels[idx]

def tokenize(text):
    text = text.lower()
    tokens = re.findall(r"[a-zA-Z]+", text)
    return tokens

def compute_document_frequency(token_lists):
    df = Counter()
    for toks in token_lists:
        df.update(set(toks))
    return df

def remove_top_n_and_bottom_k(df, n, k):
    sorted_words = sorted(df.items(), key=lambda x: x[1], reverse=True)
    remove_common = {w for w, _ in sorted_words[:n]}
    remove_rare = {w for w, _ in sorted_words[-k:]}
    return remove_common | remove_rare

def compute_information_gain(df, token_lists, labels):
    IG = {}
    N = len(token_lists)
    eps = 1e-9

    labels = np.array(labels)
    pos_mask = labels == 1
    neg_mask = labels == 0

    # H(Y)
    p_pos = np.mean(labels)
    p_neg = 1 - p_pos
    H_y = -(p_pos*np.log2(p_pos + eps) + p_neg*np.log2(p_neg + eps))

    token_sets = [set(toks) for toks in token_lists]

   
    word_docs = {word: [] for word in df.keys()}
    for i, toks in enumerate(token_sets):
        for word in toks:
            if word in word_docs:
                word_docs[word].append(i)

    def entropy(a, b):
        total = a + b
        if total == 0:
            return 0.0
        pa = a/total if a > 0 else eps
        pb = b/total if b > 0 else eps
        return -(pa*np.log2(pa) + pb*np.log2(pb))

    
    for word, docs in word_docs.items():
        docs = np.array(docs, dtype=int)

        contain_pos = np.sum(labels[docs] == 1)
        contain_neg = np.sum(labels[docs] == 0)

        not_pos = np.sum(pos_mask) - contain_pos
        not_neg = np.sum(neg_mask) - contain_neg

        p_word = len(docs) / N
        p_not = 1 - p_word

        H_word = entropy(contain_pos, contain_neg)
        H_not = entropy(not_pos, not_neg)

        IG[word] = H_y - (p_word * H_word + p_not * H_not)

    return IG


############################################################
# 6. BUILD VOCABULARY
############################################################
def build_vocabulary(texts, labels, n=200, k=5, m=3000):
    token_lists = [tokenize(t) for t in texts]

    clean_tokens = []
    clean_labels = []
    for toks, lab in zip(token_lists, labels):
        if len(toks) > 0:
            clean_tokens.append(toks)
            clean_labels.append(lab)

    token_lists = clean_tokens
    labels = np.array(clean_labels, dtype=np.int32)

    df = compute_document_frequency(token_lists)
    df = {w: c for w, c in df.items() if c >= 5}

    removed = remove_top_n_and_bottom_k(df, n, k)
    filtered_df = {w: c for w, c in df.items() if w not in removed}

    IG = compute_information_gain(filtered_df, token_lists, labels)

    vocab_words = sorted(IG, key=IG.get, reverse=True)[:m]
    vocab = {w: i for i, w in enumerate(vocab_words)}

    return vocab, token_lists


############################################################
# 7. VECTORIZE
############################################################
def vectorize(token_lists, vocab):
    X = np.zeros((len(token_lists), len(vocab)), dtype=np.uint8)
    for i, toks in enumerate(token_lists):
        for tok in set(toks):
            if tok in vocab:
                X[i, vocab[tok]] = 1
    return X
