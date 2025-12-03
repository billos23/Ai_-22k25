from preprocess import load_dataset, build_vocabulary, vectorize, tokenize
from models import get_models
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import joblib


###########################################################
# LOAD DATASET USING TORCHTEXT
###########################################################
texts, labels = load_dataset("train")   # 25k reviews


###########################################################
# SPLIT INTO TRAIN / DEV
###########################################################
train_texts, dev_texts, train_labels, dev_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)
print("TRAIN LABELS:", np.bincount(train_labels))
print("DEV LABELS:  ", np.bincount(dev_labels))

###########################################################
# BUILD VOCAB
###########################################################
vocab, train_tokens = build_vocabulary(train_texts, train_labels)

dev_tokens = [tokenize(t) for t in dev_texts]

X_train = vectorize(train_tokens, vocab)
X_dev   = vectorize(dev_tokens, vocab)

models = get_models()


###########################################################
# TRAIN + DEV EVAL
###########################################################
for name, model in models.items():
    model.fit(X_train, train_labels)
    preds = model.predict(X_dev)

    p, r, f, _ = precision_recall_fscore_support(dev_labels, preds, average='binary')

    print(f"\n=== {name} ===")
    print(f"Precision: {p:.3f}")
    print(f"Recall:    {r:.3f}")
    print(f"F1 score:  {f:.3f}")


###########################################################
# SAVE VOCAB + MODELS FOR TEST SET
###########################################################
joblib.dump(vocab, "vocab.pkl")
joblib.dump(models, "models.pkl")

print("\nSaved vocab.pkl and models.pkl")
