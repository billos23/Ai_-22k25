import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split

from preprocess import load_dataset, tokenize, build_vocabulary, vectorize

############################################################
# 1. LOAD FULL TRAIN DATA
############################################################
texts, labels = load_dataset("train")

# Split into train/dev
train_texts, dev_texts, train_labels, dev_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

############################################################
# 2. BUILD VOCAB FROM TRAIN SET
############################################################
vocab, train_tokens = build_vocabulary(train_texts, train_labels)

dev_tokens = [tokenize(t) for t in dev_texts]

X_train = vectorize(train_tokens, vocab)
X_dev   = vectorize(dev_tokens, vocab)

############################################################
# 3. TRAINING SIZES FOR LEARNING CURVES
############################################################
sizes = [500, 1000, 2000, 4000, 8000, 12000, 16000, len(train_labels)]

train_precision = []
train_recall = []
train_f1 = []

dev_precision = []
dev_recall = []
dev_f1 = []

############################################################
# 4. LOOP OVER TRAIN SIZES
############################################################
for s in sizes:
    print(f"Training with {s} examples...")

    X_train_slice = X_train[:s]
    y_train_slice = train_labels[:s]

    model = BernoulliNB()
    model.fit(X_train_slice, y_train_slice)

    # Predictions
    train_pred = model.predict(X_train_slice)
    dev_pred   = model.predict(X_dev)

    # Compute metrics for POSITIVE CLASS = 1
    p_train, r_train, f_train, _ = precision_recall_fscore_support(
        y_train_slice, train_pred, average='binary', pos_label=1
    )

    p_dev, r_dev, f_dev, _ = precision_recall_fscore_support(
        dev_labels, dev_pred, average='binary', pos_label=1
    )

    train_precision.append(p_train)
    train_recall.append(r_train)
    train_f1.append(f_train)

    dev_precision.append(p_dev)
    dev_recall.append(r_dev)
    dev_f1.append(f_dev)

############################################################
# 5. PLOT RESULTS
############################################################
plt.figure(figsize=(10, 6))
plt.plot(sizes, train_precision, label="Train precision")
plt.plot(sizes, dev_precision, label="Dev precision")
plt.xlabel("Training examples")
plt.ylabel("Precision (Positive class)")
plt.title("Learning Curve – Precision")
plt.legend()
plt.grid()
plt.savefig("learning_precision.png")
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(sizes, train_recall, label="Train recall")
plt.plot(sizes, dev_recall, label="Dev recall")
plt.xlabel("Training examples")
plt.ylabel("Recall (Positive class)")
plt.title("Learning Curve – Recall")
plt.legend()
plt.grid()
plt.savefig("learning_recall.png")
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(sizes, train_f1, label="Train F1")
plt.plot(sizes, dev_f1, label="Dev F1")
plt.xlabel("Training examples")
plt.ylabel("F1 (Positive class)")
plt.title("Learning Curve – F1 Score")
plt.legend()
plt.grid()
plt.savefig("learning_f1.png")
plt.close()

print("DONE. Saved: learning_precision.png, learning_recall.png, learning_f1.png")
