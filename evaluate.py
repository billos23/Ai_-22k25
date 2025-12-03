from preprocess import load_dataset, tokenize, vectorize
import joblib
from sklearn.metrics import precision_recall_fscore_support

###########################################################
# LOAD TEST SET
###########################################################
texts, labels = load_dataset("test")
token_lists = [tokenize(t) for t in texts]

###########################################################
# LOAD SAVED MODELS + VOCAB
###########################################################
vocab = joblib.load("vocab.pkl")
models = joblib.load("models.pkl")

X_test = vectorize(token_lists, vocab)


###########################################################
# EVALUATE ALL MODELS
###########################################################
for name, model in models.items():
    preds = model.predict(X_test)

    p, r, f, _ = precision_recall_fscore_support(labels, preds, average=None)
    p_micro, r_micro, f_micro, _ = precision_recall_fscore_support(labels, preds, average="micro")
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(labels, preds, average="macro")

    print(f"\n=== {name} ===")
    print(f"Class 0 (neg): P={p[0]:.3f}, R={r[0]:.3f}, F1={f[0]:.3f}")
    print(f"Class 1 (pos): P={p[1]:.3f}, R={r[1]:.3f}, F1={f[1]:.3f}")
    print(f"Micro avg:     F1={f_micro:.3f}")
    print(f"Macro avg:     F1={f_macro:.3f}")
