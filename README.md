# Εργασία 2 - Τεχνητή Νοημοσύνη

## Απαιτήσεις

- Python 3.10+
- pip

## Εγκατάσταση

```bash
pip install numpy scikit-learn matplotlib joblib torch torchvision torchtext
```

## Δεδομένα

1. **IMDB Dataset**: Αποσυμπιέστε το `aclImdb_v1.tar.gz` στον φάκελο `imdb/`
2. **GloVe Embeddings**: Κατεβάστε το `glove.6B.zip` από https://nlp.stanford.edu/projects/glove/ και αποσυμπιέστε στον φάκελο `embeddings/`
3. **FashionMNIST**: Κατεβαίνει αυτόματα κατά την εκτέλεση

## Εκτέλεση

### Μέρος Α (Κλασική Μηχανική Μάθηση)

```bash
# Εκπαίδευση μοντέλων
python train_dev_test.py

# Καμπύλες μάθησης
python learning_curves.py

# Αξιολόγηση στο test set
python evaluate.py
```

### Μέρος Β (RNN)

```bash
# Εκπαίδευση GRU
python train_rnn.py

# Αξιολόγηση στο test set
python evaluate_rnn.py
```

### Μέρος Γ (CNN)

```bash
# Εκπαίδευση CNN
python train_cnn.py

# Αξιολόγηση στο test set
python evaluate_cnn.py
```

## Αρχεία

| Αρχείο | Περιγραφή |
|--------|-----------|
| `preprocess.py` | Προεπεξεργασία κειμένου, vocabulary, vectorization |
| `models.py` | Ορισμός μοντέλων sklearn |
| `train_dev_test.py` | Εκπαίδευση Μέρους Α |
| `learning_curves.py` | Καμπύλες μάθησης Μέρους Α |
| `evaluate.py` | Αξιολόγηση Μέρους Α |
| `dataset_rnn.py` | Προεπεξεργασία για RNN |
| `glove_loader.py` | Φόρτωση GloVe embeddings |
| `rnn_model.py` | Αρχιτεκτονική GRU |
| `train_rnn.py` | Εκπαίδευση Μέρους Β |
| `evaluate_rnn.py` | Αξιολόγηση Μέρους Β |
| `cnn_model.py` | Αρχιτεκτονική CNN |
| `train_cnn.py` | Εκπαίδευση Μέρους Γ |
| `evaluate_cnn.py` | Αξιολόγηση Μέρους Γ |

## Παραγόμενα Αρχεία

- `learning_precision.png`, `learning_recall.png`, `learning_f1.png` - Καμπύλες Μέρους Α
- `rnn_loss_train.png` - Loss curves Μέρους Β
- `cnn_loss_train.png` - Loss curves Μέρους Γ
