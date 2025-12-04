Μέρος Α: Προεπεξεργασία και Αναπαράσταση Κειμένου
1. Εισαγωγή

Στο παρόν μέρος πραγματοποιήθηκε η προεπεξεργασία του IMDB sentiment analysis dataset, με στόχο τη δημιουργία μίας ποιοτικής και αποδοτικής αναπαράστασης κειμένου που θα χρησιμοποιηθεί για εκπαίδευση κλασικών αλγορίθμων Μηχανικής Μάθησης.
Χρησιμοποιήθηκε το τοπικό IMDB dataset (pos/neg) και όχι η έκδοση του torchtext, ώστε να διασφαλιστεί ότι και οι δύο κλάσεις φορτώνονται σωστά.

2. Φόρτωση και Οργάνωση Δεδομένων

Η δομή του dataset είναι:

imdb/
   train/
      pos/
      neg/
   test/
      pos/
      neg/


Για τη φόρτωση χρησιμοποιήθηκε custom loader που:

διαβάζει όλα τα .txt αρχεία από pos και neg,

προσθέτει ετικέτες (0=neg, 1=pos),

τα ανακατεύει για να αποφευχθεί bias από τη σειρά των αρχείων.

Μετά τη φόρτωση επιβεβαιώθηκε ότι οι κλάσεις είναι ισορροπημένες:

TRAIN LABELS: [10000 10000]
DEV LABELS:   [2500 2500]

3. Tokenization

Το tokenization έγινε με κανονικές εκφράσεις (regex):

όλα τα γράμματα μετατρέπονται σε πεζά,

κρατούνται μόνο αλφαβητικοί χαρακτήρες,

δεν γίνεται αφαίρεση token λόγω μήκους (π.χ. "I", "bad", "am" κρατιούνται).

Αυτό διασφαλίζει ότι δεν θα χαθεί σημασιολογικό περιεχόμενο και ότι καμία κλάση δεν θα εξαφανιστεί λόγω υπερβολικού φιλτραρίσματος.

Παράδειγμα tokenization:

Κείμενο	Tokens
"I loved this movie! Amazing acting."	["i", "loved", "this", "movie", "amazing", "acting"]
4. Document Frequency (DF)

Για κάθε λέξη υπολογίστηκε το πλήθος των εγγράφων στα οποία εμφανίζεται (όχι το πλήθος εμφανίσεων).
Χρησιμοποιήθηκε Counter πάνω στα set(tokens) ώστε οι πολυεμφανίσεις μέσα στο ίδιο κείμενο να μην μετρούν.

Απομακρύνθηκαν λέξεις με DF < 5 ώστε να αποφευχθούν εξαιρετικά σπάνια tokens.

5. Αφαίρεση πιο συχνών και πιο σπάνιων tokens

Από το DF αφαιρέθηκαν:

τα 200 πιο συχνά tokens (n=200)

τα 5 πιο σπάνια (k=5)

Σκοπός:

τα πολύ συχνά tokens (π.χ. "the", "and", "is") δεν έχουν πληροφορία για συναίσθημα,

τα υπερβολικά σπάνια θορυβούν την IG.

Το αποτέλεσμα είναι ένα καθαρό σύνολο λέξεων πριν τον IG.

6. Υπολογισμός Information Gain (IG)

Για κάθε υποψήφιο token υπολογίστηκε:

η εντροπία H(Y),

η εντροπία παρουσίας H(Y|token),

η IG ως:

IG(token)=H(Y)−P(token)⋅H(Y∣token)−P(¬token)⋅H(Y∣¬token)


Υλοποιήθηκε βελτιστοποιημένη και γρήγορη IG, που υπολογίζει:

πόσα θετικά & αρνητικά κείμενα περιέχουν το token,

πόσα δεν το περιέχουν,

χωρίς διπλές επαναλήψεις ή αργά nested loops.

7. Επιλογή Λεξιλογίου (Vocabulary)

Τα tokens ταξινομήθηκαν βάσει Information Gain και επιλέχθηκαν:

top 3000 tokens (m=3000)

Τα tokens αντιστοιχήθηκαν σε δείκτες:

vocab[word] = index


Το παραγόμενο λεξιλόγιο αποθηκεύτηκε ως:

vocab.pkl

8. Μετατροπή Κειμένου σε Δυαδικό Διάνυσμα (Binary Bag-of-Words)

Για κάθε κείμενο:

δημιουργήθηκε πίνακας X[i, j] = 1 αν η λέξη vocab[j] υπάρχει στο κείμενο,

αλλιώς 0.

Η τελική αναπαράσταση είναι διαστάσεων:

Train:  20000 × 3000
Dev:     5000 × 3000


Αυτή η μορφή είναι κατάλληλη για Bernoulli Naive Bayes, Logistic Regression και Random Forests.

9. Outputs του Μέρους Α

Το ΜΕΡΟΣ Α παράγει:

✔ vocab.pkl (λεξιλόγιο)
✔ vectorized train/dev sets μέσω του κώδικα
✔ πλήρη διαδικασία tokenization → filtering → DF → IG → vocab → vectorization

Αυτή η έξοδος χρησιμοποιήθηκε επιτυχώς στο ΜΕΡΟΣ Β.

🔥 ΣΥΜΠΕΡΑΣΜΑ ΜΕΡΟΥΣ Α

Η διαδικασία προεπεξεργασίας ολοκληρώθηκε με επιτυχία και τα αποτελέσματα ήταν:

πλήρως ισορροπημένο dataset,

σωστή εξαγωγή Web–ready vocabulary,

ταχύτερος και ορθός υπολογισμός Information Gain,

σωστό binary Bag-of-Words representation,

 Αναφορά υπερ-παραμέτρων
n = 200

k = 5

m = 3000

RandomForest: n_estimators = 200

Logistic Regression: solver = liblinear, C = 1.0

Μερος Β
⭐ ΜΕΡΟΣ Β – Αναγνώριση Συναισθήματος σε Κριτικές IMDB με GRU RNN
1. Εισαγωγή

Στο Μέρος Β υλοποίησα ένα μοντέλο Αναγνώρισης Συναισθήματος (Sentiment Analysis) χρησιμοποιώντας αναδρομικό νευρωνικό δίκτυο τύπου GRU (Gated Recurrent Unit).
Στόχος ήταν η ταξινόμηση κριτικών ταινιών από το dataset IMDB σε δύο κατηγορίες:
θετικές και αρνητικές.

Το μοντέλο επεξεργάζεται τις κριτικές σε μορφή ακολουθιών tokens και μαθαίνει να αναγνωρίζει μοτίβα που συνδέονται με το συναίσθημα του κειμένου.

2. Προεπεξεργασία Δεδομένων

Για τη δημιουργία του συνόλου εκπαίδευσης χρησιμοποιήθηκαν τα δεδομένα IMDB του torchtext.

Βήματα προεπεξεργασίας:

Όλα τα κείμενα μετατράπηκαν σε πεζά.

Αφαιρέθηκαν σύμβολα εκτός από γράμματα και αριθμούς.

Έγινε tokenization σε απλό word-level (split με βάση το κενό).

Χτίστηκε vocabulary με κατώτατο όριο εμφάνισης min_freq=2.

Προστέθηκαν ειδικά tokens:

<pad>: index 0

<unk>: index 1

Κάθε review μετατράπηκε σε sequence από ακέραιους (token IDs).

Τα sequences γέμισαν με padding ώστε να έχουν κοινό μήκος μέσα στα batches.

3. Αρχιτεκτονική του Μοντέλου

Χρησιμοποίησα τη δομή:

Embedding layer: 100 διαστάσεων (pretrained GloVe 6B 100d)

2-layer GRU, bidirectional

Hidden size = 128

Dropout = 0.3

Max pooling πάνω στα hidden states

Fully connected classifier:

Linear → ReLU → Dropout → Linear → Softmax

Ο συνδυασμός Bi-GRU και max-pooling επιτρέπει στο μοντέλο να “συνοψίζει” μια ολόκληρη πρόταση σε ένα σταθερού μεγέθους vector.

4. Εκπαίδευση

Hyperparameters used:

GRU layers: 1 (bidirectional)
Hidden size: 128
Embedding: GloVe 6B 100d (frozen)
Dropout: 0.3
Optimizer: Adam (lr=0.001)
Batch size: 128
Max sequence length: 200
Best epoch: 4 (selected via dev loss)

5. Αξιολόγηση & Αποτελέσματα

Για την αξιολόγηση χρησιμοποιήθηκε το test set του IMDB (25.000 κριτικές).
Τα δεδομένα προεπεξεργάστηκαν με ακριβώς το ίδιο pipeline που χρησιμοποιήθηκε στην εκπαίδευση, ώστε να εξασφαλιστεί συμβατότητα των token IDs και του vocabulary.

Μετά τη διαδικασία inference σε batches των 32 δειγμάτων, υπολογίστηκαν τα κλασικά metrics:
precision, recall, f1-score και accuracy.

(Εδώ θα βάλεις το classification_report που έβγαλε το evaluate.)
          precision    recall  f1-score   support

  === TEST SET RESULTS ===

              precision    recall  f1-score   support

         neg     0.8816    0.8610    0.8712     12500
         pos     0.8642    0.8844    0.8742     12500

    accuracy                         0.8727     25000
   macro avg     0.8729    0.8727    0.8727     25000
weighted avg     0.8729    0.8727    0.8727     25000

6. Συμπεράσματα

Το GRU μοντέλο κατάφερε να μάθει αποτελεσματικά τις δομές των κειμένων και να αναγνωρίσει το συναίσθημα των κριτικών IMDB.
Η χρήση προεκπαιδευμένων GloVe embeddings βοήθησε σημαντικά, καθώς ενίσχυσε την αντιστοίχιση λέξεων με παρόμοιες σημασίες.

Πιθανές βελτιώσεις:

Αύξηση του embedding dimension

Χρήση LSTM ή Transformer αρχιτεκτονικής

Regularization με μεγαλύτερο dropout

Fine-tuning των embeddings

Περισσότερα epochs

Συνολικά, το μοντέλο παρουσίασε ικανοποιητική ακρίβεια και γενικά καλή απόδοση στη ταξινόμηση συναισθήματος.


Μερος Γ
Epoch 1/10 | Train Loss: 0.4707 | Dev Loss: 0.2759 | Dev Acc: 0.8973
  -> Saved best model
Epoch 2/10 | Train Loss: 0.2706 | Dev Loss: 0.2558 | Dev Acc: 0.9061
  -> Saved best model
Epoch 3/10 | Train Loss: 0.2247 | Dev Loss: 0.2318 | Dev Acc: 0.9156
  -> Saved best model
Epoch 4/10 | Train Loss: 0.1925 | Dev Loss: 0.2310 | Dev Acc: 0.9146
  -> Saved best model
Epoch 5/10 | Train Loss: 0.1692 | Dev Loss: 0.2340 | Dev Acc: 0.9162
Epoch 6/10 | Train Loss: 0.1500 | Dev Loss: 0.2245 | Dev Acc: 0.9221
  -> Saved best model
Epoch 7/10 | Train Loss: 0.1311 | Dev Loss: 0.2472 | Dev Acc: 0.9173
Epoch 8/10 | Train Loss: 0.1157 | Dev Loss: 0.2289 | Dev Acc: 0.9231
Epoch 9/10 | Train Loss: 0.1021 | Dev Loss: 0.2545 | Dev Acc: 0.9223
Epoch 10/10 | Train Loss: 0.0919 | Dev Loss: 0.2735 | Dev Acc: 0.9159
Saved cnn_loss_train.png
Training complete.


=== TEST SET RESULTS ===

              precision    recall  f1-score   support

 T-shirt/top     0.8537    0.8810    0.8671      1000
     Trouser     0.9940    0.9890    0.9915      1000
    Pullover     0.8691    0.8900    0.8794      1000
       Dress     0.9127    0.9100    0.9114      1000
        Coat     0.8705    0.8470    0.8586      1000
      Sandal     0.9820    0.9820    0.9820      1000
       Shirt     0.7796    0.7500    0.7645      1000
     Sneaker     0.9677    0.9600    0.9639      1000
         Bag     0.9754    0.9930    0.9841      1000
  Ankle boot     0.9643    0.9710    0.9676      1000

    accuracy                         0.9173     10000
   macro avg     0.9169    0.9173    0.9170     10000
weighted avg     0.9169    0.9173    0.9170     10000