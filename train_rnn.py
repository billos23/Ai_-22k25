# train_rnn.py
import joblib
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from dataset_rnn import prepare_datasets, collate_fn
from glove_loader import load_glove_embeddings
from rnn_model import GRUClassifier


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading dataset...")
X_train, y_train, X_test, y_test, vocab = prepare_datasets()

# Check label distribution
print(f"Train labels distribution: 0={sum(y_train == 0).item()}, 1={sum(y_train == 1).item()}")

embeddings = load_glove_embeddings(
    "embeddings/glove.6B.100d.txt",
    vocab,
    embedding_dim=100
)

vocab_size = len(vocab)
print(f"Vocab size: {vocab_size}")

# Save vocab FIRST (before training)
joblib.dump(vocab, "vocab.pkl")
print("Saved vocab.pkl")

# Create dataset with proper label handling
train_dataset = list(zip(X_train, y_train.tolist()))  # Convert tensor to list

train_size = int(0.8 * len(train_dataset))
dev_size   = len(train_dataset) - train_size

train_data, dev_data = random_split(train_dataset, [train_size, dev_size])

# Bigger batch size = faster training
train_loader = DataLoader(train_data, batch_size=128, shuffle=True, collate_fn=collate_fn, num_workers=0, pin_memory=True)
dev_loader   = DataLoader(dev_data, batch_size=128, shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=True)

# Simpler model = faster training
model = GRUClassifier(
    vocab_size=vocab_size,
    embed_dim=100,
    hidden_size=128,
    num_layers=1,  # Reduced from 2 to 1 layer
    pretrained_embeddings=embeddings,
    freeze_embeddings=True,  # Freeze embeddings = faster + less overfitting
    dropout=0.3
).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)  # Higher LR with frozen embeddings

epochs = 4  # Usually converges in 3-4 epochs
train_losses, dev_losses = [], []

print("Training GRU model...")
best_dev = float("inf")

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_losses.append(running_loss / len(train_loader))

    # Evaluate on dev
    model.eval()
    dev_loss = 0.0
    with torch.no_grad():
        for x, y in dev_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            dev_loss += criterion(preds, y).item()

    dev_losses.append(dev_loss / len(dev_loader))

    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_losses[-1]:.4f} | Dev Loss: {dev_losses[-1]:.4f}")

    # Calculate dev accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dev_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            predicted = torch.argmax(preds, dim=1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    
    dev_acc = correct / total
    print(f"  Dev Accuracy: {dev_acc:.4f}")

    # Save best model
    if dev_losses[-1] < best_dev:
        best_dev = dev_losses[-1]
        torch.save(model.state_dict(), "best_rnn_model.pt")
        print("  -> Saved best model")

# Save embeddings AFTER training (with fine-tuned weights)
np.save("trained_embeddings.npy", model.embedding.weight.detach().cpu().numpy())
print("Saved trained_embeddings.npy (fine-tuned)")

# Plot losses
plt.plot(train_losses, label="Train Loss")
plt.plot(dev_losses, label="Dev Loss")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("GRU Training Curve")
plt.savefig("rnn_loss_train.png")
print("Saved rnn_loss_train.png")

print("Training complete.")
