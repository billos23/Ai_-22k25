# train_cnn.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from cnn_model import FashionCNN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data augmentation for training
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# No augmentation for dev/test
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

print("Loading FashionMNIST...")
full_train = datasets.FashionMNIST(root="./data", train=True, download=True, transform=train_transform)
test_dataset = datasets.FashionMNIST(root="./data", train=False, download=True, transform=test_transform)

# Split train into train/dev (80/20)
train_size = int(0.8 * len(full_train))
dev_size = len(full_train) - train_size
train_dataset, dev_dataset = random_split(full_train, [train_size, dev_size])

# Dev set uses test transform (no augmentation)
dev_dataset.dataset.transform = test_transform

print(f"Train: {len(train_dataset)}, Dev: {len(dev_dataset)}, Test: {len(test_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
dev_loader = DataLoader(dev_dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

# Model
model = FashionCNN(num_classes=10, freeze_backbone=False).to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

epochs = 10
train_losses, dev_losses = [], []
best_dev_loss = float("inf")

print("Training CNN...")
for epoch in range(epochs):
    # Training
    model.train()
    running_loss = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    train_losses.append(running_loss / len(train_loader))
    
    # Dev evaluation
    model.eval()
    dev_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dev_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            dev_loss += criterion(out, y).item()
            preds = torch.argmax(out, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    
    dev_losses.append(dev_loss / len(dev_loader))
    dev_acc = correct / total
    
    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_losses[-1]:.4f} | Dev Loss: {dev_losses[-1]:.4f} | Dev Acc: {dev_acc:.4f}")
    
    # Save best model
    if dev_losses[-1] < best_dev_loss:
        best_dev_loss = dev_losses[-1]
        torch.save(model.state_dict(), "best_cnn_model.pt")
        print("  -> Saved best model")

# Plot losses
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(dev_losses, label="Dev Loss")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CNN Training Curve (FashionMNIST)")
plt.savefig("cnn_loss_train.png")
print("Saved cnn_loss_train.png")

print("Training complete.")
