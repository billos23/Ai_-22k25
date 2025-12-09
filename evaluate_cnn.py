# evaluate_cnn.py
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report

from cnn_model import FashionCNN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# FashionMNIST class names
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

print("Loading test data...")
test_dataset = datasets.FashionMNIST(root="./data", train=False, download=True, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(f"Test set size: {len(test_dataset)}")

print("Loading model...")
model = FashionCNN(num_classes=10).to(device)
model.load_state_dict(torch.load("best_cnn_model.pt", map_location=device))
model.eval()


all_preds = []
all_labels = []

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        out = model(x)
        preds = torch.argmax(out, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.numpy())

preds = np.array(all_preds)
labels = np.array(all_labels)


print("\n=== TEST SET RESULTS ===\n")
print(classification_report(
    labels,
    preds,
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0
))


accuracy = (preds == labels).mean()
print(f"Overall Accuracy: {accuracy:.4f}")
