# Paste your rnn_model.py code here
# rnn_model.py
import torch
import torch.nn as nn


class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers,
                 pretrained_embeddings, freeze_embeddings=False, dropout=0.3):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embedding.weight.data.copy_(torch.tensor(pretrained_embeddings))
        self.embedding.weight.requires_grad = not freeze_embeddings

        # Note: dropout only applies between layers, so set to 0 if num_layers=1
        self.gru = nn.GRU(
            embed_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        embedded = self.embedding(x)
        outputs, _ = self.gru(embedded)
        pooled, _ = torch.max(outputs, dim=1)
        return self.classifier(pooled)
