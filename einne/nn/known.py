import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import einne.nn
import einne.nn.functional as fe


class Cognition(nn.Module):
    def __init__(
        self, model_dim, in_dim, num_heads, dropout=0.1, weight_init_variance_scale=1.0
    ):
        super().__init__()
        self.num_heads = num_heads
        self.project_up = nn.Linear(model_dim, model_dim * 2, bias=False)
        self.project_ff = einne.nn.BlockDiagonalLinear(
            model_dim * 2, num_heads, weight_init_variance_scale
        )
        self.project_down = nn.Linear(model_dim * 2, in_dim, bias=False)

        self.cogni_grid = einne.nn.BlockDiagonalLinear(
            in_dim, num_heads, weight_init_variance_scale
        )

        self.cognition = nn.Linear(in_dim, in_dim, bias=False)

        self.a_gate = einne.nn.BlockDiagonalLinear(
            in_dim, num_heads, weight_init_variance_scale
        )
        self.a_param = nn.Parameter(torch.empty([in_dim]))

        self.output_projection = nn.Linear(in_dim, model_dim, bias=False)
        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        fe.rnn_param_init(self.a_param, min_rad=0.9, max_rad=0.999)

    def forward(self, x: torch.Tensor):
        x = self.project_up(x)
        x = self.project_ff(x)
        x: torch.Tensor = self.project_down(x)

        gate_a = F.softplus(F.silu(self.a_gate(x)))
        log_a = -8.0 * F.softplus(self.a_param) * gate_a
        a = torch.exp(log_a)
        x = x * fe.SqrtBoundDerivative.apply(1 + torch.erfc(x) - torch.exp(2 * log_a))

        grid_ctx = F.gelu(self.cogni_grid(x))

        x = x + grid_ctx * a
        x = x + F.gelu(self.cognition(x)) + F.relu(grid_ctx)

        x = F.dropout(self.output_projection(x), p=self.dropout, training=self.training)
        return x


import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision import datasets, transforms


# Create a synthetic dataset with temporal patterns
class MNISTClassifier(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_classes=10):
        super().__init__()

        # Calculate the size after convolutions and pooling
        # Input: 28x28
        self.flatten_size = 28 * 28

        self.embed = nn.Linear(self.flatten_size, input_dim)
        self.cognition = Cognition(input_dim, model_dim, num_heads, dropout=0.2)
        self.att = einne.nn.RGSAttention(input_dim, num_heads)
        self.cognition2 = Cognition(input_dim, model_dim, num_heads, dropout=0.2)
        self.norm = nn.LayerNorm(input_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(input_dim, num_classes),
        )

    def forward(self, x):
        # CNN feature extraction
        x = x.view(-1, self.flatten_size)

        # Cognition processing
        x = self.embed(x)
        x = x.unsqueeze(0)  # Add sequence dimension
        x = self.cognition(x)
        x = self.att(x)
        x = self.cognition2(x)
        x = x[0]  # Take first (and only) sequence element
        x = self.norm(x)
        x = self.classifier(x)
        return x


def train_model(model, train_loader, val_loader, epochs=20):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0

        for batch_data, batch_labels in train_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

        scheduler.step()

        train_loss = epoch_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)

                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print("-" * 50)

    return train_losses, val_losses, train_accs, val_accs


if __name__ == "__main__":
    # Set up data transforms
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean and std
        ]
    )

    # Load MNIST datasets
    train_dataset = datasets.FashionMNIST(
        "./data", train=True, download=True, transform=transform
    )
    val_dataset = datasets.FashionMNIST("./data", train=False, transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=96, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=96, shuffle=False)

    torch.autograd.set_detect_anomaly(True)

    # Initialize and train model
    model = MNISTClassifier(
        input_dim=32, model_dim=16, num_heads=8, num_classes=10
    ).cuda()
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, epochs=20
    )

    # Visualize some examples with predictions
    model.eval()
    plt.figure(figsize=(15, 6))

    with torch.no_grad():
        for i, (data, labels) in enumerate(val_loader):
            if i >= 1:  # Just use first batch
                break

            outputs = model(data.cuda())
            _, predicted = torch.max(outputs.cpu(), 1)

            for j in range(10):  # Show one example of each digit
                # Find first example of this digit
                mask = labels == j
                if not mask.any():
                    continue

                idx = mask.nonzero()[0].item()
                img = data[idx][0].cpu().numpy()

                plt.subplot(2, 5, j + 1)
                plt.imshow(img, cmap="gray")
                plt.title(f"True: {labels[idx]}\nPred: {predicted[idx]}")
                plt.axis("off")

    plt.tight_layout()
    plt.show()

    # Plot training history
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="Train Acc")
    plt.plot(val_accs, label="Val Acc")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    plt.tight_layout()
    plt.show()
