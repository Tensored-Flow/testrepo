"""Model training loop — Category B (GPU / torch dependency)."""

import torch
import torch.nn as nn


def train_model(model, dataloader, optimizer, epochs: int = 10) -> dict:
    """Train a classification model on GPU.

    Cannot be benchmarked without CUDA-capable GPU and real data.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    history = {"loss": [], "accuracy": []}

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        model.train()

        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = correct / total if total > 0 else 0.0

        history["loss"].append(epoch_loss)
        history["accuracy"].append(epoch_acc)

        print(f"Epoch {epoch + 1}/{epochs} — loss: {epoch_loss:.4f}, acc: {epoch_acc:.4f}")

    return history
