import torch
import torch.nn as nn
from sklearn.metrics import classification_report, precision_recall_fscore_support


def train_model(model, train_loader, val_loader, criterion, optimizer, classes, device, num_epochs=10):
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0

        for images, labels_tuple in train_loader:
            inputs = images.to(device)
            labels = torch.tensor([classes.index(l) for l in labels_tuple]).to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        train_acc = correct_train / total_train
        train_losses.append(epoch_loss)
        train_accuracies.append(train_acc)

        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        with torch.no_grad():
            for images, labels_tuple in val_loader:
                inputs = images.to(device)
                labels = torch.tensor([classes.index(l) for l in labels_tuple]).to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_loss /= len(val_loader.dataset)
        val_acc = correct_val / total_val
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f'Epoch [{epoch+1}/{num_epochs}]  '
              f'Train Loss: {epoch_loss:.4f}  Train Acc: {train_acc:.4f}  '
              f'Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}')

    return model, train_losses, val_losses, train_accuracies, val_accuracies


def evaluate_model(model, data_loader, classes, device, criterion):
    model.eval()
    correct, total, total_loss = 0, 0, 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels_tuple in data_loader:
            inputs = images.to(device)
            labels = torch.tensor([classes.index(l) for l in labels_tuple]).to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader.dataset)
    accuracy = correct / total

    report = classification_report(all_labels, all_preds, target_names=classes, zero_division=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=1
    )

    return avg_loss, accuracy, all_preds, all_labels, report, precision, recall, f1
