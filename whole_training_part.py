import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GlobalAttention
import random
import numpy as np
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

# Define your dataset and collate_fn here

# Collate function
def collate_fn(batch):
    sequences, graph_data, features, labels = zip(*batch)
    padded_sequences = torch.stack(sequences)
    batched_graph = Batch.from_data_list(graph_data)
    features_stacked = torch.stack(features)
    labels_stacked = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    return padded_sequences, batched_graph, features_stacked, labels_stacked, batched_graph.batch

# Define the SEDenseLayer, SEBlock, and ProteinClassifier classes here

# Set random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

# Dataloaders
data_loader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

# Training and evaluation functions
def train_one_epoch(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for sequences, graph_data, features, labels, batch_index in data_loader:
        sequences, graph_data, features, labels, batch_index = (
            sequences.to(device),
            graph_data.to(device),
            features.to(device),
            labels.to(device),
            batch_index.to(device)
        )
        optimizer.zero_grad()
        outputs = model(sequences, graph_data, features, batch_index)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    avg_loss = total_loss / total
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for sequences, graph_data, features, labels, batch_index in data_loader:
            sequences, graph_data, features, labels, batch_index = (
                sequences.to(device),
                graph_data.to(device),
                features.to(device),
                labels.to(device),
                batch_index.to(device)
            )
            outputs = model(sequences, graph_data, features, batch_index)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_loss = total_loss / total
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs, device, patience=20):
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)

        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%')
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

        # Early stopping and saving the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), 'best_model.pth') 
            print("Best model saved!") 
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f'Early stopping at epoch {epoch+1}!')
                break

# Initialize and train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ProteinClassifier().to(device)

# Initialize weights
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

initialize_weights(model)

# Adjust learning rate if necessary
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Handle class imbalance if needed

criterion = torch.nn.BCEWithLogitsLoss()

train_model(model, train_loader, val_loader, optimizer, criterion, 400, device,patience = 50)
