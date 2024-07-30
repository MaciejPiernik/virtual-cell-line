import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from VirtualCellLine import VirtualCellLine
from utils import train_epoch, test_epoch


run = datetime.now().strftime('%Y%m%d%H%M%S')

# Load the data
print('Loading data...')
# data = pd.read_parquet('data/filtered_expression_data.parquet')
data = pd.read_csv('data/toy_gene_expression_dataset.csv')

X_train, X_test = train_test_split(data, test_size=0.2, random_state=23)
# # DEBUG: Overfit the model on two samples
# X_train, X_test = data.iloc[:2], data.iloc[:2]

# # Standardize the data
# print('Standardizing the data...')
# scaler = MinMaxScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train.values, dtype=torch.float32)
X_test = torch.tensor(X_test.values, dtype=torch.float32)

# Hyperparameters
input_size = X_train.shape[1]

num_epochs = 100000
learning_rate = 0.001
masking_value = 0
batch_size = 64
min_genes_to_mask = 0.2
max_genes_to_mask = 0.5
patience = 1000
patience_counter = 0
best_loss = float('inf')

# Model
model = VirtualCellLine(input_size)

# Loss function, optimizer, and scheduler
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

# Create data loaders
train_loader = DataLoader(TensorDataset(X_train), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test), batch_size=batch_size, shuffle=False)

train_losses = []
test_losses = []

# Record initial losses
print('Recording initial losses...')
initial_train_loss = test_epoch(min_genes_to_mask, max_genes_to_mask, model, criterion, train_loader)
initial_test_loss = test_epoch(min_genes_to_mask, max_genes_to_mask, model, criterion, test_loader)
train_losses.append(initial_train_loss)
test_losses.append(initial_test_loss)
print(f'Initial losses, Train: {initial_train_loss:.4f}, Test: {initial_test_loss:.4f}')

for epoch in range(num_epochs):
    train_loss = train_epoch(min_genes_to_mask, max_genes_to_mask, model, criterion, optimizer, train_loader)
    train_losses.append(train_loss)

    test_loss = test_epoch(min_genes_to_mask, max_genes_to_mask, model, criterion, test_loader)
    test_losses.append(test_loss)

    scheduler.step(test_loss)

    print(f'Epoch [{epoch}/{num_epochs}] losses, Train: {train_loss:.6f}, Test: {test_loss:.6f}')

    # Save the model after each epoch
    torch.save(model.state_dict(), f'autoencoder_{run}.pth')

    # Save the losses
    losses_df = pd.DataFrame({'train_loss': train_losses, 'test_loss': test_losses})
    losses_df.to_csv(f'autoencoder_losses_{run}.csv', index=False)

    # Early stopping
    if test_loss < best_loss:
        best_loss = test_loss
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print('Early stopping...')
        break