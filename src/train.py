import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau

from VirtualCellLine import VirtualCellLine
from train_utils import get_cellxgene_data_loaders, train_epoch, test_epoch


run = f"cellxgene_colon_{datetime.now().strftime('%Y%m%d%H%M%S')}"

# Hyperparameters
batch_size = 128
num_epochs = 10000
learning_rate = 0.0001
masking_value = 0
min_genes_to_mask = 0.2
max_genes_to_mask = 0.5
scheduler_patience = 23
patience = 100
patience_counter = 0
best_loss = float('inf')

# Get data loaders
# train_loader, test_loader, input_size = get_data_loaders(batch_size)
train_loader, test_loader, input_size = get_cellxgene_data_loaders(batch_size)

# Model
model = VirtualCellLine(input_size)

# Loss function, optimizer, and scheduler
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=scheduler_patience)

train_losses = []
test_losses = []

# # Record initial losses
# print('Recording initial losses...')
# initial_train_loss = test_epoch(min_genes_to_mask, max_genes_to_mask, model, criterion, train_loader)
# initial_test_loss = test_epoch(min_genes_to_mask, max_genes_to_mask, model, criterion, test_loader)
# train_losses.append(initial_train_loss)
# test_losses.append(initial_test_loss)
# print(f'Initial losses, Train: {initial_train_loss:.4f}, Test: {initial_test_loss:.4f}')

for epoch in range(num_epochs):
    train_loss = train_epoch(min_genes_to_mask, max_genes_to_mask, model, criterion, optimizer, train_loader)
    train_losses.append(train_loss)

    test_loss = test_epoch(min_genes_to_mask, max_genes_to_mask, model, criterion, test_loader)
    test_losses.append(test_loss)

    scheduler.step(test_loss)

    print(f'Epoch [{epoch}/{num_epochs}] losses, Train: {train_loss:.6f}, Test: {test_loss:.6f}, Lr: {scheduler.get_last_lr()[0]}')

    # Save the model after each epoch
    torch.save(model.state_dict(), f'model_{run}.pth')

    # Save the losses
    losses_df = pd.DataFrame({'train_loss': train_losses, 'test_loss': test_losses})
    losses_df.to_csv(f'model_losses_{run}.csv', index=False)

    # Early stopping
    if test_loss < best_loss:
        best_loss = test_loss
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print('Early stopping...')
        break