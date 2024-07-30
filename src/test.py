import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from Autoencoder import Autoencoder
from VirtualCellLine import VirtualCellLine
from utils import train_epoch, test_epoch


# Load the data
print('Loading data...')
data = pd.read_parquet('data/filtered_expression_data.parquet')

X_train, X_test = train_test_split(data, test_size=0.2, random_state=23)

# Standardize the data
print('Standardizing the data...')
scaler = MinMaxScaler()
_ = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_test = torch.tensor(X_test, dtype=torch.float32)

# Hyperparameters
input_size = X_test.shape[1]
num_epochs = 1000
learning_rate = 0.0001
masking_value = 0
batch_size = 128
min_genes_to_mask = 0.5
max_genes_to_mask = 0.8

# Model
# model = Autoencoder(layer_dims, dropout_rate)
model = VirtualCellLine(input_size)

# Loss function
criterion = nn.MSELoss()

# Create data loaders
test_loader = DataLoader(TensorDataset(X_test), batch_size=batch_size, shuffle=False)

model.load_state_dict(torch.load('autoencoder_20240727191457.pth'))
model.eval()

with torch.no_grad():
    # Create a mask to mask the gene at index 2
    mask = torch.zeros_like(X_test)
    mask[:, 1] = 1

    # Run inference on the first sample from the test set
    test_sample = X_test[0].unsqueeze(0)
    result_single = model(test_sample, mask[0].unsqueeze(0))

    # Run inference on the first batch_size samples from the test set
    result_batch = model(X_test[:batch_size], mask[:batch_size])
    result_batch = result_batch[0].unsqueeze(0)

    loss = criterion(result_single, result_batch)
    print(f'Loss: {loss.item()}')

    print(f"Original: {test_sample[0, :10]}")
    print(f"Single: {result_single[0, 1].item()}")
    print(f"Batch: {result_batch[0, 1].item()}")