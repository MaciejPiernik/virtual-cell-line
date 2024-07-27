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
embedding_dim = 128
num_heads = 8
num_transformer_layers = 6
layer_dims = [input_size, embedding_dim]
num_epochs = 1000
dropout_rate = 0
learning_rate = 0.0001
masking_value = 0
batch_size = 32
min_genes_to_mask = 0.5
max_genes_to_mask = 0.8

# Model
# model = Autoencoder(layer_dims, dropout_rate)
model = VirtualCellLine(input_size, embedding_dim=embedding_dim, num_heads=num_heads, num_transformer_layers=num_transformer_layers, dropout_rate=dropout_rate)

# Loss function
criterion = nn.MSELoss()

# Create data loaders
test_loader = DataLoader(TensorDataset(X_test), batch_size=batch_size, shuffle=False)

model.load_state_dict(torch.load('autoencoder.pth'))
model.eval()
