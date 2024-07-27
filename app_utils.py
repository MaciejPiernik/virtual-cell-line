import torch
import umap
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt

from torch import nn


class Autoencoder(nn.Module):
    def __init__(self, layer_dims):
        super(Autoencoder, self).__init__()
        encoder_layers = []
        for i in range(1, len(layer_dims)):
            encoder_layers.append(nn.Linear(layer_dims[i-1], layer_dims[i]))
            encoder_layers.append(nn.ReLU())
            
        self.encoder = nn.Sequential(*encoder_layers)
        
        decoder_layers = []
        for i in range(len(layer_dims) - 1, 0, -1):
            decoder_layers.append(nn.Linear(layer_dims[i], layer_dims[i-1]))
            decoder_layers.append(nn.ReLU())

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        x = torch.relu(self.encoder(x))
        x = self.decoder(x)
        return x


@st.cache_data
def load_data():
    filtered_df = pd.read_parquet('data/filtered_expression_data.parquet')
    inflammation_labels = pd.read_csv('data/inflammation_labels.csv', index_col=0, header=0, names=['Inflammation']).squeeze()

    dataset = filtered_df.join(inflammation_labels, how='inner')

    return dataset


@st.cache_data
def load_predicted_data():
    return pd.read_parquet('data/predicted_expression_data.parquet')


@st.cache_data
def load_predicted_umap():
    return pd.read_csv('data/predicted_umap.csv', index_col=0)


@st.cache_resource
def fit_umap(data):
    reducer = umap.UMAP()
    reducer = reducer.fit(data)

    return reducer
    

@st.cache_resource
def load_model(layer_dims):
    model = Autoencoder(layer_dims)
    model.load_state_dict(torch.load('models/autoencoder_1024.pth'))
    model.eval()

    return model


# def predict_gene_expression(data, model):
#     with torch.no_grad():
#         output = model(data)
#     return output


# def predict_expression(cell, model):
#     # create a square matrix with the same cell repeated multiple times, mask the diagonal, predict the expression of the diagonal, collapse the diagonal to a vector
#     cell_copy = cell.clone()
#     cell_copy = cell_copy.repeat(cell.shape[1], 1)
#     mask = torch.eye(cell.shape[1]) == 0
#     cell_copy[mask] = 0

#     predicted_expression = predict_gene_expression(cell_copy, model)

#     predicted_expression = torch.diag(predicted_expression).reshape(1, -1)

#     return predicted_expression   


def run_simulation(model, test_sample, gene_index, target_value, max_iterations=100, step_frac=0.1):
    perturbed_sample = test_sample.clone()
    perturbation_gene_expression = perturbed_sample[0, gene_index]
    step = step_frac * (target_value - perturbed_sample[0, gene_index])
    cell_states = []
    with torch.no_grad():
        predicted_expression = model(perturbed_sample)

    cell_states.append(predicted_expression)

    for i in range(max_iterations):
        perturbation_gene_expression += step
        if np.isclose(perturbation_gene_expression, target_value, atol=1e-5):
            step = 0
            perturbation_gene_expression = target_value

        predicted_expression = model.simulate(predicted_expression, [gene_index], [perturbation_gene_expression])

        if step == 0 and torch.allclose(predicted_expression, cell_states[-1], rtol=0, atol=0.01):
            break

        cell_states.append(predicted_expression)

    cell_states = torch.cat(cell_states, dim=0)

    return cell_states


def plot_perturbation_path(all_embeddings, cell_states_embedding):
    sns.scatterplot(data=all_embeddings, x='UMAP1', y='UMAP2', hue='Inflammation', s=2);

    # plot a path following the cell states, coloring each edge and state from red to green, but don't add the colors to the legend
    colors = np.linspace(0, 1, cell_states_embedding.shape[0])
    plt.scatter(cell_states_embedding[0, 0], cell_states_embedding[0, 1], color='red', label='Start')
    plt.scatter(cell_states_embedding[-1, 0], cell_states_embedding[-1, 1], color='green', label='End')
    plt.scatter(cell_states_embedding[1:-1, 0], cell_states_embedding[1:-1, 1], color=plt.cm.RdYlGn(colors[1:-1]))

    # plot lines connecting the cell states, coloring each edge from red to green
    for i in range(1, cell_states_embedding.shape[0]):
        plt.plot([cell_states_embedding[i-1, 0], cell_states_embedding[i, 0]], [cell_states_embedding[i-1, 1], cell_states_embedding[i, 1]], color=plt.cm.RdYlGn(colors[i]), linewidth=2)

    # move the legend outside the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left');

    st.pyplot()


def init_session():
    if 'cell_states' not in st.session_state:
        st.session_state['cell_states'] = None

    if 'test_sample' not in st.session_state:
        st.session_state['test_sample'] = None

    if 'gene_name' not in st.session_state:
        st.session_state['gene_name'] = None

    if 'simulated_gene' not in st.session_state:
        st.session_state['simulated_gene'] = None

    if 'test_sample_id' not in st.session_state:
        st.session_state['test_sample_id'] = None


def clear_session():
    st.session_state['cell_states'] = None
    st.session_state['test_sample'] = None
    st.session_state['gene_name'] = None
    st.session_state['simulated_gene'] = None
    st.session_state['test_sample_id'] = None