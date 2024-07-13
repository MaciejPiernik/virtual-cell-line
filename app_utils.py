import gzip
import pickle
import torch
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
    with gzip.open('data/GSE116222_Expression_matrix.txt.gz', 'rt') as f:
        expr_matrix = pd.read_csv(f, sep='\t', index_col=0)

    expression_data = expr_matrix.T

    labels = expression_data.index.str.split('-').str[1].to_list()

    inflammation_labels = ['UC inflamed' if x.endswith('3') else 'UC non-inflamed' if x.endswith('2') else 'Healthy' for x in labels]
    inflammation_labels = pd.Series(inflammation_labels, index=expression_data.index, name='Inflammation')

    with open('data/protein_coding_genes.txt', 'r') as f:
        protein_coding_genes = f.read().splitlines()

    filtered_df = expression_data[[gene for gene in expression_data.columns if gene in protein_coding_genes]]

    # drop columns with zero variance
    filtered_df = filtered_df.loc[:, filtered_df.var() > 0]

    dataset = filtered_df.join(inflammation_labels, how='inner')

    return dataset


@st.cache_data
def load_umap():
    return pd.read_csv('data/predicted_umap.csv', index_col=0)


@st.cache_resource
def load_reducer():
    with open('models/umap_reducer.pkl', 'rb') as f:
        reducer = pickle.load(f)

    return reducer
    

@st.cache_resource
def load_model(layer_dims):
    model = Autoencoder(layer_dims)
    model.load_state_dict(torch.load('models/autoencoder_1024.pth'))

    return model


def predict_gene_expression(data, model):
    with torch.no_grad():
        output = model(data)
    return output


def predict_expression(cell, model):
    # create a square matrix with the same cell repeated multiple times, mask the diagonal, predict the expression of the diagonal, collapse the diagonal to a vector
    cell_copy = cell.clone()
    cell_copy = cell_copy.repeat(cell.shape[1], 1)
    mask = torch.eye(cell.shape[1]) == 0
    cell_copy[mask] = 0

    predicted_expression = predict_gene_expression(cell_copy, model)

    predicted_expression = torch.diag(predicted_expression).reshape(1, -1)

    return predicted_expression   


def run_simulation(model, test_sample, gene_index, target_value, max_iterations=100, step_frac=0.1):
    perturbed_sample = test_sample.clone()
    perturbation_gene_expression = perturbed_sample[0, gene_index]
    step = step_frac * (target_value - perturbed_sample[0, gene_index])
    cell_states = []
    predicted_expression = predict_expression(perturbed_sample, model)
    cell_states.append(predicted_expression)
    for i in range(max_iterations):
        perturbation_gene_expression += step
        if np.isclose(perturbation_gene_expression, target_value, atol=1e-5):
            step = 0

        predicted_expression[0, gene_index] = perturbation_gene_expression

        predicted_expression = predict_expression(predicted_expression, model)

        if step == 0 and torch.allclose(predicted_expression, cell_states[-1], atol=1e-5):
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