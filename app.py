import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.express as px
import gzip
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

from torch import nn


st.set_page_config(layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)


if 'cell_states' not in st.session_state:
    st.session_state['cell_states'] = None


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


@st.cache_data
def load_reducer():
    with open('models/umap_reducer.pkl', 'rb') as f:
        reducer = pickle.load(f)

    return reducer
    

@st.cache_data
def load_model(layer_dims):
    model = Autoencoder(layer_dims)
    model.load_state_dict(torch.load('models/autoencoder_1024.pth'))

    return model


def predict_gene_expression(data):
    with torch.no_grad():
        output = model(data)
    return output


def run_simulation(test_sample, gene_index, target_value, max_iterations=100, step_frac=0.1):
    perturbed_sample = test_sample.clone()
    perturbation_gene_expression = perturbed_sample[0, gene_index]
    step = step_frac * (target_value - perturbed_sample[0, gene_index])
    cell_states = []
    predicted_expression = predict_gene_expression(perturbed_sample)
    cell_states.append(predicted_expression)
    for i in range(max_iterations):
        perturbation_gene_expression += step
        if np.isclose(perturbation_gene_expression, target_value, atol=1e-5):
            step = 0

        predicted_expression[0, gene_index] = perturbation_gene_expression

        predicted_expression = predict_gene_expression(predicted_expression)

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


filtered_df = load_data()
predicted_umap = load_umap()
reducer = load_reducer()
model = load_model([filtered_df.shape[1] - 1, 1024])


# Streamlit app
st.title("Virtual Cell Line")

# User inputs
# gene_name = st.selectbox("Select a gene", filtered_df.columns)
# gene_index = filtered_df.columns.get_loc(gene_name)
# inflammation_status = st.selectbox("Select inflammation status", inflammation_labels.unique())
# gene_range = st.slider("Select gene expression range", min_value=filtered_df[gene_name].min(), max_value=filtered_df[gene_name].max(), value=(filtered_df[gene_name].min(), filtered_df[gene_name].max()))

# # Filter cells based on user input
# filtered_cells = filtered_df[(filtered_df[gene_name] >= gene_range[0]) & (filtered_df[gene_name] <= gene_range[1]) & (inflammation_labels == inflammation_status)]
# sample_id = st.selectbox("Select a cell", filtered_cells.index)
# test_sample = torch.tensor(filtered_df.loc[sample_id].values, dtype=torch.float32).reshape(1, -1)

query = st.text_input("Write a query to filter cells", 'Inflammation == "Healthy"')
selected_cell = filtered_df.query(query).drop('Inflammation', axis=1).sample(1, random_state=23)
test_sample = torch.tensor(selected_cell.values, dtype=torch.float32)

gene_name = st.selectbox("Pick a gene to perturb", filtered_df.columns)
gene_index = filtered_df.columns.get_loc(gene_name)

with st.form('perturbation_form'):
    # Display the initial gene expression on a slider for the user to change (make the initial value the same as the selected cell's expression and mark it with a tick)
    perturbation_columns = st.columns(4)
    with perturbation_columns[0]:
        initial_expression = test_sample[0, gene_index].item()
        st.markdown(f"Initial expression: {initial_expression}")
    with perturbation_columns[1]:
        target_value = st.slider("Change gene expression", min_value=filtered_df[gene_name].min(), max_value=filtered_df[gene_name].max(), value=initial_expression, step=0.1)
    with perturbation_columns[2]:
        step_frac = st.slider("Step fraction", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
    with perturbation_columns[3]:
        max_iterations = st.slider("Max iterations", min_value=1, max_value=100, value=20, step=1)

    is_run = st.form_submit_button("Run Simulation")

plot_columns = st.columns(2)

# Run simulation
if is_run:
    cell_states = run_simulation(test_sample, gene_index, target_value, step_frac=step_frac, max_iterations=max_iterations)

    st.session_state['cell_states'] = cell_states

if st.session_state['cell_states'] is not None:
    cell_states = st.session_state['cell_states']

    with plot_columns[0]:
        # Convert cell states to DataFrame for plotting
        cell_states_embedding = reducer.transform(cell_states.numpy())

        plot_perturbation_path(predicted_umap, cell_states_embedding)

    with plot_columns[1]:
        # Convert cell states to DataFrame
        cell_states_df = pd.DataFrame(cell_states.numpy(), columns=filtered_df.columns[:-1])

        genes_to_inspect = st.multiselect("Select genes to inspect", filtered_df.columns[:-1], default=[gene_name])

        # Plot gene expression changes
        gene_expression_fig = px.line(cell_states_df[genes_to_inspect], title='Gene Expression Changes')
        st.plotly_chart(gene_expression_fig)
else:
    fig = sns.scatterplot(data=predicted_umap, x='UMAP1', y='UMAP2', hue='Inflammation', s=2);

    st.plotly_chart(fig)
