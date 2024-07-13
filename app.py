import streamlit as st
import torch
import pandas as pd
import plotly.express as px

from app_utils import load_data, load_umap, load_reducer, load_model, run_simulation, plot_perturbation_path


st.set_page_config(layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)

if 'cell_states' not in st.session_state:
    st.session_state['cell_states'] = None

if 'test_sample' not in st.session_state:
    st.session_state['test_sample'] = None

if 'gene_name' not in st.session_state:
    st.session_state['gene_name'] = None

if 'simulated_gene' not in st.session_state:
    st.session_state['simulated_gene'] = None

filtered_df = load_data()
predicted_umap = load_umap()
reducer = load_reducer()
model = load_model([filtered_df.shape[1] - 1, 1024])

# Streamlit app
st.title("Virtual Cell Line")

with st.form('filter_form'):
    filter_columns = st.columns([5, 2])
    with filter_columns[0]:
        query = st.text_input("Write a query to pick a random cell with the specified characteristics", 'Inflammation == "Healthy"')
        selected_cell = filtered_df.query(query).drop('Inflammation', axis=1).sample(1, random_state=23)
        test_sample = torch.tensor(selected_cell.values, dtype=torch.float32)

    with filter_columns[1]:
        gene_name = st.selectbox("Pick a gene to perturb", filtered_df.columns.sort_values().tolist(), index=0)
        gene_index = filtered_df.columns.get_loc(gene_name)

    is_filter = st.form_submit_button("Select")

if is_filter:
    st.session_state['test_sample'] = test_sample
    st.session_state['gene_name'] = gene_name


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

if is_run:
    cell_states = run_simulation(model, test_sample, gene_index, target_value, step_frac=step_frac, max_iterations=max_iterations)

    st.session_state['simulated_gene'] = gene_name
    st.session_state['cell_states'] = cell_states


if st.session_state['cell_states'] is not None:
    st.markdown(f"### Simulation results for gene: {st.session_state['simulated_gene']}")

    cell_states = st.session_state['cell_states']

    # Convert cell states to DataFrame
    cell_states_df = pd.DataFrame(cell_states.numpy(), columns=filtered_df.columns[:-1])

    tabs = st.tabs(["Perturbation Path", "Gene Expression Changes", "Most Impacted Genes"])
    with tabs[0]:
        # Convert cell states to DataFrame for plotting
        cell_states_embedding = reducer.transform(cell_states.numpy())

        plot_perturbation_path(predicted_umap, cell_states_embedding)

    with tabs[1]:
        with st.form('expression_form'):
            genes_to_inspect = st.multiselect("Select genes to inspect", filtered_df.columns[:-1], default=[gene_name])

            is_plot = st.form_submit_button("Plot Gene Expression Changes")

        if is_plot:
            # Plot gene expression changes
            gene_expression_fig = px.line(cell_states_df[genes_to_inspect], title='Gene Expression Changes')

            st.plotly_chart(gene_expression_fig)

    with tabs[2]:
        # Calculate the most impacted genes
        ranking = (cell_states_df.iloc[0] - cell_states_df.iloc[-1]).sort_values(ascending=False)
        ranking.name = 'Expression Change'
        ranking = ranking.round(4)

        st.markdown("### The most impacted genes")

        ranking_columns = st.columns(2)
        with ranking_columns[0]:
            st.table(ranking.head(20))

        with ranking_columns[1]:
            st.table(ranking.tail(20).sort_values(ascending=True))

        