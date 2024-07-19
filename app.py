import streamlit as st

st.set_page_config(page_title="Virtual Cell Line", page_icon="🧫", layout="wide")

import torch
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

from datetime import datetime

from app_utils import load_data, load_predicted_umap, load_model, run_simulation, init_session, clear_session

st.markdown("""
        <style>
               .block-container {
                    padding-top: 1rem;
                    padding-bottom: 0rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
        </style>
        """, unsafe_allow_html=True)

# center the title
st.markdown("<h1 style='text-align: center; color: #f63366;'>Virtual Cell Line</h1>", unsafe_allow_html=True)

init_session()

filtered_df = load_data()
all_embeddings = load_predicted_umap()
# predicted_data = load_predicted_data()
# reducer = fit_umap(predicted_data)
model = load_model([filtered_df.shape[1] - 1, 1024])

# @st.cache_data
# def transform_umap(data, inflammation_labels):
#     predicted_data_umap = reducer.transform(data)

#     predicted_data_umap = pd.DataFrame(predicted_data_umap, columns=['UMAP1', 'UMAP2'])
#     predicted_data_umap['Inflammation'] = inflammation_labels

#     return predicted_data_umap

# predicted_data_umap = transform_umap(predicted_data, filtered_df['Inflammation'])

main_columns = st.columns(2)

with main_columns[0]:
    st.markdown("### Controls", help="1. Pick a cell and a gene to perturb.\n2. Change the gene expression to a desired value.\n3. Observe how the expression of other genes changes over time.")
    with st.form('filter_form'):
        # filter_columns = st.columns([5, 2])
        # with filter_columns[0]:
        query = st.text_input("Write a query to pick a random cell with the specified characteristics", 'Inflammation == "UC inflamed" and ACSL4 > 2', help="You can filter cells based on their gene expression values or Inflammation status, as shown in the example. The Inflammation takes one of three values: 'Healthy', 'UC inflamed', 'UC non-inflamed'. The gene expression values can be filtered using the gene names and five comparison operations: '>', '>=', '<', '<=', '=='.")

        # with filter_columns[1]:
        sorted_columns = filtered_df.columns.sort_values().tolist()
        gene_name = st.selectbox("Pick a gene to perturb",sorted_columns , index=sorted_columns.index('ACSL4'))

        is_filter = st.form_submit_button("Select")

    if is_filter:
        clear_session()

        try:
            query_results = filtered_df.query(query).drop('Inflammation', axis=1)
        except Exception as e:
            st.error(f"Invalid query. Please check the syntax and try again.")
            st.stop()

        if query_results.empty:
            selected_cell = filtered_df.drop('Inflammation', axis=1).sample(1, random_state=datetime.now().microsecond)
            st.markdown("No cells found with the specified characteristics. A random cell is selected.")
        else:
            selected_cell = query_results.sample(1, random_state=datetime.now().microsecond)
            st.markdown(f"Found {query_results.shape[0]} cells with the specified characteristics. Picking one at random.")
        
        test_sample_id = selected_cell.index[0]
        test_sample = torch.tensor(selected_cell.values, dtype=torch.float32)

        st.session_state['test_sample_id'] = test_sample_id
        st.session_state['test_sample'] = test_sample
        st.session_state['gene_name'] = gene_name

    if st.session_state['test_sample'] is not None:
        gene_index = filtered_df.columns.get_loc(gene_name)

        with st.form('perturbation_form'):
            initial_expression = st.session_state['test_sample'][0, gene_index].item()
            # perturbation_columns = st.columns(2)
            # with perturbation_columns[0]:
            #     st.markdown(f"Initial {st.session_state['gene_name']} expression: {initial_expression:.4f}")
            # with perturbation_columns[1]:
            min_value = 0.0
            max_value = max(filtered_df[st.session_state['gene_name']].max(), 1)
            target_value = st.slider(f"Change {st.session_state['gene_name']} expression (initial value = {initial_expression:.4f})", min_value=min_value, max_value=max_value, value=initial_expression, step=0.1)

            with st.expander("Advanced Options", expanded=False):
                advanced_columns = st.columns(2)
                with advanced_columns[0]:
                    step_frac = st.slider("Step fraction", min_value=0.01, max_value=1.0, value=0.1, step=0.01, help="The fraction of the difference between the initial and target expression values to change the expression by in each iteration. E.g., if the initial expression is 1, the target expression is 2.5, and the step fraction is 0.1, the expression will be increased by 0.15 in each iteration. To change the expression in a single step just set the step fraction to 1.")
                with advanced_columns[1]:
                    max_iterations = st.slider("Max iterations", min_value=1, max_value=100, value=20, step=1, help="When the target expression is reached, the simulation may still run for some number of iterations if the expression profile of the cell continues to change. The simulation will stop if the expression profile stabilizes or the maximum number of iterations is reached.")

            is_run = st.form_submit_button("Run Simulation")

        if is_run:
            cell_states = run_simulation(model, st.session_state['test_sample'], gene_index, target_value, step_frac=step_frac, max_iterations=max_iterations+1)

            st.session_state['simulated_gene'] = gene_name
            st.session_state['cell_states'] = cell_states

with main_columns[1]:
    st.markdown("### Results")
    with st.container():
        if st.session_state['cell_states'] is not None:
            # st.markdown(f"### Simulation results for sample {st.session_state['test_sample_id']}, gene {st.session_state['simulated_gene']}")

            cell_states = st.session_state['cell_states']

            # Convert cell states to DataFrame
            cell_states_df = pd.DataFrame(cell_states.numpy()[:-1], columns=filtered_df.columns[:-1])

            tabs = st.tabs(["Selected cell", "Most impacted genes", "Gene expression changes over time"])
            with tabs[0]:
                # # Convert cell states to DataFrame for plotting
                # cell_states_embedding = reducer.transform(cell_states.numpy())

                # plot_perturbation_path(predicted_data_umap, cell_states_embedding)

                st.markdown("The plot illustrates all cells in the dataset projected onto a 2D space. The selected cell is highlighted in red.")

                sns.scatterplot(data=all_embeddings, x='UMAP1', y='UMAP2', hue='Inflammation', s=2);
                selected_cell_embedding = all_embeddings[all_embeddings.index == st.session_state['test_sample_id']]
                plt.scatter(selected_cell_embedding['UMAP1'], selected_cell_embedding['UMAP2'], color='red', label='Selected Cell', s=15)

                st.pyplot()

            with tabs[1]:
                # Calculate the most impacted genes
                ranking = (cell_states_df.iloc[-1] - cell_states_df.iloc[0]).sort_values(ascending=False)
                ranking.name = 'Expression Change'

                ranking_columns = st.columns(2)
                with ranking_columns[0]:
                    st.table(ranking.head(20).to_frame().rename(columns={'Expression Change': 'Expression Increase'}).style.format("{:.4f}"))

                with ranking_columns[1]:
                    st.table(ranking.tail(20).sort_values(ascending=True).to_frame().rename(columns={'Expression Change': 'Expression Decrease'}).style.format("{:.4f}"))

            with tabs[2]:
                with st.form('expression_form'):
                    genes_to_inspect = st.multiselect("Select genes to inspect how their expression changed over time", filtered_df.columns[:-1], default=[gene_name], help="The gene that was perturbed is selected by default. You can select multiple genes to compare their expression changes.")

                    is_plot = st.form_submit_button("Plot Gene Expression Changes")

                if is_plot:
                    # Plot gene expression changes
                    gene_expression_fig = px.line(cell_states_df[genes_to_inspect], title='Gene Expression Changes')
                    gene_expression_fig.update_layout(xaxis_title='Iteration', yaxis_title='Expression')

                    st.plotly_chart(gene_expression_fig)

        else:
            st.markdown("No simulation results to display yet. Please run a simulation to see the results.")