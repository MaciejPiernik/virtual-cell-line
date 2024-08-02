import numpy as np
import torch
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

from tqdm import tqdm


def train_epoch(min_genes_to_mask, max_genes_to_mask, model, criterion, optimizer, train_loader):
    epoch_losses = []
    for batch, in tqdm(train_loader):
        mask = create_mask(batch, min_genes_to_mask, max_genes_to_mask)

        # Forward pass
        output = model(batch, mask)

        loss = criterion(output[mask.bool()], batch[mask.bool()])
        # loss = criterion(output[batch != 0], batch[batch != 0])
        
        epoch_losses.append(loss.item())

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return np.mean(epoch_losses)


def test_epoch(min_genes_to_mask, max_genes_to_mask, model, criterion, test_loader):
    with torch.no_grad():
        losses = []
        for batch, in test_loader:
            mask = create_mask(batch, min_genes_to_mask, max_genes_to_mask)

            output = model(batch, mask)

            loss = criterion(output[mask.bool()], batch[mask.bool()])
            # loss = criterion(output[batch != 0], batch[batch != 0])

            losses.append(loss.item())
            
        val_selected_loss = np.mean(losses)

    return val_selected_loss


def create_mask(batch, min_genes_to_mask, max_genes_to_mask):
    mask = torch.zeros_like(batch)
    for i in range(batch.size(0)):
        non_empty_indexes = torch.nonzero(batch[i] != 0).flatten()
        if min_genes_to_mask < 1:
            no_of_genes_to_mask = torch.randint(int(min_genes_to_mask * len(non_empty_indexes)), int(max_genes_to_mask * len(non_empty_indexes))+1, (1,))
        else:
            no_of_genes_to_mask = torch.randint(int(min_genes_to_mask), int(max_genes_to_mask)+1, (1,))

        if no_of_genes_to_mask == 0:
            continue
        selected_genes = non_empty_indexes[torch.randint(len(non_empty_indexes), (no_of_genes_to_mask,))]

        mask[i, selected_genes] = 1

    # # DEBUG, only mask the first 2 non-zero genes in each sample
    # for i in range(batch.size(0)):
    #     non_empty_indexes = torch.nonzero(batch[i] != 0).flatten()
    #     selected_genes = non_empty_indexes[:5]

    #     mask[i, selected_genes] = 1

    return mask


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

        predicted_expression[0, gene_index] = perturbation_gene_expression

        predicted_expression = model.simulate(predicted_expression, [gene_index], [perturbation_gene_expression])

        if step == 0 and torch.allclose(predicted_expression, cell_states[-1], rtol=0, atol=0.01):
            break

        cell_states.append(predicted_expression)

    cell_states = torch.cat(cell_states, dim=0)

    return cell_states


def validate_dependencies(model, test_data, dependencies):
    model.eval()
    with torch.no_grad():
        for dependent_gene, source_gene in dependencies.items():
            dependent_idx = int(dependent_gene.split('_')[1])
            source_idx = int(source_gene.split('_')[1])
            
            # Original prediction
            original_output = model(test_data)
            
            # Perturb source gene
            perturbed_data = test_data.clone()
            perturbed_data[:, source_idx] += 1 # Increase expression by 1 standard deviation
            perturbed_output = model(perturbed_data)
            
            # Calculate effect on dependent gene
            effect = perturbed_output[:, dependent_idx] - original_output[:, dependent_idx]
            avg_effect = effect.mean().item()

            print(f"Perturbing {source_gene} affects {dependent_gene} by an average of {avg_effect:.4f}")


def verify_sine_dependency(model, gene_source, gene_target, num_points=100):
    model.eval()
    
    # Generate a range of values for the source gene with data scaled using min-max scaling
    x = np.linspace(-3, 3, num_points)  # Assuming standardized data
    x_min = x.min()
    x_max = x.max()
    x = (x - x_min) / (x_max - x_min)
    
    # Create input data
    input_data = torch.zeros((num_points, model.num_genes))
    input_data[:, gene_source] = torch.tensor(x, dtype=torch.float32)
    
    # Get model predictions
    with torch.no_grad():
        output = model(input_data)
    
    # Extract predictions for the target gene
    y_pred = output[:, gene_target].numpy()
    
    # Calculate the true sine values (with the same noise level as in data generation)
    noise_level = 0.1  # Adjust this to match your data generation process
    # reverse the scaling
    x = x * (x_max - x_min) + x_min
    y_true = np.sin(x) + np.random.normal(0, noise_level, num_points)
    y_true = (y_true - y_true.min()) / (y_true.max() - y_true.min())
    
    
    # Calculate correlation
    correlation, _ = pearsonr(y_pred, y_true)
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y_true, label='True Sine (with noise)', alpha=0.5)
    plt.plot(x, y_pred, label='Model Prediction', color='red')
    pure_sine = np.sin(x)
    pure_sine_scaled = (pure_sine - pure_sine.min()) / (pure_sine.max() - pure_sine.min())
    plt.plot(x, pure_sine_scaled, label='Pure Sine', color='green', linestyle='--')
    plt.title(f'Dependency of gene_{gene_target} on gene_{gene_source}')
    plt.xlabel(f'gene_{gene_source} expression')
    plt.ylabel(f'gene_{gene_target} expression')
    plt.legend()
    plt.grid(True)
    
    print(f"Correlation between true (noisy) values and predictions: {correlation:.4f}")
    
    plt.show()