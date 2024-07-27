import numpy as np
import torch

from tqdm import tqdm


def train_epoch(min_genes_to_mask, max_genes_to_mask, model, criterion, optimizer, train_loader):
    epoch_losses = []
    for batch, in tqdm(train_loader):
        mask = create_mask(batch, min_genes_to_mask, max_genes_to_mask)

        # Forward pass
        output = model(batch, mask)

        # Compute loss only for the selected genes
        loss = criterion(output[mask.bool()], batch[mask.bool()])
        
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

            losses.append(loss.item())
            
        val_selected_loss = np.mean(losses)

    return val_selected_loss


def create_mask(batch, min_genes_to_mask, max_genes_to_mask):
    mask = torch.zeros_like(batch)
    for i in range(batch.size(0)):
        non_empty_indexes = torch.nonzero(batch[i] != 0).flatten()
        no_of_genes_to_mask = torch.randint(int(min_genes_to_mask * len(non_empty_indexes)), int(max_genes_to_mask * len(non_empty_indexes)), (1,))
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