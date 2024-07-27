
import torch

from torch import nn


class BaseSimulator(nn.Module):
    def simulate(self, x, changed_genes, change_values):
        with torch.no_grad():
            x_changed = x.clone()
            x_changed[0, changed_genes] = torch.tensor(change_values)

            changed_output = self(x_changed)
            
        return changed_output