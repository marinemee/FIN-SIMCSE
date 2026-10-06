import torch
import torch.nn as nn
import torch.nn.functional as F


class SimCSELoss(nn.Module):
    def __init__(self, temperature=0.05):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        """
        z1, z2: (batch_size, hidden_dim)
        Two differently dropped-out views of the same batch.
        """
        batch_size = z1.size(0)

        # Normalize
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        # Cosine similarity matrix
        sim = torch.mm(z1, z2.t()) / self.temperature   # (B, B)

        # Labels: diagonal is the positive pair
        labels = torch.arange(batch_size).to(z1.device)

        loss = F.cross_entropy(sim, labels)
        return loss