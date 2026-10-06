import torch
import torch.nn as nn
from transformers import AutoModel


class SimCSEModel(nn.Module):
    def __init__(self, model_name="bert-base-uncased", temperature=0.05):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.temperature = temperature

        hidden_size = self.encoder.config.hidden_size

        # Projection head (only used during training)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Mean pooling
        last_hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        summed = torch.sum(last_hidden * mask, dim=1)
        counted = torch.clamp(mask.sum(dim=1), min=1e-9)
        mean_pooled = summed / counted

        # Projection
        projected = self.mlp(mean_pooled)

        return projected, mean_pooled   # (for training, for inference)