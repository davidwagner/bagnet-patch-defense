import torch
import numpy as np

class MarginLogitLayer(torch.nn.Module):
    def __init__(self):
        super(MarginLogitLayer, self).__init__()

    def forward(shape, logits, label_logit):
        mask = logits.ge(label_logit)
        best_other_logit = torch.max(torch.masked_select(logits, mask))
        return label_logit - best_value
