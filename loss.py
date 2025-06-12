import torch
from torch import nn
import torch.nn.functional as F

class FeatureOrthogonalLoss(nn.Module):
    def __init__(self):
        super(FeatureOrthogonalLoss, self).__init__()

    def forward(self, y1, y2):
        return torch.abs(F.cosine_similarity(y1, y2).mean())