import torch
from torch import nn
import numpy as np
# from datas.dataset_3d import  *
from torch.nn import functional as F


class LinearLayerClip(nn.Module):
    def __init__(self, dim_in, dim_out, k):
        super(LinearLayerClip, self).__init__()
        self.fc = nn.ModuleList([nn.Linear(dim_in, dim_out) for i in range(k)])

    def forward(self, tokens):
        out = []
        for i, t in enumerate(tokens):
            if t.dim() == 3:                    # (B, N, C) from OpenCLIP
                t = t.transpose(0, 1)           # (N, B, C)
                t = self.fc[i](t)               # keep **all** batches
                t = t.transpose(0, 1)           # (B, N, C)  ← optional
            else:                               # legacy 4-D path
                B, C, H, W = t.shape
                t = self.fc[i](
                        t.view(B, C, -1)
                         .permute(0, 2, 1).contiguous())
            out.append(t)
        return out
