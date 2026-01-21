import torch
import torch.nn as nn

class SparseLinear(nn.Module):
    def __init__(self, in_features, out_features, sparsity=0.9):
        super().__init__()
        self.W = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.mask = nn.Parameter(torch.randn(out_features, in_features))
        self.sparsity = float(sparsity)

    def forward(self, x):
        k = int(self.sparsity * self.mask.numel())
        k = min(max(k, 0), self.mask.numel() - 1)

        threshold = torch.topk(self.mask.flatten(), k, largest=False).values.max() if k > 0 else -float("inf")
        M = (self.mask > threshold).float()

        W_sparse = self.W * M
        return x @ W_sparse.T

    @torch.no_grad()
    def hard_mask(self):
        k = int(self.sparsity * self.mask.numel())
        k = min(max(k, 0), self.mask.numel() - 1)
        threshold = torch.topk(self.mask.flatten(), k, largest=False).values.max() if k > 0 else -float("inf")
        return (self.mask > threshold).float()
