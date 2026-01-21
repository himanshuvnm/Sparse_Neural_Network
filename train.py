import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.sparse_linear import SparseLinear
from utils.device import get_device
from utils.seed import set_seed

# Load config
with open("configs/sparse_nn.yaml") as f:
    cfg = yaml.safe_load(f)

set_seed(cfg["seed"])
device = get_device(cfg["device"])

# Toy regression dataset
X = torch.randn(cfg["data"]["n_train"], cfg["data"]["input_dim"])
y = X.sum(dim=1, keepdim=True)

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=cfg["train"]["batch_size"], shuffle=True)

# Model
model = SparseLinear(cfg["data"]["input_dim"], cfg["model"]["hidden_dim"], cfg["model"]["sparsity"]).to(device)
head = nn.Linear(cfg["model"]["hidden_dim"], 1).to(device)

optimizer = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=cfg["train"]["lr"])
loss_fn = nn.MSELoss()

# Training
for epoch in range(1, cfg["train"]["epochs"] + 1):
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)

        z = model(xb)
        pred = head(z)
        loss = loss_fn(pred, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % cfg["train"]["print_every"] == 0:
        print(f"Epoch {epoch:04d} | Loss = {loss.item():.6f}")
