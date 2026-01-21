import torch
import yaml
from models.sparse_linear import SparseLinear
from utils.plotting import plot_predictions

with open("configs/sparse_nn.yaml") as f:
    cfg = yaml.safe_load(f)

X_test = torch.randn(cfg["data"]["n_test"], cfg["data"]["input_dim"])
y_test = X_test.sum(dim=1, keepdim=True)

model = SparseLinear(cfg["data"]["input_dim"], cfg["model"]["hidden_dim"], cfg["model"]["sparsity"])
head = torch.nn.Linear(cfg["model"]["hidden_dim"], 1)

pred = head(model(X_test))

plot_predictions(y_test.numpy(), pred.detach().numpy(), save_path="results/prediction.png")
