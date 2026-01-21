# Sparse Neural Network

This repository demonstrates a sparse neural network using hard-threshold masking on learned mask parameters. The model learns a sparse linear transformation by pruning a fixed fraction of weights based on learned importance scores. The goal is to study how structured sparsity can be learned in neural networks and how aggressive pruning affects model accuracy, stability, and generalization.

---
## 🔹 Objective
We study a supervised regression problem where a neural network in the present setting must learn a linear or nonlinear mapping under a strict sparsity constraint. More details are as follows:

We have $x\in\mathbb{R}^D$ and the target is the synthetic regression function $y=f(x)$. We aim to construct $\tilde{f}(x)$ using a sparse linear layer. The loss function is the standard mean squared error. 

--- 
## 🔹 Method
We implement a `SparseLinear` layer that learns a weight matrix $W$ and a corresponding importance score matrix $S$. A fixed fraction of connections is pruned using a hard top-k threshold:

$$ M= \mathbb{I}(S>\tau);~~W_{\text{sparse}}=W\otimes M,$$

where $W\otimes M$ represents the Hadamard product, that is, element-wise product between the two linear operators. $M$ is a binary mask and $\tau$ is chosen such that a fixed sparsity ratio is enforced. The sparse layer is embedded inside a simple MLP architecture: `Input → SparseLinear → ReLU → Linear → Output`.

---
## 🔹 Results
The project produces the following outputs (not in order of their python code execution):
- Training loss curve
- Sparse weight visualization
- Prediction vs ground truth plots

---
## 🔹 Quickstart
Install dependencies `pip install -r requirements.txt`
Model Training `python train.py`
Mode Eval `python evaluate.py`

---
🔹 Limitations
It is good to know the limitations of the model and some of them are given as follows: 

We use hard threshold masking, which is non-differentiable. As a result, we get unstable mask learning, grads. might not flow through the pruning decision, and lastly too much of sparsity leads to underfitting. 

Possible future extensions:
- Gradual pruning schedules
- Kernel-Structured sparsity (block/channel)
- Soft masks (sigmoid relaxation)
