# Sparse Neural Network

This repository demonstrates a sparse neural network using hard-threshold masking on learned mask parameters. The model learns a sparse linear transformation by pruning a fixed fraction of weights based on learned importance scores. The goal is to study how structured sparsity can be learned in neural networks and how aggressive pruning affects model accuracy, stability, and generalization.

---
## 🔹 Objective
We study a supervised regression problem where a neural network in the present setting must learn a linear or nonlinear mapping under a strict sparsity constraint. More details are as follows:

We have $x\in\mathbb{R}^D$ and the target is the synthetic regression function $y=f(x)$. We aim to construct $\tilde{f}(x)$ using a sparse linear layer. The loss function is the standard mean squared error. 

--- 
## 🔹 Method
We implement a `SparseLinear` layer that learns a weight matrix $W$ and a corresponding importance score matrix $S$. A fixed fraction of connections is pruned using a hard top-k threshold:

$$ M = \mathbb{I}\left(\tau<S\right),\quad W_{\text{sparse}} = W \otimes M$$

$$\left( \sum_{k=1}^n a_k b_k \right)^2 \leq \left( \sum_{k=1}^n a_k^2 \right) \left( \sum_{k=1}^n b_k^2 \right)$$

M=I(S>τ),Wsparse​=W⊙M
