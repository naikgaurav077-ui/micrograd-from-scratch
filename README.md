# micrograd from scratch

A small autograd engine in pure Python that does backpropagation on scalar values, plus a simple neural network (MLP) trained with gradient descent. I built it to understand how neural networks actually learn.

> **Credit:** This follows Andrej Karpathy's *Neural Networks: Zero to Hero* micrograd lecture. `micrograd_lecture.py` is my write-up while following the lecture; `engine.py`, `nn.py`, the tests and the training demo are my own extended version.

## What's inside

| File | Purpose |
|---|---|
| `engine.py` | Scalar `Value` class that builds the computation graph and runs reverse-mode autodiff (topological sort + chain rule). Supports `+ - * / **`, `tanh`, `relu`, `exp`, `log` |
| `nn.py` | `Neuron`, `Layer`, `MLP` built on the engine |
| `train_demo.py` | Trains an MLP with gradient descent (hinge loss, L2 regularisation, learning-rate decay) to separate two noisy circles |
| `test_micrograd.py` | Gradient check: autograd gradients vs numerical finite-difference gradients |
| `micrograd_lecture.py` | Single-file version (engine + MLP + tiny training loop) written while following the lecture |

## Run it

No dependencies, just Python 3.

```bash
python test_micrograd.py     # verifies gradients are correct
python train_demo.py         # trains the MLP, prints loss and accuracy
python micrograd_lecture.py  # single-file demo
```

## Example output

```
d/dx0: autograd=-0.11652011  numeric=-0.11652011  OK
...
All gradient checks passed.
step   0  loss 0.8800  accuracy 56.7%
step  99  loss 0.1743  accuracy 93.3%
```

## What I learned

- How the chain rule becomes code: every operation stores a local backward function and gradients accumulate through the graph (`+=`, not `=`, because a node can be used more than once).
- Why gradient checking against numerical derivatives is a reliable way to test backprop.
- How a neural network is just parameters, a loss, gradients and repeated small updates.
