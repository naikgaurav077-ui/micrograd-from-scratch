"""Train the MLP with gradient descent to separate two noisy circles (pure Python)."""
import math
import random
from engine import Value
from nn import MLP

random.seed(1)


def make_data(n=60):
    xs, ys = [], []
    for i in range(n):
        label = 1 if i % 2 == 0 else -1
        r = 0.5 if label == 1 else 1.5
        a = random.uniform(0, 2 * math.pi)
        xs.append([r * math.cos(a) + random.gauss(0, .1), r * math.sin(a) + random.gauss(0, .1)])
        ys.append(label)
    return xs, ys


X, y = make_data()
model = MLP(2, [8, 8, 1])
print("number of parameters:", len(model.parameters()))


def loss_fn():
    scores = [model(x) for x in X]
    losses = [(1 + -yi * s).relu() for yi, s in zip(y, scores)]   # hinge loss
    data_loss = sum(losses) * (1.0 / len(losses))
    reg = 1e-4 * sum((p * p for p in model.parameters()), Value(0.0))
    total = data_loss + reg
    acc = sum((yi > 0) == (s.data > 0) for yi, s in zip(y, scores)) / len(y)
    return total, acc


for k in range(100):
    total, acc = loss_fn()
    model.zero_grad()
    total.backward()
    lr = 0.1 - 0.09 * k / 100                     # learning-rate decay
    for p in model.parameters():
        p.data -= lr * p.grad
    if k % 10 == 0 or k == 99:
        print(f"step {k:3d}  loss {total.data:.4f}  accuracy {acc * 100:.1f}%")
