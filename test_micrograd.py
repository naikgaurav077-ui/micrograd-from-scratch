"""Gradient check: compare autograd gradients with numerical (finite-difference) gradients."""
import math
from engine import Value


def f(a, b, c):
    x = a * b + c ** 2
    y = (x.tanh() + b.relu()) / (1 + a.exp())
    return y + (c * c + 2).log()


def numeric_grad(vals, i, h=1e-6):
    hi = list(vals); hi[i] += h
    lo = list(vals); lo[i] -= h
    return (f(*[Value(v) for v in hi]).data - f(*[Value(v) for v in lo]).data) / (2 * h)


vals = [0.7, -1.3, 0.4]
a, b, c = [Value(v) for v in vals]
out = f(a, b, c)
out.backward()

ok = True
for i, node in enumerate([a, b, c]):
    num = numeric_grad(vals, i)
    good = math.isclose(node.grad, num, rel_tol=1e-5, abs_tol=1e-7)
    ok &= good
    print(f"d/dx{i}: autograd={node.grad:+.8f}  numeric={num:+.8f}  {'OK' if good else 'MISMATCH'}")
assert ok, "gradient check failed"
print("All gradient checks passed.")
