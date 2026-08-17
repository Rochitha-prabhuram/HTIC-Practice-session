"""
Visualize common neural network activation functions and their gradients.
Run in VSCode: right-click -> Run Python File, or use the Jupyter-style
'# %%' cells with the Python extension for an interactive experience.
"""

import numpy as np
import matplotlib.pyplot as plt

# ---- Activation functions and their derivatives ----

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_grad(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)

def tanh_grad(x):
    return 1 - np.tanh(x) ** 2

def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(float)

def leaky_relu(x, alpha=0.1):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_grad(x, alpha=0.1):
    return np.where(x > 0, 1.0, alpha)

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def gelu_grad(x, h=1e-4):
    # numerical derivative (simple, avoids the messy closed-form expression)
    return (gelu(x + h) - gelu(x - h)) / (2 * h)


functions = {
    "Sigmoid":     (sigmoid,     sigmoid_grad),
    "Tanh":        (tanh,        tanh_grad),
    "ReLU":        (relu,        relu_grad),
    "Leaky ReLU":  (leaky_relu,  leaky_relu_grad),
    "GELU":        (gelu,        gelu_grad),
}

# ---- Plot ----

x = np.linspace(-5, 5, 500)

fig, axes = plt.subplots(1, len(functions), figsize=(20, 4), sharey=False)

for ax, (name, (f, df)) in zip(axes, functions.items()):
    ax.plot(x, f(x), label="f(x)", color="#2a78d6", linewidth=2)
    ax.plot(x, df(x), label="f'(x)", color="#eb6834", linewidth=2, linestyle="--")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_title(name)
    ax.set_xlabel("x")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("activation_functions.png", dpi=150)
plt.show()