import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(42)


# ----------------------------- Data -----------------------------------
def load_data():
    digits = load_digits()
    X = digits.data.astype(np.float64) / 16.0          # scale pixels to [0, 1]
    y = digits.target.astype(np.int64)

    n_classes = 10
    Y = np.eye(n_classes)[y]                            # one-hot labels

    X_train, X_test, y_train, y_test, Y_train, Y_test = train_test_split(
        X, y, Y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test, Y_train, Y_test


# ----------------------------- Model ------------------------------------
class MLP:
    """A simple fully-connected network: input -> hidden (ReLU) -> output (softmax)."""

    def __init__(self, n_in, n_hidden, n_out, lr=0.1):
        # He initialization for ReLU layer, Xavier-ish for output layer
        self.W1 = rng.normal(0, np.sqrt(2.0 / n_in), size=(n_in, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(0, np.sqrt(1.0 / n_hidden), size=(n_hidden, n_out))
        self.b2 = np.zeros(n_out)
        self.lr = lr

    @staticmethod
    def relu(z):
        return np.maximum(0, z)

    @staticmethod
    def relu_grad(z):
        return (z > 0).astype(z.dtype)

    @staticmethod
    def softmax(z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2

    @staticmethod
    def cross_entropy(probs, Y_onehot):
        eps = 1e-9
        return -np.mean(np.sum(Y_onehot * np.log(probs + eps), axis=1))

    def backward(self, X, Y_onehot):
        n = X.shape[0]

        # Output layer gradient (softmax + cross-entropy combine nicely)
        dz2 = (self.a2 - Y_onehot) / n
        dW2 = self.a1.T @ dz2
        db2 = dz2.sum(axis=0)

        # Hidden layer gradient
        da1 = dz2 @ self.W2.T
        dz1 = da1 * self.relu_grad(self.z1)
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0)

        # SGD update
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)


# ----------------------------- Training ---------------------------------
def train(model, X_train, Y_train, X_test, y_test, epochs=100, batch_size=32):
    n = X_train.shape[0]
    loss_history = []
    test_acc_history = []

    for epoch in range(epochs):
        perm = rng.permutation(n)
        X_shuf, Y_shuf = X_train[perm], Y_train[perm]

        epoch_losses = []
        for start in range(0, n, batch_size):
            end = start + batch_size
            xb, yb = X_shuf[start:end], Y_shuf[start:end]

            probs = model.forward(xb)
            loss = model.cross_entropy(probs, yb)
            epoch_losses.append(loss)

            model.backward(xb, yb)

        avg_loss = float(np.mean(epoch_losses))
        loss_history.append(avg_loss)

        preds = model.predict(X_test)
        acc = float(np.mean(preds == y_test))
        test_acc_history.append(acc)

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d} | loss {avg_loss:.4f} | test acc {acc:.4f}")

    return loss_history, test_acc_history


# ----------------------------- Main -------------------------------------
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, Y_train, Y_test = load_data()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    model = MLP(n_in=64, n_hidden=64, n_out=10, lr=0.5)
    loss_history, test_acc_history = train(
        model, X_train, Y_train, X_test, y_test, epochs=100, batch_size=32
    )

    final_acc = test_acc_history[-1]
    print(f"\nFinal test accuracy: {final_acc:.4f}")

    # ------------------------- Plot -------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.plot(loss_history, color="#2563eb", linewidth=2)
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.grid(alpha=0.3)

    ax2.plot(test_acc_history, color="#16a34a", linewidth=2)
    ax2.set_title("Test Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1.05)
    ax2.grid(alpha=0.3)

    fig.suptitle("From-Scratch NumPy MLP on Digits (MNIST-style) Dataset", fontsize=13)
    fig.tight_layout()
    fig.savefig("mlp_loss_curve.png", dpi=150)
    print("Saved plot to mlp_loss_curve.png")