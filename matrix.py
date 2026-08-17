import numpy as np

# ---------- 1. Dot product (1D vectors) ----------
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

dot_result = np.dot(a, b)      # or: a @ b
print("Dot product:", dot_result)  # 32


# ---------- 2. Matrix multiplication (2D arrays) ----------
A = np.array([[1, 2],
              [3, 4]])

B = np.array([[5, 6],
              [7, 8]])

matmul_result = A @ B          # or: np.matmul(A, B) or np.dot(A, B)
print("Matrix multiplication:\n", matmul_result)
# [[19 22]
#  [43 50]]


# ---------- 3. Element-wise multiplication (NOT matrix mult) ----------
elementwise_result = A * B
print("Element-wise multiplication:\n", elementwise_result)
# [[ 5 12]
#  [21 32]]


# ---------- 4. Matrix @ Vector ----------
v = np.array([5, 6])
mv_result = A @ v
print("Matrix-vector product:", mv_result)  # [17 39]