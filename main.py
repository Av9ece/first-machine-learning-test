import numpy as np

# XOR input and output
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([[0], [1], [1], [0]])

# ReLU and its derivative
def relu(z):
  return np.maximum(0, z)

def relu_deriv(z):
  return (z > 0).astype(float)

# Sigmoid and its derivative
def sigmoid(z):
  return 1 / (1 + np.exp(-z))

def sigmoid_deriv(z):
  s = sigmoid(z)
  return s * (1 - s)

def binary_cross_entropy(y_true, y_pred):
  epsilon = 1e-8 # avoid log(0)
  return -np.mean(y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon))

np.random.seed(42)
n_input, n_hidden, n_output = 2, 4, 1
lr = 0.1
epochs = 10000

# Init weights and biases
W1 = np.random.randn(n_input, n_hidden)
b1 = np.zeros((1, n_hidden))
W2 = np.random.randn(n_hidden, n_output)
b2 = np.zeros((1, n_output))

for epoch in range(epochs):
  # === FORWARD PASS ===
  z1 = X @ W1 + b1
  a1 = relu(z1)
  z2 = a1 @ W2 + b2
  a2 = sigmoid(z2)

  # === LOSS ===
  loss = binary_cross_entropy(y, a2)

  # === BACKPROP ===
  # Derivative of loss w.r.t. a2
  dz2 = a2 - y          # derivative of loss w.r.t. z2
  dW2 = a1.T @ dz2      # gradient of weights from hidden to output
  db2 = np.sum(dz2, axis=0, keepdims=True)

  da1 = dz2 @ W2.T     # backprop into hidden layer
  dz1 = da1 * relu_deriv(z1) # apply relu derivative
  dW1 = X.T @ dz1       # gradient of weights from input to hidden
  db1 = np.sum(dz1, axis=0, keepdims=True)

  # === PARAMETER UPDATE ===
  W2 -= lr * dW2
  b2 -= lr * db2
  W1 -= lr * dW1
  b1 -= lr * db1

  if epoch % 1000 == 0:
    print(f"Epoch {epoch}, Loss: {loss:.4f}")

# === ACCURACY CHECK ===
preds = (a2 > 0.5).astype(int)
accuracy = np.mean(preds == y)
print("Final accuracy:", accuracy)

# === VISUALIZATION ===
import matplotlib.pyplot as plt

# Plot decision boundary
def plot_decision_boundary(X, y, model):
    h = 0.01
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    input_grid = np.c_[xx.ravel(), yy.ravel()]
    z1 = input_grid @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    a2 = sigmoid(z2)
    Z = a2.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap="coolwarm", alpha=0.6)
    plt.scatter(X[:,0], X[:,1], c=y.flatten(), edgecolors='k')
    plt.title("Decision Boundary")
    plt.show()

plot_decision_boundary(X, y, None)
