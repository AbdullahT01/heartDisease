import pandas as pd
import numpy as np 

# neural networks need to learn non linear patterns, but the formula is Z = XW + b. That's why we need to pass Z into an activation function

# where the the sigmoid will squash any value between 0 and 1. will be used to the output layer
def sigmoid(x):
    return 1 / (1 + np.exp(-x)) # np.exp(-x) is e to the power of -x

# will be used for backpropagation, the derivatives of the sigmoid function
def sigmoid_deriv(x):
    return sigmoid(x) * (1 - sigmoid(x))

# to replace all negative vals with 0, will be used in the hidden layer
def relu(x):
    return np.maximum(0, x)

# basically return 1 if above 0 and 0 if below 0, used in 	Hidden layer backprop
def relu_deriv(x):
    return (x > 0).astype(float)

# this is to initialze the input layer, hidden layer, output layer
def initialize_params(input_dim, hidden_dim, output_dim):
    np.random.seed(42)
    return {
        "W1": np.random.randn(input_dim, hidden_dim) * 0.01, # Initializes weights from input to hidden layer
        "b1": np.zeros((1, hidden_dim)),                     # Biases for the hidden layer
        "W2": np.random.randn(hidden_dim, output_dim) * 0.01,# Weights from hidden to output layer
        "b2": np.zeros((1, output_dim)),                     # Bias for the output neuron (just 1 in our case)
    }


def forward_pass(X, params): # X is the input data (the samples basically), params is the networks weights and biases
    Z1 = X @ params["W1"] + params["b1"] # @ means dot product, You multiply input X by weight matrix W1, then add bias b1
    A1 = relu(Z1)                        # This adds non-linearity, keeping only the positive values
    Z2 = A1 @ params["W2"] + params["b2"]
    A2 = sigmoid(Z2)                     # Apply sigmoid to turn the output score into a probability between 0 and 1

    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2} # Save all the intermediate steps into a cache diction
    return A2, cache # The output prediction and A cache of intermediate values used later in backpropagation

# this function is at the core of how your model learns. 
# It calculates the binary cross-entropy loss, which tells the network how wrong its predictions are.
def compute_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)  # ✅ new line
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))




def backward_pass(X, y_true, params, cache):
    m = X.shape[0]
    
    # Unpack
    A1 = cache["A1"]
    A2 = cache["A2"]
    Z1 = cache["Z1"]

    # Output layer error (sigmoid derivative + cross-entropy)
    dZ2 = A2 - y_true
    dW2 = A1.T @ dZ2 / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    # Hidden layer error
    dA1 = dZ2 @ params["W2"].T
    dZ1 = dA1 * relu_deriv(Z1)
    dW1 = X.T @ dZ1 / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    # Return gradients
    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}


def update_params(params, grads, lr=0.01):
    params["W1"] -= lr * grads["dW1"]
    params["b1"] -= lr * grads["db1"]
    params["W2"] -= lr * grads["dW2"]
    params["b2"] -= lr * grads["db2"]
    return params


def train(X, y, input_dim, hidden_dim, output_dim, epochs=1000, lr=0.01):
    params = initialize_params(input_dim, hidden_dim, output_dim)
    
    print("🧪 Any NaNs in X?", np.isnan(X).any())
    print("🧪 Any NaNs in y?", np.isnan(y).any())


    for i in range(epochs):
        y_pred, cache = forward_pass(X, params)

        # 🔎 Debug 1: Check if predictions contain nan or 0/1 extremes
        if i == 0:
            print("🔎 y_pred sample:", y_pred[:5].T)
            print("🔎 Any NaNs in y_pred?", np.isnan(y_pred).any())
            print("🔎 Any 0 or 1 in y_pred?", np.any(y_pred == 0), np.any(y_pred == 1))

        loss = compute_loss(y, y_pred)

        # 🔎 Debug 2: Print loss before updating
        if i % 100 == 0 or np.isnan(loss):
            print(f"Epoch {i}, Loss: {loss}")

        if np.isnan(loss):
            print("⚠️ NaN loss detected. Stopping early.")
            break

        grads = backward_pass(X, y, params, cache)
        params = update_params(params, grads, lr)

    return params



def predict(X, params):
    A2, _ = forward_pass(X, params)
    return (A2 > 0.5).astype(int)


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)
