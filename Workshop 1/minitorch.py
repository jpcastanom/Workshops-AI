"""
MiniTorch - A minimal deep learning framework built from scratch
Educational implementation for understanding neural networks
"""

import torch


class Net:
    """
    A simple sequential container for custom layers.
    Provides PyTorch-like train()/eval() switches and
    runs forward/backward/update across all layers.
    """

    def __init__(self):
        """
        Start with an empty list of layers and set the network
        to training mode by default.
        """
        self.layers = []
        self.training = True

    def add(self, layer):
        """Add a layer to the network."""
        self.layers.append(layer)

    def train(self):
        """Switch the whole network to training mode."""
        self.training = True
        for layer in self.layers:
            if hasattr(layer, "train"):
                layer.train()
        return self

    def eval(self):
        """Switch the whole network to evaluation mode."""
        self.training = False
        for layer in self.layers:
            if hasattr(layer, "eval"):
                layer.eval()
        return self

    def forward(self, X):
        """Forward pass through all layers."""
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, dZ):
        """Backward pass through all layers in reverse order."""
        for layer in reversed(self.layers):
            dZ = layer.backward(dZ)
        return dZ

    def update(self, lr):
        """Update parameters of all trainable layers."""
        for layer in self.layers:
            if hasattr(layer, "update"):
                layer.update(lr)


class Linear:
    """
    Fully connected (dense) layer.
    Performs: Z = XW + b
    """

    def __init__(self, nin, nout, device="cpu"):
        """Initialize weights and biases."""
        # Xavier/Glorot initialization for better training
        self.W = torch.randn(nin, nout, device=device) * (2.0 / nin) ** 0.5
        self.b = torch.zeros(nout, device=device)
        self.training = True

    def train(self):
        self.training = True
        return self

    def eval(self):
        self.training = False
        return self

    def forward(self, X):
        """
        Forward pass: Z = XW + b
        X: (batch_size, nin)
        W: (nin, nout)
        Z: (batch_size, nout)
        """
        self.X = X  # Store for backward pass
        Z = torch.matmul(X, self.W) + self.b
        return Z

    def backward(self, dZ):
        """
        Backward pass: compute gradients w.r.t. W, b, and X.
        dZ: (batch_size, nout) - gradient from next layer
        """
        batch_size = self.X.size(0)
        
        # Gradient w.r.t. weights: dW = X^T @ dZ
        self.dW = torch.matmul(self.X.T, dZ) / batch_size
        
        # Gradient w.r.t. bias: db = sum(dZ) over batch
        self.db = torch.sum(dZ, dim=0) / batch_size
        
        # Gradient w.r.t. input: dX = dZ @ W^T
        self.dX = torch.matmul(dZ, self.W.T)
        
        return self.dX

    def update(self, lr):
        """Update parameters using gradient descent."""
        self.W -= lr * self.dW
        self.b -= lr * self.db


class ReLU:
    """
    ReLU activation: output = max(0, input)
    """

    def forward(self, Z):
        """
        Forward pass: apply ReLU element-wise.
        """
        self.A = torch.maximum(Z, torch.zeros_like(Z))
        return self.A

    def backward(self, dA):
        """
        Backward pass: gradient flows through where input > 0.
        """
        # Gradient is 1 where A > 0, else 0
        dZ = dA * (self.A > 0).float()
        return dZ

    def update(self, lr):
        """ReLU has no parameters to update."""
        pass


class CrossEntropyFromLogits:
    """
    Cross-entropy loss from raw logits (includes softmax).
    Implemented from scratch using only basic operations.
    """

    def forward(self, Z, Y):
        """
        Compute cross-entropy loss.
        Z: (batch_size, n_classes) - raw logits
        Y: (batch_size,) - true class indices
        """
        self.Y = Y
        
        # Manual softmax: subtract max for numerical stability
        Z_shifted = Z - torch.max(Z, dim=1, keepdim=True)[0]
        exp_Z = torch.exp(Z_shifted)
        sum_exp_Z = torch.sum(exp_Z, dim=1, keepdim=True)
        self.A = exp_Z / sum_exp_Z
        
        # Manual log-softmax
        log_sum_exp = torch.log(sum_exp_Z)
        log_softmax_Z = Z_shifted - log_sum_exp
        
        # Select log-probabilities of correct classes
        log_probs = log_softmax_Z[range(len(Y)), Y]
        
        # Cross-entropy loss: average negative log-likelihood
        loss = -torch.mean(log_probs)
        
        return loss

    def backward(self, n_classes):
        """
        Compute gradient: dZ = (A - Y_one_hot) / batch_size
        """
        batch_size = len(self.Y)
        
        # One-hot encode true labels manually
        Y_one_hot = torch.zeros_like(self.A)
        Y_one_hot[range(batch_size), self.Y] = 1
        
        # Gradient: softmax output minus one-hot labels
        dZ = (self.A - Y_one_hot) / batch_size
        
        return dZ


class BatchNorm1D:
    """
    Batch Normalization for 1D inputs (batch, features).
    Normalizes activations to have mean=0, variance=1.
    """

    def __init__(self, n_features, eps=1e-5, momentum=0.1, device="cpu"):
        self.eps = eps
        self.momentum = momentum
        self.device = device

        # Learnable parameters
        self.gamma = torch.ones(n_features, device=device)
        self.beta = torch.zeros(n_features, device=device)

        # Running statistics for inference
        self.running_mean = torch.zeros(n_features, device=device)
        self.running_var = torch.ones(n_features, device=device)

        self.training = True

    def train(self):
        self.training = True
        return self

    def eval(self):
        self.training = False
        return self

    def forward(self, X):
        """
        Forward pass with batch normalization.
        """
        if self.training:
            # Compute batch statistics
            self.batch_mean = torch.mean(X, dim=0)
            self.batch_var = torch.var(X, dim=0, unbiased=False)
            
            # Normalize
            self.std = torch.sqrt(self.batch_var + self.eps)
            self.X_hat = (X - self.batch_mean) / self.std
            
            # Update running statistics (exponential moving average)
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * self.batch_mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * self.batch_var
        else:
            # Use running statistics for inference
            self.std = torch.sqrt(self.running_var + self.eps)
            self.X_hat = (X - self.running_mean) / self.std

        # Store input for backward
        self.X = X

        # Scale and shift
        Y = self.gamma * self.X_hat + self.beta
        return Y

    def backward(self, dY):
        """
        Backward pass through batch normalization.
        """
        if not self.training:
            raise RuntimeError("Backward called in eval() mode.")

        m = dY.size(0)  # batch size

        # Gradient w.r.t. gamma and beta
        self.dbeta = torch.sum(dY, dim=0)
        self.dgamma = torch.sum(dY * self.X_hat, dim=0)

        # Gradient w.r.t. normalized input
        dx_hat = dY * self.gamma

        # Backprop through normalization
        x_mu = self.X - self.batch_mean
        invstd = 1.0 / self.std

        dvar = torch.sum(dx_hat * x_mu * -0.5 * (invstd ** 3), dim=0)
        dmean = torch.sum(-dx_hat * invstd, dim=0) + dvar * torch.mean(-2.0 * x_mu, dim=0)

        dX = dx_hat * invstd + (2.0 / m) * x_mu * dvar + dmean / m

        return dX

    def update(self, lr):
        """Update gamma and beta."""
        self.gamma -= lr * self.dgamma
        self.beta -= lr * self.dbeta


class Dropout:
    """
    Inverted Dropout: randomly zeros activations during training.
    """

    def __init__(self, p=0.5, device="cpu"):
        """
        Args:
            p: Drop probability (fraction of units to zero out)
        """
        assert 0.0 <= p < 1.0, "p must be in [0, 1)."
        self.p = p
        self.device = device
        self.training = True
        self.mask = None

    def train(self):
        self.training = True
        return self

    def eval(self):
        self.training = False
        return self

    def forward(self, X):
        """
        Forward pass with inverted dropout.
        """
        if self.training and self.p > 0.0:
            # Compute keep probability
            keep_prob = 1.0 - self.p
            
            # Sample Bernoulli mask
            self.mask = (torch.rand_like(X) < keep_prob).float()
            
            # Scale by 1/keep_prob (inverted dropout)
            self.mask /= keep_prob
            
            # Apply mask
            return X * self.mask
        else:
            # No dropout in eval mode
            self.mask = torch.ones_like(X)
            return X

    def backward(self, dY):
        """
        Backward pass: gradient flows through the same mask.
        """
        return dY * self.mask

    def update(self, lr):
        """No parameters to update."""
        pass


# Utility function for accuracy calculation
def accuracy(predictions, labels):
    """
    Calculate classification accuracy.
    """
    _, predicted_classes = torch.max(predictions, 1)
    correct = (predicted_classes == labels).sum().item()
    total = labels.size(0)
    return correct / total
