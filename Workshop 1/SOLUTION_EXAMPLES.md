# MiniTorch Workshop - Solution Examples

## This file shows you how to fill in the TODOs in your workshop notebook

### 1. Linear Layer Implementation

```python
class Linear:
    def forward(self, X):
        self.X = X  # Store for backward pass
        # TODO: Implement Z = XW + b
        Z = torch.matmul(X, self.W) + self.b
        return Z

    def backward(self, dZ):
        # TODO: Compute self.dW, self.db, and self.dX
        batch_size = self.X.size(0)
        self.dW = torch.matmul(self.X.T, dZ) / batch_size
        self.db = torch.sum(dZ, dim=0) / batch_size
        self.dX = torch.matmul(dZ, self.W.T)
        return self.dX

    def update(self, lr):
        # TODO: Update W and b using self.dW and self.db
        self.W -= lr * self.dW
        self.b -= lr * self.db
```

### 2. CrossEntropyFromLogits Implementation

```python
class CrossEntropyFromLogits:
    def forward(self, Z, Y):
        self.Y = Y
        
        # TODO: Manual softmax (subtract max for stability)
        Z_shifted = Z - torch.max(Z, dim=1, keepdim=True)[0]
        exp_Z = torch.exp(Z_shifted)
        sum_exp_Z = torch.sum(exp_Z, dim=1, keepdim=True)
        self.A = exp_Z / sum_exp_Z
        
        # TODO: Manual log-softmax
        log_sum_exp = torch.log(sum_exp_Z)
        log_softmax_Z = Z_shifted - log_sum_exp
        
        # TODO: Select log-probabilities of correct classes
        log_probs = log_softmax_Z[range(len(Y)), Y]
        
        # TODO: Cross-entropy loss
        loss = -torch.mean(log_probs)
        
        return loss

    def backward(self, n_classes):
        batch_size = len(self.Y)
        
        # TODO: One-hot encode the true labels
        Y_one_hot = torch.zeros_like(self.A)
        Y_one_hot[range(batch_size), self.Y] = 1
        
        # TODO: Derivative of cross-entropy w.r.t logits
        dZ = (self.A - Y_one_hot) / batch_size
        
        return dZ
```

### 3. Training Loop TODOs

```python
for epoch in range(1, num_epochs + 1):
    # TRAIN
    for batch_idx, (images, labels) in enumerate(trainloader):
        X = images.view(images.size(0), -1).to(device)
        Y = labels.to(device)

        # Forward
        Z = net.forward(X)  # TODO
        loss = CELoss.forward(Z, Y)  # TODO

        # Backward + update
        dZ = CELoss.backward(n_classes)  # TODO
        _ = net.backward(dZ)  # TODO
        net.update(learning_rate)  # TODO

        # Stats
        running_loss += loss.item()  # TODO
        
    train_loss = running_loss / len(trainloader)  # TODO
    train_acc = tot_correct / tot_samples  # TODO

    # VALIDATION
    with torch.no_grad():
        for images, labels in valloader:
            X = images.view(images.size(0), -1).to(device)
            Y = labels.to(device)

            Z = net.forward(X)  # TODO
            vloss = CELoss.forward(Z, Y)  # TODO
            val_running_loss += vloss.item()  # TODO
```

### 4. ReLU Implementation

```python
class ReLU:
    def forward(self, Z):
        # TODO: Apply ReLU
        self.A = torch.maximum(Z, torch.zeros_like(Z))
        return self.A

    def backward(self, dA):
        # TODO: Gradient flows where A > 0
        dZ = dA * (self.A > 0).float()
        return dZ
```

### 5. Net Class Implementation

```python
class Net:
    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)  # TODO
        return X

    def backward(self, dZ):
        for layer in reversed(self.layers):
            dZ = layer.backward(dZ)  # TODO
        return dZ
```

### 6. BatchNorm1D Implementation (Key Parts)

```python
class BatchNorm1D:
    def forward(self, X):
        if self.training:
            # TODO: compute batch statistics
            self.batch_mean = torch.mean(X, dim=0)
            self.batch_var = torch.var(X, dim=0, unbiased=False)
            
            # TODO: normalize
            self.std = torch.sqrt(self.batch_var + self.eps)
            self.X_hat = (X - self.batch_mean) / self.std
            
            # TODO: update running stats
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * self.batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * self.batch_var
        else:
            # TODO: use running stats
            self.std = torch.sqrt(self.running_var + self.eps)
            self.X_hat = (X - self.running_mean) / self.std

        self.X = X
        
        # TODO: scale and shift
        Y = self.gamma * self.X_hat + self.beta
        return Y
```

### 7. Dropout Implementation

```python
class Dropout:
    def forward(self, X):
        if self.training and self.p > 0.0:
            # TODO: Inverted dropout
            keep_prob = 1.0 - self.p
            self.mask = (torch.rand_like(X) < keep_prob).float()
            self.mask /= keep_prob  # Scale by 1/keep_prob
            return X * self.mask
        else:
            # TODO: No dropout in eval
            self.mask = torch.ones_like(X)
            return X

    def backward(self, dY):
        # TODO: Backprop through mask
        return dY * self.mask
```

## 🎯 Quick Tips

1. **Matrix Dimensions**: Always check shapes!
   - X: (batch, features_in)
   - W: (features_in, features_out)
   - Z: (batch, features_out)

2. **Gradient Averaging**: Divide by batch_size to average gradients

3. **Store for Backward**: Save inputs/outputs in forward() for use in backward()

4. **Test Incrementally**: Get each layer working before moving to the next

## 🚀 How to Use

1. Copy the implementations above into your notebook
2. Run the training loop
3. Watch your network learn!
4. Experiment with different architectures

Good luck! 🎉
