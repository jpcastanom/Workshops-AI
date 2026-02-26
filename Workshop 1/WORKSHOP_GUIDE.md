# MiniTorch Workshop - Complete Guide

## 🎯 What You're Building

You're building a **mini deep learning framework from scratch** to understand how neural networks really work!

## 📖 Key Concepts Explained Simply

### 1. **Linear Layer (Fully Connected Layer)**
Think of it like a weighted sum:
- **Input**: Numbers representing an image (784 pixels for MNIST)
- **Weights (W)**: How important each input is
- **Bias (b)**: A constant offset
- **Output**: `Z = X @ W + b` (matrix multiplication + bias)

**Forward**: Calculate output
**Backward**: Calculate how to adjust W and b to reduce error
**Update**: Actually adjust W and b using gradients

### 2. **ReLU Activation**
Makes the network non-linear (able to learn complex patterns):
- **Forward**: `output = max(0, input)` - Keep positive, zero out negative
- **Backward**: Pass gradient through where input was positive

### 3. **Cross-Entropy Loss**
Measures how wrong your predictions are:
- **Softmax**: Converts raw scores to probabilities (sum to 1)
- **Cross-Entropy**: Penalizes wrong predictions heavily

### 4. **Batch Normalization**
Normalizes activations to stabilize training:
- Makes training faster and more stable
- Keeps running statistics for inference

### 5. **Dropout**
Randomly turns off neurons during training:
- Prevents overfitting (memorizing training data)
- Makes the network more robust

## 🔧 Implementation Steps

### Step 1: Linear Layer
```python
# Forward: Z = X @ W + b
Z = torch.matmul(X, self.W) + self.b

# Backward: Calculate gradients
self.dW = torch.matmul(X.T, dZ) / batch_size
self.db = torch.sum(dZ, dim=0) / batch_size
self.dX = torch.matmul(dZ, self.W.T)

# Update: Apply gradient descent
self.W -= lr * self.dW
self.b -= lr * self.db
```

### Step 2: Cross-Entropy Loss
```python
# Forward: Softmax + Log + NLL
self.A = torch.softmax(Z, dim=1)  # Probabilities
log_probs = torch.log(self.A[range(len(Y)), Y])  # Log of correct class
loss = -torch.mean(log_probs)  # Average negative log-likelihood

# Backward: Elegant formula!
Y_one_hot = torch.zeros_like(Z)
Y_one_hot[range(len(Y)), Y] = 1
dZ = (self.A - Y_one_hot) / batch_size
```

### Step 3: ReLU
```python
# Forward
self.A = torch.maximum(Z, torch.zeros_like(Z))

# Backward
dZ = dA * (self.A > 0).float()
```

### Step 4: Training Loop
```python
# 1. Forward pass
Z = net.forward(X)
loss = CELoss.forward(Z, Y)

# 2. Backward pass
dZ = CELoss.backward(n_classes)
net.backward(dZ)

# 3. Update parameters
net.update(learning_rate)
```

## 💡 Tips for Success

1. **Start Simple**: Get Linear + Loss working first
2. **Test Each Component**: Print shapes to debug
3. **Understand Dimensions**: 
   - X: (batch_size, features)
   - W: (features_in, features_out)
   - Z: (batch_size, features_out)

4. **Common Mistakes**:
   - Forgetting to divide by batch_size in gradients
   - Wrong matrix multiplication order
   - Not storing values needed for backward pass

## 🎓 Learning Outcomes

After this workshop, you'll understand:
- ✅ How neural networks compute predictions (forward pass)
- ✅ How they learn from mistakes (backward pass/backpropagation)
- ✅ Why we need activation functions (non-linearity)
- ✅ How regularization prevents overfitting (Dropout, BatchNorm)
- ✅ The math behind gradient descent

## 🚀 Next Steps

1. Complete the TODOs in the notebook
2. Train your network and see it learn!
3. Experiment with different architectures
4. Package your code into `minitorch.py`
5. Use it for the Kaggle competition!

Good luck! 🎉
