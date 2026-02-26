# MiniTorch Workshop - Step-by-Step Tutorial

## 🎓 Understanding Neural Networks from Scratch

This tutorial will help you understand each component of the workshop.

---

## Part 1: The Linear Layer (Fully Connected Layer)

### What does it do?
Transforms input data using weights and biases:
```
Output = Input × Weights + Bias
```

### Example with Numbers:
```
Input X = [1, 2, 3]  (3 features)
Weights W = [[0.5, 0.2],
             [0.3, 0.4],
             [0.1, 0.6]]  (3×2 matrix)
Bias b = [0.1, 0.2]

Output Z = X @ W + b
         = [1×0.5 + 2×0.3 + 3×0.1, 1×0.2 + 2×0.4 + 3×0.6] + [0.1, 0.2]
         = [1.4, 3.0] + [0.1, 0.2]
         = [1.5, 3.2]
```

### Forward Pass (Easy!):
```python
Z = torch.matmul(X, self.W) + self.b
```

### Backward Pass (The Magic):
When training, we need to know:
1. How to adjust W to reduce error → `dW`
2. How to adjust b to reduce error → `db`
3. How error flows back to previous layer → `dX`

```python
# If we have gradient dZ from the next layer:
self.dW = X.T @ dZ / batch_size  # How to change W
self.db = sum(dZ) / batch_size   # How to change b
self.dX = dZ @ W.T               # Pass gradient back
```

### Update (Learning!):
```python
W = W - learning_rate × dW  # Move W in direction that reduces error
b = b - learning_rate × db  # Move b in direction that reduces error
```

---

## Part 2: ReLU Activation

### Why do we need it?
Without activation functions, stacking multiple linear layers is still just one big linear transformation. ReLU adds **non-linearity** so the network can learn complex patterns.

### What does it do?
```
ReLU(x) = max(0, x)
```
- Positive numbers stay the same
- Negative numbers become zero

### Example:
```
Input:  [-2, -1, 0, 1, 2]
Output: [ 0,  0, 0, 1, 2]
```

### Forward Pass:
```python
output = torch.maximum(input, 0)
```

### Backward Pass:
Gradient flows through where input was positive:
```python
gradient_out = gradient_in × (input > 0)
```

---

## Part 3: Cross-Entropy Loss

### What is it?
Measures how wrong your predictions are. Lower is better!

### Steps:
1. **Softmax**: Convert raw scores to probabilities
   ```
   Scores: [2.0, 1.0, 0.1]
   
   Step 1: Subtract max for stability
   Shifted: [0.0, -1.0, -1.9]
   
   Step 2: Exponentiate
   Exp: [1.0, 0.368, 0.150]
   
   Step 3: Divide by sum
   Sum = 1.518
   Probabilities: [0.66, 0.24, 0.10]  (sum = 1.0)
   ```

2. **Cross-Entropy**: Penalize wrong predictions
   ```
   If true class is 0 (first class):
   Loss = -log(0.66) = 0.41
   
   If true class was 2 (third class):
   Loss = -log(0.10) = 2.30  (much worse!)
   ```

### Forward Pass (Manual Implementation):
```python
# Manual softmax for numerical stability
Z_shifted = Z - torch.max(Z, dim=1, keepdim=True)[0]
exp_Z = torch.exp(Z_shifted)
sum_exp_Z = torch.sum(exp_Z, dim=1, keepdim=True)
probabilities = exp_Z / sum_exp_Z

# Manual log-softmax
log_sum_exp = torch.log(sum_exp_Z)
log_softmax = Z_shifted - log_sum_exp

# Calculate loss
log_probs = log_softmax[range(len(labels)), labels]
loss = -torch.mean(log_probs)
```

### Why subtract max?
For numerical stability! Large exponentials can overflow:
```
Bad:  exp(1000) = inf (overflow!)
Good: exp(1000 - 1000) = exp(0) = 1.0
```

### Backward Pass (Beautiful Formula!):
```python
# One-hot encode labels: [0, 1, 0, 0, ...] for class 1
Y_one_hot = torch.zeros_like(probabilities)
Y_one_hot[range(batch_size), labels] = 1

# Gradient is simply: predictions - true_labels
gradient = (probabilities - Y_one_hot) / batch_size
```

---

## Part 4: Batch Normalization

### Why do we need it?
During training, the distribution of layer inputs keeps changing. This makes training unstable and slow. BatchNorm fixes this!

### What does it do?
Normalizes each batch to have mean=0 and variance=1:

```
1. Calculate mean and variance of the batch
2. Normalize: x_normalized = (x - mean) / sqrt(variance + epsilon)
3. Scale and shift: output = gamma × x_normalized + beta
```

### Training vs Inference:
- **Training**: Use batch statistics, update running averages
- **Inference**: Use stored running statistics (no updates)

### Benefits:
- ✅ Faster training
- ✅ More stable gradients
- ✅ Can use higher learning rates
- ✅ Acts as regularization

---

## Part 5: Dropout

### Why do we need it?
Prevents overfitting (memorizing training data instead of learning patterns).

### What does it do?
Randomly "turns off" neurons during training:

```
Input:  [1.0, 2.0, 3.0, 4.0]
Mask:   [1.0, 0.0, 1.0, 0.0]  (50% dropout)
Output: [1.0, 0.0, 3.0, 0.0]
```

### Inverted Dropout (Important!):
Scale the kept values so the expected output stays the same:

```python
keep_prob = 1 - drop_prob
mask = (random < keep_prob) / keep_prob  # Scale by 1/keep_prob
output = input × mask
```

### Training vs Inference:
- **Training**: Apply dropout
- **Inference**: No dropout (use all neurons)

---

## Part 6: Putting It All Together

### A Complete Network:
```python
net = Net()
net.add(Linear(784, 256))      # Input layer
net.add(BatchNorm1D(256))      # Normalize
net.add(ReLU())                # Non-linearity
net.add(Dropout(0.2))          # Regularization
net.add(Linear(256, 10))       # Output layer
```

### Training Loop:
```python
for epoch in range(num_epochs):
    for images, labels in trainloader:
        # 1. Forward: Make predictions
        predictions = net.forward(images)
        
        # 2. Loss: How wrong are we?
        loss = loss_function.forward(predictions, labels)
        
        # 3. Backward: Calculate gradients
        gradient = loss_function.backward()
        net.backward(gradient)
        
        # 4. Update: Adjust weights to reduce error
        net.update(learning_rate)
```

---

## 🎯 Key Concepts Summary

1. **Forward Pass**: Calculate predictions
   - Data flows through layers: Input → Hidden → Output

2. **Loss**: Measure error
   - How far are predictions from true labels?

3. **Backward Pass**: Calculate gradients
   - How should we change each weight to reduce error?
   - Uses chain rule from calculus

4. **Update**: Adjust weights
   - Move weights in direction that reduces error
   - Learning rate controls step size

5. **Repeat**: Do this many times
   - Network gradually learns patterns in data

---

## 💡 Debugging Tips

### Check Shapes:
```python
print(f"X shape: {X.shape}")  # Should be (batch_size, features)
print(f"W shape: {W.shape}")  # Should be (features_in, features_out)
print(f"Z shape: {Z.shape}")  # Should be (batch_size, features_out)
```

### Check Values:
```python
print(f"Loss: {loss.item()}")  # Should decrease over time
print(f"Accuracy: {accuracy}")  # Should increase over time
```

### Common Errors:
1. **Shape mismatch**: Check matrix dimensions
2. **NaN loss**: Learning rate too high, or forgot to normalize
3. **Loss not decreasing**: Learning rate too low, or bug in backward pass
4. **Overfitting**: Add dropout or reduce model size

---

## 🚀 Next Steps

1. **Understand each component** before moving to the next
2. **Test incrementally**: Get Linear working, then add ReLU, etc.
3. **Experiment**: Try different architectures, learning rates, dropout rates
4. **Visualize**: Plot loss curves to see if training is working
5. **Debug**: Print shapes and values when something goes wrong

Remember: Understanding > Memorizing! 🧠

Good luck with your workshop! 🎉
