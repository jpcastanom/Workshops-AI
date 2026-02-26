# Manual Implementations Reference

## 🎯 Allowed Operations

✅ **You CAN use:**
- `torch.matmul()` or `@` - Matrix multiplication
- `torch.max()`, `torch.min()` - Max/min values
- `torch.mean()`, `torch.sum()` - Aggregations
- `torch.exp()`, `torch.log()`, `torch.sqrt()` - Basic math
- `torch.zeros()`, `torch.ones()`, `torch.zeros_like()` - Tensor creation
- Basic operations: `+`, `-`, `*`, `/`, `**`

❌ **You CANNOT use:**
- `torch.softmax()` - Must implement manually
- `torch.log_softmax()` - Must implement manually
- `torch.nn.*` - Any neural network modules
- Autograd features

---

## 📚 Manual Implementations

### 1. Softmax (From Scratch)

**Formula:**
```
softmax(x_i) = exp(x_i) / sum(exp(x_j) for all j)
```

**Problem:** Large values cause overflow!
```python
exp(1000) = inf  # Overflow!
```

**Solution:** Subtract max for numerical stability
```python
# Manual softmax implementation
def manual_softmax(Z):
    # Subtract max for stability (doesn't change result!)
    Z_shifted = Z - torch.max(Z, dim=1, keepdim=True)[0]
    
    # Exponentiate
    exp_Z = torch.exp(Z_shifted)
    
    # Divide by sum
    sum_exp_Z = torch.sum(exp_Z, dim=1, keepdim=True)
    probabilities = exp_Z / sum_exp_Z
    
    return probabilities
```

**Why it works:**
```
softmax(x - c) = exp(x - c) / sum(exp(x - c))
               = exp(x) * exp(-c) / (sum(exp(x)) * exp(-c))
               = exp(x) / sum(exp(x))
               = softmax(x)
```

### 2. Log-Softmax (From Scratch)

**Formula:**
```
log_softmax(x_i) = log(softmax(x_i))
                 = log(exp(x_i) / sum(exp(x_j)))
                 = x_i - log(sum(exp(x_j)))
```

**Implementation:**
```python
# Manual log-softmax
def manual_log_softmax(Z):
    # Subtract max for stability
    Z_shifted = Z - torch.max(Z, dim=1, keepdim=True)[0]
    
    # Compute log(sum(exp(Z)))
    exp_Z = torch.exp(Z_shifted)
    sum_exp_Z = torch.sum(exp_Z, dim=1, keepdim=True)
    log_sum_exp = torch.log(sum_exp_Z)
    
    # log_softmax = Z - log(sum(exp(Z)))
    log_softmax = Z_shifted - log_sum_exp
    
    return log_softmax
```

### 3. Cross-Entropy Loss (Complete)

```python
class CrossEntropyFromLogits:
    def forward(self, Z, Y):
        """
        Z: (batch_size, n_classes) - raw logits
        Y: (batch_size,) - true class indices
        """
        self.Y = Y
        
        # Step 1: Manual softmax
        Z_shifted = Z - torch.max(Z, dim=1, keepdim=True)[0]
        exp_Z = torch.exp(Z_shifted)
        sum_exp_Z = torch.sum(exp_Z, dim=1, keepdim=True)
        self.A = exp_Z / sum_exp_Z  # Probabilities
        
        # Step 2: Manual log-softmax
        log_sum_exp = torch.log(sum_exp_Z)
        log_softmax_Z = Z_shifted - log_sum_exp
        
        # Step 3: Select log-probs of correct classes
        log_probs = log_softmax_Z[range(len(Y)), Y]
        
        # Step 4: Average negative log-likelihood
        loss = -torch.mean(log_probs)
        
        return loss
    
    def backward(self, n_classes):
        """Gradient: softmax - one_hot"""
        batch_size = len(self.Y)
        
        # One-hot encode manually
        Y_one_hot = torch.zeros_like(self.A)
        Y_one_hot[range(batch_size), self.Y] = 1
        
        # Gradient
        dZ = (self.A - Y_one_hot) / batch_size
        
        return dZ
```

---

## 🔍 Step-by-Step Example

### Input:
```python
Z = torch.tensor([[2.0, 1.0, 0.1]])  # Logits for 1 sample, 3 classes
Y = torch.tensor([0])                 # True class is 0
```

### Step 1: Subtract max
```python
max_Z = torch.max(Z, dim=1, keepdim=True)[0]  # [2.0]
Z_shifted = Z - max_Z  # [[0.0, -1.0, -1.9]]
```

### Step 2: Exponentiate
```python
exp_Z = torch.exp(Z_shifted)  # [[1.0, 0.368, 0.150]]
```

### Step 3: Sum
```python
sum_exp_Z = torch.sum(exp_Z, dim=1, keepdim=True)  # [[1.518]]
```

### Step 4: Softmax
```python
probabilities = exp_Z / sum_exp_Z  # [[0.659, 0.242, 0.099]]
```

### Step 5: Log-softmax
```python
log_sum_exp = torch.log(sum_exp_Z)  # [[0.417]]
log_softmax = Z_shifted - log_sum_exp  # [[-0.417, -1.417, -2.317]]
```

### Step 6: Select correct class
```python
log_prob = log_softmax[0, 0]  # -0.417 (class 0)
```

### Step 7: Loss
```python
loss = -log_prob  # 0.417
```

### Step 8: Gradient (backward)
```python
Y_one_hot = [[1.0, 0.0, 0.0]]
gradient = probabilities - Y_one_hot  # [[-0.341, 0.242, 0.099]]
```

---

## 💡 Why Numerical Stability Matters

### Without max subtraction:
```python
Z = [1000, 999, 998]
exp(1000) = inf  # Overflow!
```

### With max subtraction:
```python
Z = [1000, 999, 998]
Z_shifted = [0, -1, -2]
exp(0) = 1.0  # No overflow!
```

**Result is mathematically identical but numerically stable!**

---

## 🎯 Common Mistakes

### ❌ Mistake 1: Forgetting keepdim
```python
# Wrong - shape mismatch
max_Z = torch.max(Z, dim=1)  # Shape: (batch_size,)
Z_shifted = Z - max_Z  # Broadcasting error!

# Correct
max_Z = torch.max(Z, dim=1, keepdim=True)[0]  # Shape: (batch_size, 1)
Z_shifted = Z - max_Z  # Works!
```

### ❌ Mistake 2: Using torch.softmax
```python
# Not allowed in this workshop!
probs = torch.softmax(Z, dim=1)

# Use manual implementation instead
Z_shifted = Z - torch.max(Z, dim=1, keepdim=True)[0]
exp_Z = torch.exp(Z_shifted)
probs = exp_Z / torch.sum(exp_Z, dim=1, keepdim=True)
```

### ❌ Mistake 3: Wrong dimension
```python
# Wrong - summing over wrong dimension
sum_exp_Z = torch.sum(exp_Z, dim=0)  # Sums over batch!

# Correct - sum over classes
sum_exp_Z = torch.sum(exp_Z, dim=1, keepdim=True)
```

---

## 🧪 Testing Your Implementation

```python
# Test softmax
Z = torch.randn(5, 10)  # 5 samples, 10 classes

# Your implementation
Z_shifted = Z - torch.max(Z, dim=1, keepdim=True)[0]
exp_Z = torch.exp(Z_shifted)
manual_softmax = exp_Z / torch.sum(exp_Z, dim=1, keepdim=True)

# Check: probabilities should sum to 1
print(torch.sum(manual_softmax, dim=1))  # Should be all 1.0

# Check: all values should be between 0 and 1
print(torch.min(manual_softmax), torch.max(manual_softmax))  # Between 0 and 1
```

---

## 📊 Dimension Reference

```
Input Z:        (batch_size, n_classes)
max(Z):         (batch_size, 1)         ← keepdim=True!
Z_shifted:      (batch_size, n_classes)
exp_Z:          (batch_size, n_classes)
sum_exp_Z:      (batch_size, 1)         ← keepdim=True!
softmax:        (batch_size, n_classes)
log_softmax:    (batch_size, n_classes)
Y:              (batch_size,)
Y_one_hot:      (batch_size, n_classes)
dZ:             (batch_size, n_classes)
```

---

## ✅ Complete Working Example

```python
import torch

# Sample data
Z = torch.tensor([[2.0, 1.0, 0.1],
                  [0.5, 2.5, 1.0]])  # 2 samples, 3 classes
Y = torch.tensor([0, 1])              # True classes

# Forward pass
Z_shifted = Z - torch.max(Z, dim=1, keepdim=True)[0]
exp_Z = torch.exp(Z_shifted)
sum_exp_Z = torch.sum(exp_Z, dim=1, keepdim=True)
A = exp_Z / sum_exp_Z

log_sum_exp = torch.log(sum_exp_Z)
log_softmax_Z = Z_shifted - log_sum_exp
log_probs = log_softmax_Z[range(len(Y)), Y]
loss = -torch.mean(log_probs)

print(f"Probabilities:\n{A}")
print(f"Loss: {loss.item():.4f}")

# Backward pass
Y_one_hot = torch.zeros_like(A)
Y_one_hot[range(len(Y)), Y] = 1
dZ = (A - Y_one_hot) / len(Y)

print(f"Gradient:\n{dZ}")
```

---

## 🎓 Summary

**Key Points:**
1. ✅ Implement softmax manually: `exp(Z - max) / sum(exp(Z - max))`
2. ✅ Always subtract max for numerical stability
3. ✅ Use `keepdim=True` to maintain dimensions
4. ✅ Log-softmax: `Z - log(sum(exp(Z)))`
5. ✅ Gradient is elegant: `softmax - one_hot`

**Remember:** The goal is to understand how these functions work, not just use them! 🧠
