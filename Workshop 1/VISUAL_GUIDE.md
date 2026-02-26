# 🎨 MiniTorch Visual Guide

## Neural Network Architecture Diagram

```
INPUT (28×28 image = 784 pixels)
    ↓
┌─────────────────────────────────┐
│   Linear Layer (784 → 256)      │  ← Weights & Biases
│   Z = X @ W + b                  │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│   Batch Normalization            │  ← Normalize activations
│   (mean=0, variance=1)           │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│   ReLU Activation                │  ← Add non-linearity
│   output = max(0, input)         │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│   Dropout (p=0.2)                │  ← Regularization
│   Randomly zero 20% of neurons   │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│   Linear Layer (256 → 10)        │  ← Output layer
│   Z = X @ W + b                  │
└─────────────────────────────────┘
    ↓
OUTPUT (10 class scores)
    ↓
┌─────────────────────────────────┐
│   Softmax + Cross-Entropy        │  ← Loss calculation
│   Convert to probabilities       │
└─────────────────────────────────┘
    ↓
LOSS (single number)
```

## Forward and Backward Pass Flow

```
FORWARD PASS (Making Predictions)
═══════════════════════════════════

Input Image (784 pixels)
    │
    ├─→ Linear Layer
    │   • Multiply by weights
    │   • Add bias
    │   • Output: 256 numbers
    │
    ├─→ BatchNorm
    │   • Normalize to mean=0, var=1
    │   • Scale and shift
    │
    ├─→ ReLU
    │   • Keep positive, zero negative
    │
    ├─→ Dropout
    │   • Randomly zero some neurons
    │
    ├─→ Linear Layer
    │   • Output: 10 numbers (one per class)
    │
    └─→ Loss Function
        • Convert to probabilities (softmax)
        • Calculate error (cross-entropy)
        • Output: Loss (how wrong we are)


BACKWARD PASS (Learning from Mistakes)
═══════════════════════════════════════

Loss (error value)
    │
    ├─→ Loss Gradient
    │   • dZ = probabilities - true_labels
    │
    ├─→ Linear Layer Backward
    │   • Calculate dW, db (how to change weights)
    │   • Pass gradient to previous layer
    │
    ├─→ Dropout Backward
    │   • Gradient flows through same mask
    │
    ├─→ ReLU Backward
    │   • Gradient flows where input > 0
    │
    ├─→ BatchNorm Backward
    │   • Calculate gradients for gamma, beta
    │   • Pass gradient to previous layer
    │
    └─→ Linear Layer Backward
        • Calculate dW, db
        • (No need to pass further back)


UPDATE (Adjusting Weights)
═══════════════════════════

For each layer with parameters:
    W_new = W_old - learning_rate × dW
    b_new = b_old - learning_rate × db
```

## Matrix Dimensions Flow

```
MNIST Example (batch_size = 64)

Input:  [64, 784]  (64 images, 784 pixels each)
           ↓
Linear: [64, 784] @ [784, 256] + [256]
           ↓
Output: [64, 256]  (64 samples, 256 features)
           ↓
BatchNorm: [64, 256] → [64, 256]
           ↓
ReLU:   [64, 256] → [64, 256]
           ↓
Dropout: [64, 256] → [64, 256]
           ↓
Linear: [64, 256] @ [256, 10] + [10]
           ↓
Output: [64, 10]  (64 samples, 10 class scores)
           ↓
Softmax: [64, 10] → [64, 10]  (probabilities)
           ↓
Loss:   [64, 10] + [64] labels → scalar
```

## Gradient Flow (Backward Pass)

```
Loss (scalar)
    ↓
dZ: [64, 10]  ← Gradient of loss w.r.t. output
    ↓
Linear Layer:
    dW: [256, 10]  ← Gradient w.r.t. weights
    db: [10]       ← Gradient w.r.t. bias
    dX: [64, 256]  ← Gradient to pass back
    ↓
Dropout:
    dX: [64, 256]  ← Gradient through mask
    ↓
ReLU:
    dX: [64, 256]  ← Gradient where input > 0
    ↓
BatchNorm:
    dgamma: [256]  ← Gradient w.r.t. scale
    dbeta:  [256]  ← Gradient w.r.t. shift
    dX: [64, 256]  ← Gradient to pass back
    ↓
Linear Layer:
    dW: [784, 256] ← Gradient w.r.t. weights
    db: [256]      ← Gradient w.r.t. bias
    dX: [64, 784]  ← (Not needed, we're at input)
```

## Training Loop Visualization

```
EPOCH 1
═══════════════════════════════════════════════

Batch 1:  [Forward] → [Loss: 2.30] → [Backward] → [Update]
Batch 2:  [Forward] → [Loss: 2.15] → [Backward] → [Update]
Batch 3:  [Forward] → [Loss: 1.98] → [Backward] → [Update]
...
Batch N:  [Forward] → [Loss: 0.85] → [Backward] → [Update]

Validation: [Forward only] → [Loss: 0.92, Acc: 72%]

EPOCH 2
═══════════════════════════════════════════════

Batch 1:  [Forward] → [Loss: 0.78] → [Backward] → [Update]
Batch 2:  [Forward] → [Loss: 0.71] → [Backward] → [Update]
...

Validation: [Forward only] → [Loss: 0.45, Acc: 87%]

...

EPOCH 10
═══════════════════════════════════════════════

Validation: [Forward only] → [Loss: 0.12, Acc: 97%]

✅ Training Complete!
```

## How Gradient Descent Works

```
Imagine you're on a hill and want to reach the valley (minimum loss):

Current Position (Loss = 2.5)
        🏔️
       /  \
      /    \
     /      \
    /        \
   /          \
  /            \
 /              \
🚶 ← You are here

Step 1: Calculate gradient (which way is down?)
        ↓ (gradient points down)

Step 2: Take a step in that direction
        learning_rate controls step size

After Update (Loss = 2.1)
        🏔️
       /  \
      /    \
     /      \
    /   🚶   \  ← You moved down!
   /          \
  /            \
 /              \

Repeat many times...

Final Position (Loss = 0.1)
        🏔️
       /  \
      /    \
     /      \
    /        \
   /          \
  /            \
 /      🚶      \  ← Reached the valley!
```

## Batch Normalization Effect

```
WITHOUT BATCH NORMALIZATION:
Layer 1 output: [-100, 50, 200, -80, ...]  ← Unstable!
Layer 2 output: [-1000, 500, 2000, ...]    ← Getting worse!
Layer 3 output: [NaN, NaN, NaN, ...]       ← Exploded! 💥

WITH BATCH NORMALIZATION:
Layer 1 output: [-1.2, 0.5, 2.0, -0.8, ...]  ← Normalized!
Layer 2 output: [-0.9, 0.3, 1.5, -0.6, ...]  ← Still stable!
Layer 3 output: [-1.1, 0.4, 1.8, -0.7, ...]  ← Working! ✅
```

## Dropout Visualization

```
TRAINING MODE (Dropout p=0.5):

Before Dropout:
[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

Random Mask (50% kept):
[1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]

After Dropout (scaled by 1/0.5 = 2):
[2.0, 0.0, 6.0, 0.0, 10.0, 0.0, 14.0, 0.0]
 ✓    ✗    ✓    ✗     ✓     ✗     ✓     ✗

INFERENCE MODE:
[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
(No dropout, all neurons active)
```

## Loss Curve (What Success Looks Like)

```
Loss
 │
3│ ●
 │  ●
2│   ●●
 │     ●●
1│       ●●●
 │          ●●●●
0│              ●●●●●●●●●●
 └─────────────────────────────→ Epochs
  1  2  3  4  5  6  7  8  9  10

● = Training Loss
○ = Validation Loss

Good signs:
✅ Both losses decrease
✅ Validation follows training closely
✅ Smooth curve

Bad signs:
❌ Loss increases
❌ Validation much higher than training (overfitting)
❌ Erratic jumps
```

## Accuracy Curve (What Success Looks Like)

```
Accuracy (%)
 │
100│                    ●●●●●●●●
 90│              ●●●●●●
 80│         ●●●●●
 70│     ●●●●
 60│  ●●●
 50│ ●
  0└─────────────────────────────→ Epochs
    1  2  3  4  5  6  7  8  9  10

● = Training Accuracy
○ = Validation Accuracy

Target for MNIST:
✅ Training: 98-99%
✅ Validation: 97-98%
```

## Memory Flow in a Layer

```
LINEAR LAYER MEMORY:

Stored during Forward (for Backward):
┌──────────────────────────────┐
│ self.X = input               │  ← Need for dW calculation
└──────────────────────────────┘

Calculated during Backward:
┌──────────────────────────────┐
│ self.dW = X.T @ dZ / m       │  ← Gradient for weights
│ self.db = sum(dZ) / m        │  ← Gradient for bias
│ self.dX = dZ @ W.T           │  ← Gradient to pass back
└──────────────────────────────┘

Used during Update:
┌──────────────────────────────┐
│ W = W - lr × dW              │  ← Update weights
│ b = b - lr × db              │  ← Update bias
└──────────────────────────────┘
```

## Complete Example: One Training Step

```
INPUT: Image of digit "3"
[0.1, 0.2, ..., 0.9]  (784 pixels)

FORWARD:
Linear1:  [784] → [256]
BatchNorm: normalize
ReLU:     keep positive
Dropout:  random mask
Linear2:  [256] → [10]

OUTPUT: [0.1, 0.05, 0.08, 0.7, 0.02, 0.01, 0.01, 0.02, 0.01, 0.0]
         0    1     2     3    4     5     6     7     8     9
                          ↑
                    Predicted: 3 ✅

LOSS: -log(0.7) = 0.36  (pretty good!)

BACKWARD:
Calculate how to adjust all weights to make 0.7 → 1.0

UPDATE:
Adjust weights slightly in the right direction

NEXT IMAGE: Repeat!
```

## Summary: The Big Picture

```
┌─────────────────────────────────────────────────┐
│                                                 │
│  INPUT → LAYERS → OUTPUT → LOSS                │
│           ↑                    ↓                │
│           │                    │                │
│           └──── GRADIENTS ─────┘                │
│                                                 │
│  Repeat thousands of times...                  │
│  Network gradually learns patterns!            │
│                                                 │
└─────────────────────────────────────────────────┘
```

---

## 🎯 Key Takeaways

1. **Forward Pass**: Data flows through layers to make predictions
2. **Loss**: Measures how wrong predictions are
3. **Backward Pass**: Calculates how to improve (gradients)
4. **Update**: Adjusts weights to reduce error
5. **Repeat**: Network learns through repetition

## 💡 Remember

- Shapes matter! Always check dimensions
- Gradients flow backward through the same path
- Learning rate controls how big each step is
- Batch size affects gradient averaging

Good luck with your workshop! 🚀
