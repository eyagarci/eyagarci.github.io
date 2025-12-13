---
title: "Mathematics ML Cheatsheet"
date: 2025-11-27 06:00:00
categories: [Machine-Learning]
tags: [Mathematics, Probability, Statistics]    
image:
  path: /assets/imgs/headers/math.jpeg
---

## Introduction

This cheatsheet covers essential mathematical concepts for succeeding in Machine Learning and Data Science interviews.

## 1. Linear Algebra

Linear algebra is fundamental in ML: it allows representing and manipulating data, understanding linear transformations, and optimizing models.

### Vectors

```python
import numpy as np

# Vector
v = np.array([1, 2, 3])

# L2 norm (Euclidean length)
np.linalg.norm(v)  # √(1² + 2² + 3²) = √14

# L1 norm (Manhattan)
np.linalg.norm(v, ord=1)  # |1| + |2| + |3| = 6

# Dot product
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
np.dot(a, b)  # 1*4 + 2*5 + 3*6 = 32

# Angle between two vectors
cos_theta = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
theta = np.arccos(cos_theta)
```

**Key concepts:**
- **Orthogonality**: two vectors are orthogonal if their dot product = 0
- **Normalization**: divide a vector by its norm to obtain a unit vector
- **Projection**: proj_b(a) = (a·b / ||b||²) * b

### Matrices

```python
# Matrix
A = np.array([[1, 2], [3, 4]])

# Transpose
A.T  # [[1, 3], [2, 4]]

# Matrix multiplication
B = np.array([[5, 6], [7, 8]])
np.matmul(A, B)  # or A @ B

# Determinant
np.linalg.det(A)  # 1*4 - 2*3 = -2

# Inverse (if det ≠ 0)
np.linalg.inv(A)

# Trace (sum of diagonal elements)
np.trace(A)  # 1 + 4 = 5

# Rank (number of linearly independent rows/columns)
np.linalg.matrix_rank(A)
```

**Important properties:**
- A matrix is **invertible** if det(A) ≠ 0
- **(AB)^T = B^T A^T**
- **(AB)^(-1) = B^(-1) A^(-1)**
- **I · A = A** (identity matrix)

### Eigenvalues and Eigenvectors

Eigenvalues λ and eigenvectors v satisfy: **Av = λv**

```python
A = np.array([[4, 2], [1, 3]])

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

print(eigenvalues)    # [5, 2]
print(eigenvectors)   # Columns are the eigenvectors
```

**Applications in ML:**
- **PCA**: uses eigenvectors of the covariance matrix
- **Spectral Clustering**: uses eigenvalues of the graph
- **Neural network stability**: related to eigenvalues

### Matrix Decompositions

#### 1. Singular Value Decomposition (SVD)

**A = U Σ V^T**

```python
A = np.array([[1, 2, 3], [4, 5, 6]])
U, S, Vt = np.linalg.svd(A)

# Reconstruction
A_reconstructed = U @ np.diag(S) @ Vt
```

**Applications:**
- Dimensionality reduction
- Image compression
- Recommendation (matrix factorization)
- PCA

#### 2. QR Decomposition

**A = QR** where Q is orthogonal and R is upper triangular

```python
Q, R = np.linalg.qr(A)
```

**Applications:**
- Solving linear systems
- Computing eigenvalues

#### 3. Cholesky Decomposition

For a symmetric positive definite matrix: **A = LL^T**

```python
A = np.array([[4, 2], [2, 3]])
L = np.linalg.cholesky(A)
```

**Applications:**
- Optimization
- Simulation of correlated random variables

### Advanced Concepts

```python
# Mahalanobis distance
def mahalanobis_distance(x, mean, cov):
    diff = x - mean
    inv_cov = np.linalg.inv(cov)
    return np.sqrt(diff @ inv_cov @ diff.T)

# Covariance matrix
X = np.random.randn(100, 3)  # 100 samples, 3 features
cov_matrix = np.cov(X.T)

# Correlation matrix
corr_matrix = np.corrcoef(X.T)
```

## 2. Differential Calculus

Differential calculus is essential for understanding optimization and back propagation in neural networks.

### Fundamental Derivatives

| Function | Derivative |
|----------|-----------|
| f(x) = x^n | f'(x) = nx^(n-1) |
| f(x) = e^x | f'(x) = e^x |
| f(x) = ln(x) | f'(x) = 1/x |
| f(x) = sin(x) | f'(x) = cos(x) |
| f(x) = cos(x) | f'(x) = -sin(x) |
| f(x) = 1/(1+e^(-x)) | f'(x) = f(x)(1-f(x)) |

### Differentiation Rules

```
# Sum rule
(f + g)' = f' + g'

# Product rule
(f · g)' = f'g + fg'

# Quotient rule
(f/g)' = (f'g - fg') / g²

# Chain rule
(f ∘ g)' = f'(g(x)) · g'(x)
```

**Example - Backpropagation:**
```python
# Composite function: L = (Wx + b)²
# dL/dW = dL/dy · dy/dW   (chain rule)
#       = 2y · x^T

def forward(x, W, b):
    y = W @ x + b
    L = y ** 2
    return L, y

def backward(x, W, y):
    dL_dy = 2 * y
    dy_dW = x.T
    dL_dW = dL_dy @ dy_dW
    return dL_dW
```

### Gradient and Partial Derivatives

For a multivariable function f(x₁, x₂, ..., xₙ):

**Gradient**: ∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]

```python
# Example: f(x, y) = x²y + y³
# ∂f/∂x = 2xy
# ∂f/∂y = x² + 3y²

def gradient_f(x, y):
    df_dx = 2 * x * y
    df_dy = x**2 + 3 * y**2
    return np.array([df_dx, df_dy])
```

**Gradient properties:**
- Points in the direction of steepest ascent
- Perpendicular to level curves
- Used in gradient descent: **θ_new = θ_old - α∇L**

### Hessian Matrix

The Hessian is the matrix of second derivatives:

**H[i,j] = ∂²f / ∂xᵢ∂xⱼ**

```python
# For f(x, y) = x² + xy + y²
# H = [[2, 1],
#      [1, 2]]

# Nature of critical points:
# - H positive definite (eigenvalues > 0) → local minimum
# - H negative definite (eigenvalues < 0) → local maximum
# - H indefinite → saddle point
```

### Optimization with Gradient

```python
# Gradient descent
def gradient_descent(f, grad_f, x0, learning_rate=0.01, iterations=1000):
    x = x0
    history = [x]
    
    for i in range(iterations):
        gradient = grad_f(x)
        x = x - learning_rate * gradient
        history.append(x)
    
    return x, history

# Example: minimize f(x) = x²
f = lambda x: x**2
grad_f = lambda x: 2*x

x_min, history = gradient_descent(f, grad_f, x0=10.0, learning_rate=0.1)
```

## 3. Probability and Statistics

Probability is at the heart of Machine Learning: modeling uncertainty, parameter estimation, and model evaluation.

### Fundamental Concepts

**Probability**: P(A) ∈ [0, 1]

```python
# Basic rules
P(A ∪ B) = P(A) + P(B) - P(A ∩ B)  # Union
P(A ∩ B) = P(A|B) · P(B)            # Intersection
P(A|B) = P(B|A) · P(A) / P(B)       # Bayes' Theorem
```

**Independence**: A and B are independent if P(A ∩ B) = P(A) · P(B)

### Random Variables

#### Discrete Variables

```python
# Probability mass function (PMF)
# Example: dice roll
outcomes = [1, 2, 3, 4, 5, 6]
pmf = [1/6] * 6

# Expected value
E_X = sum(x * p for x, p in zip(outcomes, pmf))  # 3.5

# Variance
Var_X = sum((x - E_X)**2 * p for x, p in zip(outcomes, pmf))
```

#### Continuous Variables

```python
from scipy import stats

# Probability density function (PDF)
# Example: normal distribution
x = np.linspace(-4, 4, 100)
pdf = stats.norm.pdf(x, loc=0, scale=1)

# Cumulative distribution function (CDF)
cdf = stats.norm.cdf(x, loc=0, scale=1)
```

### Important Distributions

#### 1. Normal Distribution (Gaussian)

**X ~ N(μ, σ²)**

```python
# PDF: f(x) = (1/√(2πσ²)) exp(-(x-μ)²/(2σ²))
mu, sigma = 0, 1
X = np.random.normal(mu, sigma, 1000)

# Properties
# E[X] = μ
# Var(X) = σ²
# 68% of values in [μ-σ, μ+σ]
# 95% of values in [μ-2σ, μ+2σ]
```

**Multivariate distributions:**
```python
# Multivariate normal
mean = [0, 0]
cov = [[1, 0.5], [0.5, 1]]
X = np.random.multivariate_normal(mean, cov, 1000)
```

#### 2. Bernoulli Distribution

**X ~ Bernoulli(p)**

```python
# P(X=1) = p, P(X=0) = 1-p
p = 0.7
X = np.random.binomial(1, p, 1000)

# E[X] = p
# Var(X) = p(1-p)
```

#### 3. Binomial Distribution

**X ~ Binomial(n, p)**

```python
# Number of successes in n trials
n, p = 10, 0.5
X = np.random.binomial(n, p, 1000)

# E[X] = np
# Var(X) = np(1-p)
```

#### 4. Poisson Distribution

**X ~ Poisson(λ)**

```python
# Number of events in an interval
lam = 3.0
X = np.random.poisson(lam, 1000)

# E[X] = λ
# Var(X) = λ
```

#### 5. Uniform Distribution

```python
# Continuous
X = np.random.uniform(0, 1, 1000)

# Discrete
X = np.random.randint(1, 7, 1000)  # Dice
```

#### 6. Exponential Distribution

```python
# Time between events
lam = 1.5
X = np.random.exponential(1/lam, 1000)

# E[X] = 1/λ
# Var(X) = 1/λ²
```

### Bayes' Theorem

**P(H|E) = P(E|H) · P(H) / P(E)**

```python
# Example: medical test
# P(Disease) = 0.01
# P(Test+|Disease) = 0.95  (sensitivity)
# P(Test-|Healthy) = 0.90  (specificity)

P_disease = 0.01
P_test_pos_disease = 0.95
P_test_pos_healthy = 0.10

# P(Test+)
P_test_pos = P_test_pos_disease * P_disease + P_test_pos_healthy * (1 - P_disease)

# P(Disease|Test+)
P_disease_test_pos = (P_test_pos_disease * P_disease) / P_test_pos
print(f"P(Disease|Test+) = {P_disease_test_pos:.2%}")  # ~8.7%
```

**Applications in ML:**
- **Naive Bayes classifier**
- **Bayesian filters**
- **Bayesian inference**

### Descriptive Statistics

```python
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Central tendency
mean = np.mean(data)           # Mean
median = np.median(data)       # Median
mode = stats.mode(data)        # Mode

# Dispersion
variance = np.var(data)        # Variance
std = np.std(data)             # Standard deviation
range_val = np.ptp(data)       # Range (max - min)

# Position
q25 = np.percentile(data, 25)  # 1st quartile
q75 = np.percentile(data, 75)  # 3rd quartile
iqr = q75 - q25                # Interquartile range

# Shape
from scipy.stats import skew, kurtosis
skewness = skew(data)          # Skewness
kurt = kurtosis(data)          # Kurtosis
```

### Correlation and Covariance

```python
X = np.random.randn(100, 2)

# Covariance: measures joint variation
# Cov(X, Y) = E[(X - E[X])(Y - E[Y])]
covariance = np.cov(X[:, 0], X[:, 1])[0, 1]

# Pearson correlation: normalized covariance [-1, 1]
# ρ = Cov(X, Y) / (σ_X · σ_Y)
correlation = np.corrcoef(X[:, 0], X[:, 1])[0, 1]

# Spearman correlation (rank-based)
spearman_corr, p_value = stats.spearmanr(X[:, 0], X[:, 1])
```

**Interpretation:**
- **ρ = 1**: perfect positive linear correlation
- **ρ = -1**: perfect negative linear correlation
- **ρ = 0**: no linear correlation
- **|ρ| < 0.3**: weak, **0.3-0.7**: moderate, **> 0.7**: strong

### Hypothesis Testing

```python
from scipy import stats

# Student's t-test (comparing means)
group1 = np.random.normal(5, 1, 100)
group2 = np.random.normal(5.5, 1, 100)
t_stat, p_value = stats.ttest_ind(group1, group2)

if p_value < 0.05:
    print("Significant difference")

# Chi-square test (independence)
contingency_table = [[10, 20], [30, 40]]
chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

# ANOVA (comparing multiple groups)
group1 = np.random.normal(5, 1, 30)
group2 = np.random.normal(6, 1, 30)
group3 = np.random.normal(5.5, 1, 30)
f_stat, p_value = stats.f_oneway(group1, group2, group3)
```

### Law of Large Numbers and Central Limit Theorem

```python
# Law of large numbers: sample mean converges to expected value
n_samples = [10, 100, 1000, 10000]
true_mean = 5
for n in n_samples:
    sample_mean = np.mean(np.random.normal(true_mean, 1, n))
    print(f"n={n}: mean={sample_mean:.3f}")

# Central limit theorem: distribution of sample mean
# is approximately normal, regardless of original distribution
sample_means = [np.mean(np.random.exponential(1, 100)) for _ in range(1000)]
# sample_means approximately follows N(1, 1²/100)
```

## 4. Optimization

Optimization is at the heart of Machine Learning: finding parameters that minimize the loss function.

### Optimality Conditions

For a local minimum at x*:
1. **First-order necessary condition**: ∇f(x*) = 0
2. **Second-order sufficient condition**: Hessian H(x*) positive definite

```python
# Checking for minimum
def check_minimum(hessian):
    eigenvalues = np.linalg.eigvals(hessian)
    if all(eigenvalues > 0):
        return "Local minimum"
    elif all(eigenvalues < 0):
        return "Local maximum"
    else:
        return "Saddle point"
```

### Convex Functions

A function f is **convex** if:
**f(λx + (1-λ)y) ≤ λf(x) + (1-λ)f(y)** for all λ ∈ [0,1]

**Important properties:**
- Any local minimum is a global minimum
- Sum of convex functions is convex
- Maximum of convex functions is convex

```python
# Examples of convex functions
f1 = lambda x: x**2                    # Convex
f2 = lambda x: np.abs(x)               # Convex
f3 = lambda x: np.exp(x)               # Convex
f4 = lambda x: -np.log(x)              # Convex (x > 0)

# Examples of non-convex functions
f5 = lambda x: x**3                    # Non-convex
f6 = lambda x: np.sin(x)               # Non-convex
```

### Optimization Algorithms

#### 1. Gradient Descent

```python
def gradient_descent(grad_f, x0, lr=0.01, max_iter=1000, tol=1e-6):
    """
    Minimize f by following -∇f
    θ_{t+1} = θ_t - α∇f(θ_t)
    """
    x = x0.copy()
    for i in range(max_iter):
        grad = grad_f(x)
        x_new = x - lr * grad
        
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    
    return x
```

**Advantages:** Simple, guarantees convergence (if lr adapted)
**Disadvantages:** Can be slow, sensitive to learning rate

#### 2. Stochastic Gradient Descent (SGD)

```python
def sgd(grad_f_i, x0, data, lr=0.01, epochs=100, batch_size=32):
    """
    Updates with a mini-batch at each iteration
    Faster but noisy convergence
    """
    x = x0.copy()
    n_samples = len(data)
    
    for epoch in range(epochs):
        # Shuffle data
        indices = np.random.permutation(n_samples)
        
        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            grad = np.mean([grad_f_i(x, data[j]) for j in batch_indices], axis=0)
            x = x - lr * grad
    
    return x
```

#### 3. Momentum

```python
def momentum(grad_f, x0, lr=0.01, beta=0.9, max_iter=1000):
    """
    Accumulates velocity to smooth oscillations
    v_t = βv_{t-1} + ∇f(θ_t)
    θ_{t+1} = θ_t - αv_t
    """
    x = x0.copy()
    v = np.zeros_like(x)
    
    for i in range(max_iter):
        grad = grad_f(x)
        v = beta * v + grad
        x = x - lr * v
    
    return x
```

#### 4. Adam (Adaptive Moment Estimation)

```python
def adam(grad_f, x0, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, max_iter=1000):
    """
    Combines momentum and learning rate adaptation
    """
    x = x0.copy()
    m = np.zeros_like(x)  # 1st moment (mean)
    v = np.zeros_like(x)  # 2nd moment (variance)
    
    for t in range(1, max_iter + 1):
        grad = grad_f(x)
        
        # Update moments
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        
        # Bias correction
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        
        # Update parameters
        x = x - lr * m_hat / (np.sqrt(v_hat) + eps)
    
    return x
```

**Optimizer comparison:**

| Optimizer | Speed | Robustness | Memory | Usage |
|-----------|-------|------------|--------|-------|
| GD | Slow | Stable | Low | Small datasets |
| SGD | Fast | Noisy | Low | Large data |
| Momentum | Fast | Good | Medium | Ravines |
| Adam | Very fast | Excellent | High | Default choice |

#### 5. Newton-Raphson

```python
def newton_method(grad_f, hess_f, x0, max_iter=100, tol=1e-6):
    """
    Uses the Hessian to converge faster
    θ_{t+1} = θ_t - H^{-1}∇f(θ_t)
    """
    x = x0.copy()
    
    for i in range(max_iter):
        grad = grad_f(x)
        hess = hess_f(x)
        
        # Newton direction
        direction = np.linalg.solve(hess, grad)
        x_new = x - direction
        
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    
    return x
```

**Advantages:** Quadratic convergence (very fast)
**Disadvantages:** Expensive (computing Hessian), not always stable

### Constrained Optimization

#### Lagrange Multipliers

To minimize f(x) subject to g(x) = 0:

**L(x, λ) = f(x) + λg(x)**

KKT (Karush-Kuhn-Tucker) conditions:
```python
# ∇_x L = 0
# g(x) = 0
# For inequality constraints h(x) ≤ 0:
# λ ≥ 0
# λh(x) = 0 (complementarity)
```

**Example: SVM**
```python
# Maximize: W(α) = Σα_i - (1/2)ΣΣα_i α_j y_i y_j K(x_i, x_j)
# Subject to:
# 0 ≤ α_i ≤ C
# Σα_i y_i = 0
```

### Regularization

Regularization penalizes model complexity to avoid overfitting.

```python
# L1 regularization (Lasso): promotes sparsity
loss_L1 = mse_loss + lambda_reg * np.sum(np.abs(weights))

# L2 regularization (Ridge): penalizes large weights
loss_L2 = mse_loss + lambda_reg * np.sum(weights ** 2)

# Elastic Net: L1 + L2 combination
loss_elastic = mse_loss + alpha * (
    l1_ratio * np.sum(np.abs(weights)) + 
    (1 - l1_ratio) * np.sum(weights ** 2)
)
```

**Impact of regularization:**
- **L1**: feature selection (some weights → 0)
- **L2**: uniform weight reduction
- **λ large**: underfitting (model too simple)
- **λ small**: overfitting (model too complex)

## 5. Information Theory

Information theory quantifies uncertainty and is fundamental for understanding loss functions in ML.

### Entropy

Entropy measures the uncertainty of a distribution:

**H(X) = -Σ P(x) log P(x)**

```python
def entropy(probs):
    """Calculates Shannon entropy"""
    probs = np.array(probs)
    probs = probs[probs > 0]  # Avoid log(0)
    return -np.sum(probs * np.log2(probs))

# Examples
entropy([0.5, 0.5])        # 1.0 bit (maximum for 2 states)
entropy([1.0, 0.0])        # 0.0 bit (certainty)
entropy([0.25, 0.25, 0.25, 0.25])  # 2.0 bits
```

**Properties:**
- **H(X) ≥ 0**
- **H(X) is maximal** for uniform distribution
- **H(X) = 0** if X is deterministic

### Cross-Entropy

Measures the difference between two distributions:

**H(P, Q) = -Σ P(x) log Q(x)**

```python
def cross_entropy(y_true, y_pred):
    """
    y_true: true distribution (labels)
    y_pred: predicted distribution (probabilities)
    """
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)  # Numerical stability
    return -np.sum(y_true * np.log(y_pred))

# Binary classification
y_true = np.array([1, 0, 1, 1])
y_pred = np.array([0.9, 0.1, 0.8, 0.7])

bce = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
```

**Usage in ML:**
- **Loss function** for classification
- The closer the prediction to truth, the lower the cross-entropy

### Kullback-Leibler Divergence (KL Divergence)

Measures how much Q diverges from P:

**KL(P || Q) = Σ P(x) log(P(x) / Q(x)) = H(P, Q) - H(P)**

```python
def kl_divergence(p, q):
    """
    KL(P || Q): information lost when using Q instead of P
    """
    p = np.array(p)
    q = np.array(q)
    return np.sum(p * np.log(p / q))

# Example
p = np.array([0.5, 0.5])
q = np.array([0.8, 0.2])
kl = kl_divergence(p, q)  # ≈ 0.223
```

**Properties:**
- **KL(P || Q) ≥ 0**
- **KL(P || Q) = 0** iff P = Q
- **Asymmetric**: KL(P || Q) ≠ KL(Q || P)

**Applications:**
- **VAE** (Variational Autoencoders)
- **Model distillation**
- **Distribution optimization**

### Mutual Information

Measures the dependence between two variables:

**I(X; Y) = H(X) + H(Y) - H(X, Y)**

```python
def mutual_information(x, y, bins=10):
    """
    I(X; Y) = 0 if X and Y are independent
    I(X; Y) > 0 if X and Y are dependent
    """
    from sklearn.metrics import mutual_info_score
    
    # Discretize if necessary
    x_binned = np.digitize(x, np.linspace(x.min(), x.max(), bins))
    y_binned = np.digitize(y, np.linspace(y.min(), y.max(), bins))
    
    return mutual_info_score(x_binned, y_binned)
```

**Applications:**
- **Feature selection**
- **Dependency analysis**
- **Transfer learning**

## 6. Essential Formulas for Interviews

### Classification Metrics

```python
# Confusion matrix
#                Predicted +    Predicted -
# Actual +         TP              FN
# Actual -         FP              TN

# Accuracy
accuracy = (TP + TN) / (TP + TN + FP + FN)

# Precision
precision = TP / (TP + FP)  # Among positive predictions, how many are correct?

# Recall / Sensitivity
recall = TP / (TP + FN)  # Among true positives, how many are detected?

# Specificity
specificity = TN / (TN + FP)  # Among true negatives, how many are detected?

# F1-Score (harmonic mean of precision and recall)
f1 = 2 * (precision * recall) / (precision + recall)

# F-beta Score (weight β for recall)
f_beta = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)
```

### Regression Metrics

```python
y_true = np.array([3, 5, 2, 7])
y_pred = np.array([2.5, 5.5, 1.8, 7.2])

# Mean Absolute Error (MAE)
mae = np.mean(np.abs(y_true - y_pred))

# Mean Squared Error (MSE)
mse = np.mean((y_true - y_pred) ** 2)

# Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)

# R² Score (coefficient of determination)
ss_res = np.sum((y_true - y_pred) ** 2)
ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
r2 = 1 - (ss_res / ss_tot)

# Mean Absolute Percentage Error (MAPE)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
```

### Activation Functions

```python
# Sigmoid: [0, 1] - binary classification
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative: σ'(x) = σ(x)(1 - σ(x))
def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# Tanh: [-1, 1] - better than sigmoid (centered at 0)
def tanh(x):
    return np.tanh(x)

# Derivative: tanh'(x) = 1 - tanh²(x)
def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

# ReLU: [0, ∞] - most popular, avoids vanishing gradient
def relu(x):
    return np.maximum(0, x)

# Derivative: 1 if x > 0, 0 otherwise
def relu_derivative(x):
    return (x > 0).astype(float)

# Leaky ReLU: avoids dead neurons
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

# Softmax: multi-class classification (sum = 1)
def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Numerical stability
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
```

### Distances and Similarities

```python
# Euclidean distance (L2)
def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

# Manhattan distance (L1)
def manhattan_distance(x, y):
    return np.sum(np.abs(x - y))

# Minkowski distance (generalization)
def minkowski_distance(x, y, p):
    return np.sum(np.abs(x - y) ** p) ** (1/p)

# Cosine similarity: [-1, 1]
def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

# Cosine distance: [0, 2]
def cosine_distance(x, y):
    return 1 - cosine_similarity(x, y)

# Jaccard distance (sets)
def jaccard_distance(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return 1 - intersection / union
```

### Normalization

```python
X = np.random.randn(100, 5)

# Min-Max Scaling: [0, 1]
X_minmax = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

# Standardization (Z-score): μ=0, σ=1
X_std = (X - X.mean(axis=0)) / X.std(axis=0)

# L2 Normalization (vector normalization)
X_l2 = X / np.linalg.norm(X, axis=1, keepdims=True)

# Robust Scaling (resistant to outliers)
median = np.median(X, axis=0)
iqr = np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)
X_robust = (X - median) / iqr
```

## 7. Key Concepts and Insights

### Covariance vs Correlation

**Covariance** measures how two variables change together, but its value depends on the units of measurement. If you have height in meters vs centimeters, covariance changes dramatically.

**Correlation** (Pearson's r) normalizes covariance by dividing by the product of standard deviations, yielding a unitless measure between -1 and 1:

**ρ = Cov(X,Y) / (σ_X · σ_Y)**

- **ρ = 1**: perfect positive linear relationship
- **ρ = -1**: perfect negative linear relationship  
- **ρ = 0**: no linear relationship

**Key insight**: Correlation is scale-invariant, making it easier to interpret and compare across different datasets.

### Feature Normalization: When and Why

Feature scaling is critical for many ML algorithms:

**When to normalize:**
- Distance-based algorithms (KNN, K-means, SVM) - unscaled features dominate
- Gradient descent optimization - prevents slow convergence
- Neural networks - improves training stability
- Regularization (L1/L2) - ensures fair penalty across features

**Common techniques:**
- **Standardization (Z-score)**: mean=0, std=1 - preserves outliers
- **Min-Max scaling**: range [0,1] - sensitive to outliers
- **Robust scaling**: uses median and IQR - resistant to outliers

**When NOT to normalize:**
- Tree-based models (Random Forest, XGBoost) - scale-invariant
- When feature magnitudes carry meaningful information

### The Bias-Variance Tradeoff

This fundamental concept explains the tradeoff in model complexity:

**Bias** (underfitting):
- Error from incorrect assumptions
- Model too simple to capture patterns
- High training AND test error
- Example: linear regression on non-linear data

**Variance** (overfitting):
- Error from sensitivity to training data fluctuations
- Model too complex, memorizes noise
- Low training error, high test error
- Example: high-degree polynomial on small dataset

**Mathematical formulation:**
**E[(y - ŷ)²] = Bias² + Variance + Irreducible Error**

**Strategies to balance:**
- Cross-validation for model selection
- Regularization (L1/L2) to control complexity
- Ensemble methods (bagging reduces variance, boosting reduces bias)
- More training data generally reduces variance

### Cross-Entropy vs MSE for Classification

**Why cross-entropy is superior for classification:**

1. **Gradient behavior**: Cross-entropy maintains strong gradients even when predictions are confident but wrong. MSE gradients vanish with sigmoid activation.

2. **Probabilistic interpretation**: Directly derived from maximum likelihood estimation (MLE) for Bernoulli/Multinomial distributions.

3. **Mathematical formulation**:
   - **Binary**: BCE = -[y log(p) + (1-y) log(1-p)]
   - **Multi-class**: CCE = -Σ y_i log(p_i)

**When MSE might be acceptable:**
- Linear activation in output layer
- Regression-like probability estimation
- Specific domain requirements

### Positive Definite Matrices

A matrix A is **positive definite** if **x^T A x > 0** for all non-zero vectors x.

**Equivalent conditions:**
- All eigenvalues > 0
- All leading principal minors > 0
- Cholesky decomposition exists

**Why it matters in ML:**

1. **Optimization**: Positive definite Hessian guarantees local minimum
2. **Covariance matrices**: Always positive semi-definite (PSD)
3. **Kernel methods**: Valid kernels must be PSD
4. **Stability**: Ensures numerical stability in many algorithms

**Example**: For f(x,y) = x² + xy + y², the Hessian is:
```
H = [[2, 1],
     [1, 2]]
```
Eigenvalues: 3, 1 (both positive) → local minimum confirmed

### The Central Limit Theorem (CLT)

**Statement**: The distribution of sample means approaches a normal distribution as sample size increases, **regardless of the original distribution's shape**.

**Formally**: If X₁, X₂, ..., Xₙ are i.i.d. with mean μ and variance σ², then:

**√n(X̄ - μ) / σ → N(0, 1)** as n → ∞

**Practical implications:**

1. **Sample size rule**: n ≥ 30 often sufficient for CLT to apply
2. **Hypothesis testing**: Justifies t-tests and z-tests even for non-normal data
3. **Confidence intervals**: Enables construction even without knowing true distribution
4. **A/B testing**: Foundation for comparing group means

**Limitations:**
- Requires i.i.d. samples
- Heavy-tailed distributions need larger n
- Doesn't apply to median or other non-linear statistics

### Why Adam Optimizer Dominates

Adam (Adaptive Moment Estimation) combines the best of multiple optimization techniques:

**Key mechanisms:**

1. **Momentum**: Accumulates exponentially decaying average of past gradients
   - Smooths oscillations
   - Accelerates convergence in ravines

2. **Adaptive learning rates**: Per-parameter learning rate scaling
   - Uses second moment (variance) of gradients
   - Automatically adjusts step sizes

3. **Bias correction**: Corrects initialization bias in moment estimates

**Hyperparameters** (defaults work well):
- Learning rate α: 0.001
- β₁: 0.9 (first moment decay)
- β₂: 0.999 (second moment decay)
- ε: 10⁻⁸ (numerical stability)

**When to consider alternatives:**
- **SGD with momentum**: Better generalization in some cases
- **AdamW**: Adam with decoupled weight decay (better for transformers)
- **RAdam**: Rectified Adam with warm-up

### L1 vs L2 Regularization

Both add penalty terms to loss function, but with different effects:

**L1 Regularization (Lasso)**:
- **Penalty**: λ Σ|w_i|
- **Effect**: Drives weights exactly to zero → sparse solutions
- **Gradient**: Sign(w) - discontinuous at zero
- **Use when**:
  - Feature selection needed
  - Many irrelevant features suspected
  - Interpretability is priority

**L2 Regularization (Ridge)**:
- **Penalty**: λ Σw_i²
- **Effect**: Shrinks weights uniformly → all features contribute
- **Gradient**: 2w - smooth everywhere
- **Use when**:
  - All features potentially relevant
  - Multicollinearity present (stabilizes solution)
  - Computational efficiency matters

**Elastic Net**: Combines both: α[λ₁|w| + λ₂w²]
- Gets benefits of both
- λ₁/λ₂ ratio controls sparsity level

**Geometric interpretation**:
- **L1**: Diamond-shaped constraint region → solutions hit corners (sparsity)
- **L2**: Circular constraint region → solutions spread across dimensions

### Convex Optimization: Why It Matters

A function f is **convex** if the line segment connecting any two points on the graph lies above the graph:

**f(λx + (1-λ)y) ≤ λf(x) + (1-λ)f(y)** for λ ∈ [0,1]

**Critical property**: Any local minimum is a global minimum

**Why this is powerful in ML:**

1. **Guaranteed convergence**: Gradient descent finds global optimum
2. **No hyperparameter tuning**: Learning rate is main concern
3. **Theory-backed**: Well-understood convergence rates

**Convex ML problems:**
- Linear regression (MSE loss)
- Logistic regression (cross-entropy loss)
- Support Vector Machines (hinge loss)
- Lasso and Ridge regression

**Non-convex problems** (harder):
- Neural networks (deep learning)
- Matrix factorization
- Many clustering algorithms

**Testing convexity:**
- Check second derivative: f''(x) ≥ 0
- For multivariate: Hessian must be positive semi-definite
- Composition rules: sum of convex is convex, max of convex is convex

### Chain Rule in Deep Learning

The chain rule enables backpropagation, the foundation of training neural networks.

**Single-variable form**: (f ∘ g)'(x) = f'(g(x)) · g'(x)

**Multi-variable form** (more relevant for ML):
**∂z/∂x = (∂z/∂y)(∂y/∂x)**

**Example in neural network**:
```
Input → Layer 1 → Activation → Layer 2 → Loss
  x   →   z₁    →     a₁     →   z₂    →   L
```

**Backpropagation calculation**:
```
∂L/∂W₁ = ∂L/∂z₂ · ∂z₂/∂a₁ · ∂a₁/∂z₁ · ∂z₁/∂W₁
```

**Computational graph**: Modern frameworks (PyTorch, TensorFlow) build computational graphs and automatically apply chain rule.

**Vanishing/Exploding gradients**: Chain rule multiplication can cause:
- **Vanishing**: Products of small derivatives → very small gradients
- **Exploding**: Products of large derivatives → unstable training

**Solutions**: Proper initialization (Xavier/He), batch normalization, residual connections (ResNets)

### Understanding P-values and Statistical Significance

**P-value definition**: Probability of observing data at least as extreme as yours, assuming the null hypothesis is true.

**Common misconception**: "p-value is probability that null hypothesis is true" - **WRONG**

**Correct interpretation**:
- **p < 0.05**: If null hypothesis were true, such extreme data would occur <5% of the time
- Does NOT prove alternative hypothesis
- Does NOT measure effect size

**Best practices**:
- Report confidence intervals alongside p-values
- Consider effect size (Cohen's d, odds ratio)
- Use multiple testing corrections (Bonferroni, FDR) when testing multiple hypotheses
- Pre-register hypotheses to avoid p-hacking

### Vanishing vs Exploding Gradients

**Problem**: In deep networks, gradients can become extremely small or large during backpropagation.

**Vanishing gradients**:
- **Cause**: Multiplying many small derivatives (e.g., sigmoid derivatives ≤ 0.25)
- **Effect**: Early layers learn very slowly or not at all
- **Solutions**:
  - Use ReLU/Leaky ReLU instead of sigmoid/tanh
  - Batch normalization
  - Residual connections (skip connections)
  - Proper weight initialization (Xavier/He)

**Exploding gradients**:
- **Cause**: Multiplying many large derivatives
- **Effect**: Unstable training, NaN losses, oscillating convergence
- **Solutions**:
  - Gradient clipping (cap gradient norm)
  - Lower learning rate
  - Batch normalization
  - Better weight initialization

**Detection**:
```python
# Monitor gradient norms during training
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.norm().item()}")
```

### Batch Normalization: Why It Works

Batch normalization normalizes layer inputs across mini-batch:

**Algorithm**:
1. Compute mini-batch mean and variance
2. Normalize: x̂ = (x - μ) / √(σ² + ε)
3. Scale and shift: y = γx̂ + β (learnable parameters)

**Benefits**:

1. **Reduces internal covariate shift**: Stabilizes input distributions to layers
2. **Allows higher learning rates**: Makes optimization landscape smoother
3. **Regularization effect**: Adds noise through mini-batch statistics
4. **Reduces sensitivity to initialization**: Makes training more robust

**When to use**:
- Almost always in CNNs (after conv, before activation)
- Before or after activation in fully connected layers
- Not recommended for RNNs (use Layer Normalization instead)

**Considerations**:
- Different behavior in training vs inference (uses running statistics)
- Small batch sizes reduce effectiveness
- Can interact poorly with dropout

## Summary: Formulas important to Know 

```
# Gradient descent
θ_{t+1} = θ_t - α∇L(θ_t)

# Softmax
softmax(x_i) = exp(x_i) / Σ exp(x_j)

# Cross-Entropy Loss
L = -Σ y_i log(ŷ_i)

# Sigmoid
σ(x) = 1 / (1 + e^(-x))

# Bayes
P(A|B) = P(B|A)P(A) / P(B)

# Normal
N(x|μ,σ²) = (1/√(2πσ²)) exp(-(x-μ)²/(2σ²))

# KL Divergence
KL(P||Q) = Σ P(x) log(P(x)/Q(x))

# Eigenvalues
Av = λv

# SVD
A = UΣV^T
```

