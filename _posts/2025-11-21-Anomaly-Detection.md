---
title: "Anomaly Detection: Complete Professional Guide"
date:   2025-11-21 17:00:00
categories: [Machine-Learning]
tags: [Anomaly Detection, Outlier Detection, Machine Learning, Data Science]    
image:
  path: /assets/imgs/headers/AnomalyDetection.png
---


## Introduction

Anomaly detection (also known as outlier detection) is a critical discipline in machine learning and data science, essential for identifying patterns, observations, or events that deviate significantly from expected normal behavior. In today's industrial context, where data volumes are exploding and systems are becoming increasingly complex, the ability to automatically detect anomalies has become a major competitive advantage.

### Strategic Importance

Anomaly detection systems play a crucial role in:

- **Cybersecurity**: Intrusion detection, DDoS attacks, malicious behavior (average cost of a data breach: $4.35M according to IBM 2022)
- **Finance**: Fraud prevention, suspicious transaction detection (annual fraud losses: $40B+ globally)
- **Industry**: Predictive maintenance, failure detection before critical breakdown (30-50% reduction in downtime)
- **Healthcare**: Medical anomaly detection, patient monitoring, rare disease identification
- **IoT and Smart Cities**: Sensor monitoring, dysfunction detection in urban infrastructure
- **E-commerce**: Fraudulent behavior detection, bots, payment system abuse

### Contemporary Challenges

The field faces several major challenges:

1. **Data volume**: Real-time processing of millions of events per second
2. **Dimensionality**: Handling high-dimensional data (curse of dimensionality)
3. **Class imbalance**: Anomalies often <1% of data
4. **Concept drift**: Evolution of normal and abnormal patterns over time
5. **Explainability**: Need to understand why an observation is considered anomalous
6. **False positives**: Operational cost of false alerts

## Theoretical Foundations

### Formal Definition

An anomaly is an observation whose probability of appearing in the normal distribution model is significantly lower than a defined threshold. In other words, it's a data point that has very little chance of belonging to the normal data distribution.

An anomaly can also be defined in terms of distance: it's an observation whose distance from the center of the normal distribution exceeds a critical threshold. The further a point is from the center, the more likely it is to be an anomaly.

### Learning Framework

Anomaly detection can be formulated according to three paradigms:

#### 1. **Supervised Learning**
- Labeled data with normal and abnormal examples
- Imbalanced binary classification problem
- Requires significant set of anomaly examples
- **Advantages**: High performance, clear metrics
- **Disadvantages**: Annotation cost, bias toward known anomalies

#### 2. **Semi-Supervised Learning (One-Class Learning)**
- Training on normal data only
- Learns a decision boundary around the normal class
- The model returns: 1 if observation is normal, -1 otherwise
- **Advantages**: No need for anomalies in training
- **Disadvantages**: May miss certain types of anomalies

#### 3. **Unsupervised Learning**
- No annotation required
- Based on distribution or density assumptions
- Identifies low-density points as anomalies
- **Advantages**: Applicable without labels, discovers unknown anomalies
- **Disadvantages**: Variable performance, requires expert validation

### Fundamental Assumptions

Anomaly detection algorithms typically rely on one or more of these assumptions:

1. **Density hypothesis**: Anomalies appear in regions of low probability density
2. **Distance hypothesis**: Anomalies are far from their nearest neighbors
3. **Separability hypothesis**: Anomalies can be separated from normal data by a decision boundary
4. **Reconstruction hypothesis**: Anomalies are difficult to reconstruct by a model trained on normal data

## Types of Anomalies

### Detailed Classification

### 1. **Point Anomalies**
Individual data points that are abnormal compared to the rest of the dataset.

**Characteristics:**
- Isolated deviant observations
- Simplest to detect
- Represent ~70% of use cases

**Industrial examples:**
- Bank transaction of €50,000 when average is €100
- Abnormal temperature spike in a datacenter (80°C vs 22°C nominal)
- API response time of 30s vs 200ms typical
- Unusual electricity consumption in a building

**Suitable methods:** Z-Score, IQR, Isolation Forest, LOF

### 2. **Contextual Anomalies**
Observations that are normal in one context but abnormal in another. Context can be temporal, spatial, or multivariate.

**Characteristics:**
- Depend on observation context
- Require considering contextual features
- Common in time series

**Industrial examples:**
- Temperature of 30°C normal in summer but abnormal in winter
- High network traffic during day (normal) vs night (suspect)
- Spending €500 abroad (travel context: normal, usual context: suspect)
- CPU usage at 90% during nightly batch (normal) vs daytime (abnormal)

**Suitable methods:** LSTM, Prophet, Conditional Anomaly Detection, Seasonal Decomposition

### 3. **Collective Anomalies**
A collection of data points that, taken collectively, are abnormal even if individually they may seem normal.

**Characteristics:**
- Require analysis of sequences or groups
- More complex to identify
- Often related to business processes

**Industrial examples:**
- Coordinated purchase sequence indicating a fraud ring
- Unusual network activity pattern suggesting an APT cyberattack
- Series of micro-transactions building fraud
- Progressive performance degradation (drift) before system failure
- Coordinated bot behavior on a platform

**Suitable methods:** Sequential Pattern Mining, RNN/LSTM, HMM, Graph-based methods

### 4. **Additional Anomalies (Advanced Classification)**

#### **Conditional Anomalies**
- Violation of business rules or logical constraints
- Example: Negative age, future date in history

#### **Relative Anomalies**
- Deviations from a reference group
- Example: Store performance vs regional average

#### **Drift Anomalies**
- Gradual changes in distribution
- Example: Progressive degradation of sensor quality

## Approaches and Algorithms

### 1. Statistical Methods

Statistical methods are based on probabilistic models and hypothesis testing. They assume normal data follows a known distribution (often Gaussian).

#### Gaussian Distribution and Z-Score Test

**Theory:**
The Z-score measures how many standard deviations an observation is from the mean. It's calculated by taking the difference between the observed value and the mean, divided by the standard deviation.

**Formula:** Z-score = (value - mean) / standard deviation

An observation is considered abnormal if its absolute Z-score exceeds a threshold (typically 3, corresponding to 99.7% of data in a normal distribution).

**Production-Ready Implementation:**

```python
import numpy as np
from scipy import stats
from typing import Tuple, Optional
import logging

class ZScoreDetector:
    """
    Z-score based anomaly detector with robust handling
    """
    def __init__(self, threshold: float = 3.0, use_modified: bool = False):
        """
        Args:
            threshold: Detection threshold (number of standard deviations)
            use_modified: Use modified Z-score (MAD) for robustness
        """
        self.threshold = threshold
        self.use_modified = use_modified
        self.mu_ = None
        self.sigma_ = None
        self.mad_ = None
        self.logger = logging.getLogger(__name__)
        
    def fit(self, data: np.ndarray) -> 'ZScoreDetector':
        """
        Compute statistics on training data
        """
        if len(data) == 0:
            raise ValueError("Training data is empty")
            
        self.mu_ = np.mean(data)
        self.sigma_ = np.std(data)
        
        if self.use_modified:
            # MAD (Median Absolute Deviation) - more robust to outliers
            median = np.median(data)
            self.mad_ = np.median(np.abs(data - median))
            
        self.logger.info(f"Model trained: μ={self.mu_:.2f}, σ={self.sigma_:.2f}")
        return self
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Predict anomalies (-1) and normal (1)
        """
        scores = self.decision_function(data)
        return np.where(np.abs(scores) > self.threshold, -1, 1)
    
    def decision_function(self, data: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores (Z-scores)
        """
        if self.mu_ is None:
            raise ValueError("Model must be trained before prediction")
            
        if self.use_modified and self.mad_ is not None:
            # Modified Z-score (more robust)
            median = np.median(data)
            return 0.6745 * (data - median) / self.mad_
        else:
            # Classical Z-score
            return (data - self.mu_) / self.sigma_
    
    def get_anomaly_scores(self, data: np.ndarray) -> np.ndarray:
        """
        Return raw scores for analysis
        """
        return np.abs(self.decision_function(data))

# Usage example
np.random.seed(42)
normal_data = np.random.normal(100, 15, 1000)
test_data = np.concatenate([normal_data, [200, 250, -50]])

detector = ZScoreDetector(threshold=3.0, use_modified=True)
detector.fit(normal_data)
predictions = detector.predict(test_data)
scores = detector.get_anomaly_scores(test_data)

print(f"Detected anomalies: {(predictions == -1).sum()}")
print(f"Indices: {np.where(predictions == -1)[0]}")
```

#### IQR Method (Interquartile Range)

**Theory:**
The IQR (Interquartile Range) is robust to outliers and doesn't assume a particular distribution. The IQR is the difference between the 3rd quartile (Q3) and the 1st quartile (Q1).

**Calculation:** IQR = Q3 - Q1

Detection bounds are:
- Lower bound: Q1 - 1.5 × IQR
- Upper bound: Q3 + 1.5 × IQR

Any observation outside these bounds is considered an anomaly.

**Advanced Implementation:**

```python
class IQRDetector:
    """
    IQR-based detector with configurable options
    """
    def __init__(self, multiplier: float = 1.5, percentiles: Tuple[float, float] = (25, 75)):
        """
        Args:
            multiplier: IQR multiplier (1.5 standard, 3.0 for extreme outliers)
            percentiles: Percentiles for Q1 and Q3
        """
        self.multiplier = multiplier
        self.percentiles = percentiles
        self.q1_ = None
        self.q3_ = None
        self.iqr_ = None
        self.lower_bound_ = None
        self.upper_bound_ = None
        
    def fit(self, data: np.ndarray) -> 'IQRDetector':
        """
        Compute quartiles and bounds
        """
        self.q1_ = np.percentile(data, self.percentiles[0])
        self.q3_ = np.percentile(data, self.percentiles[1])
        self.iqr_ = self.q3_ - self.q1_
        
        self.lower_bound_ = self.q1_ - self.multiplier * self.iqr_
        self.upper_bound_ = self.q3_ + self.multiplier * self.iqr_
        
        return self
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Predict anomalies
        """
        is_anomaly = (data < self.lower_bound_) | (data > self.upper_bound_)
        return np.where(is_anomaly, -1, 1)
    
    def decision_function(self, data: np.ndarray) -> np.ndarray:
        """
        Normalized distance to bounds
        """
        scores = np.zeros_like(data, dtype=float)
        
        # Distance below lower bound
        below = data < self.lower_bound_
        scores[below] = (self.lower_bound_ - data[below]) / self.iqr_
        
        # Distance above upper bound
        above = data > self.upper_bound_
        scores[above] = (data[above] - self.upper_bound_) / self.iqr_
        
        return scores
    
    def get_bounds(self) -> Tuple[float, float]:
        """
        Return detection bounds
        """
        return self.lower_bound_, self.upper_bound_

# Usage
detector = IQRDetector(multiplier=1.5)
detector.fit(normal_data)
predictions = detector.predict(test_data)
print(f"Bounds: {detector.get_bounds()}")
```

#### Advanced Statistical Tests

**Grubbs Test (Single Outlier)**

```python
from scipy.stats import t

def grubbs_test(data: np.ndarray, alpha: float = 0.05) -> Tuple[bool, float, float]:
    """
    Grubbs test to detect a single outlier
    
    H0: No outlier
    H1: At least one outlier
    
    Returns:
        (is_outlier, test_statistic, critical_value)
    """
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    
    # Compute G statistic
    abs_deviations = np.abs(data - mean)
    max_idx = np.argmax(abs_deviations)
    G = abs_deviations[max_idx] / std
    
    # Critical value
    t_dist = t.ppf(1 - alpha / (2 * n), n - 2)
    critical_value = ((n - 1) * np.sqrt(t_dist**2)) / np.sqrt(n * (n - 2 + t_dist**2))
    
    return G > critical_value, G, critical_value

# Test
is_outlier, stat, crit = grubbs_test(test_data[-10:])
print(f"Outlier detected: {is_outlier}, G={stat:.3f}, Critical={crit:.3f}")
```

**Dixon Test**

```python
def dixon_test(data: np.ndarray, alpha: float = 0.05) -> bool:
    """
    Dixon test for small samples (n < 30)
    """
    n = len(data)
    if n < 3 or n > 30:
        raise ValueError("Dixon test applicable for 3 <= n <= 30")
    
    sorted_data = np.sort(data)
    
    # Compute Q ratio
    if n <= 7:
        Q = (sorted_data[1] - sorted_data[0]) / (sorted_data[-1] - sorted_data[0])
    else:
        Q = (sorted_data[1] - sorted_data[0]) / (sorted_data[-2] - sorted_data[0])
    
    # Critical values (simplified for alpha=0.05)
    critical_values = {3: 0.941, 4: 0.765, 5: 0.642, 6: 0.560, 7: 0.507}
    critical = critical_values.get(n, 0.5)
    
    return Q > critical
```

### 2. Machine Learning Methods

#### Isolation Forest - In-Depth Analysis

**Theoretical Principle:**

Isolation Forest is based on the principle that anomalies are "easier to isolate" than normal points. The algorithm builds isolation trees by recursively separating data with random splits.

**Anomaly score:**

The anomaly score is calculated from the path length needed to isolate an observation in the trees. The shorter the path, the easier the observation is to isolate, so the more likely it is to be an anomaly.

The score is normalized to obtain a value between 0 and 1, using the average path length as reference. The formula takes into account the expected average depth in a random isolation tree.

**Interpretation:**
- Score close to 1: strong anomaly
- Score ~0.5: normal point
- Score <0.5: very normal point (cluster center)

**Production Implementation with Optimizations:**

```python
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import numpy as np
from typing import Dict, Tuple
import joblib

class ProductionIsolationForest:
    """
    Isolation Forest optimized for production environment
    """
    def __init__(
        self,
        contamination: float = 0.1,
        n_estimators: int = 100,
        max_samples: int = 256,
        max_features: float = 1.0,
        n_jobs: int = -1,
        random_state: int = 42
    ):
        """
        Args:
            contamination: Expected proportion of anomalies
            n_estimators: Number of trees (more = stable but slow)
            max_samples: Samples per tree (256 optimal per paper)
            max_features: Proportion of features to consider
            n_jobs: Number of CPUs (-1 = all)
        """
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            n_jobs=n_jobs,
            random_state=random_state,
            warm_start=False,
            bootstrap=False
        )
        self.feature_names_ = None
        self.threshold_ = None
        
    def fit(self, X: np.ndarray, feature_names: list = None) -> 'ProductionIsolationForest':
        """
        Train the model
        """
        # Normalization
        X_scaled = self.scaler.fit_transform(X)
        
        # Training
        self.model.fit(X_scaled)
        
        # Compute decision threshold
        scores = self.model.score_samples(X_scaled)
        self.threshold_ = np.percentile(scores, self.contamination * 100)
        
        self.feature_names_ = feature_names
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies (-1) and normal (1)
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Anomaly scores (more negative = more abnormal)
        """
        X_scaled = self.scaler.transform(X)
        return self.model.score_samples(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Probability of being an anomaly [0, 1]
        """
        scores = self.decision_function(X)
        # Normalization between 0 and 1
        min_score = scores.min()
        max_score = scores.max()
        proba = 1 - (scores - min_score) / (max_score - min_score)
        return proba
    
    def explain_anomaly(self, X: np.ndarray, idx: int) -> Dict:
        """
        Explain why an observation is anomalous
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        X_scaled = self.scaler.transform(X)
        score = self.model.score_samples(X_scaled)[idx]
        prediction = self.model.predict(X_scaled)[idx]
        
        # Feature contribution (approximation)
        contributions = {}
        if self.feature_names_:
            X_mean = np.zeros_like(X[idx])
            base_score = self.model.score_samples(
                self.scaler.transform(X_mean.reshape(1, -1))
            )[0]
            
            for i, fname in enumerate(self.feature_names_):
                X_modified = X[idx].copy()
                X_modified[i] = 0  # Neutral value
                modified_score = self.model.score_samples(
                    self.scaler.transform(X_modified.reshape(1, -1))
                )[0]
                contributions[fname] = abs(score - modified_score)
        
        return {
            'anomaly_score': score,
            'is_anomaly': prediction == -1,
            'threshold': self.threshold_,
            'feature_contributions': contributions
        }
    
    def save(self, filepath: str):
        """
        Save the model
        """
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'contamination': self.contamination,
            'threshold': self.threshold_,
            'feature_names': self.feature_names_
        }, filepath)
    
    @classmethod
    def load(cls, filepath: str) -> 'ProductionIsolationForest':
        """
        Load a saved model
        """
        data = joblib.load(filepath)
        instance = cls(contamination=data['contamination'])
        instance.model = data['model']
        instance.scaler = data['scaler']
        instance.threshold_ = data['threshold']
        instance.feature_names_ = data['feature_names']
        return instance

# Advanced usage example
np.random.seed(42)

# Generate realistic data
n_samples = 10000
n_features = 10

# Normal data (multivariate distribution)
mean = np.zeros(n_features)
cov = np.eye(n_features)
X_normal = np.random.multivariate_normal(mean, cov, n_samples)

# Anomalies (different types)
n_anomalies = 500
X_anomalies = np.vstack([
    np.random.uniform(-5, 5, (n_anomalies // 2, n_features)),  # Uniform
    np.random.normal(4, 1, (n_anomalies // 2, n_features))     # Distant cluster
])

X = np.vstack([X_normal, X_anomalies])
y_true = np.hstack([np.zeros(n_samples), np.ones(n_anomalies)])

# Training
feature_names = [f'feature_{i}' for i in range(n_features)]
detector = ProductionIsolationForest(
    contamination=0.05,
    n_estimators=200,
    max_samples=256
)
detector.fit(X_normal, feature_names=feature_names)

# Prediction
predictions = detector.predict(X)
probas = detector.predict_proba(X)
scores = detector.decision_function(X)

# Explain an anomaly
anomaly_idx = np.where(predictions == -1)[0][0]
explanation = detector.explain_anomaly(X, anomaly_idx)
print(f"Explanation for anomaly #{anomaly_idx}:")
print(f"  Score: {explanation['anomaly_score']:.4f}")
print(f"  Threshold: {explanation['threshold']:.4f}")
print(f"  Contributions: {explanation['feature_contributions']}")

# Save
detector.save('isolation_forest_model.joblib')
```

#### One-Class SVM - Deep Dive

**Theory:**

One-Class SVM seeks a hypersphere (or hyperplane) of minimal volume containing most of the normal data.

**Objective function:**

The algorithm minimizes a function that balances two objectives:
1. Minimize the size of the hypersphere (to have the tightest boundary possible)
2. Minimize violations (normal points outside the boundary)

The nu parameter (ν) controls the maximum fraction of expected anomalies and must be set between 0 and 1. The rho parameter (ρ) represents the hyperplane offset from the origin.

**Optimized Implementation:**

```python
from sklearn.svm import OneClassSVM
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
import numpy as np

class OptimizedOneClassSVM:
    """
    One-Class SVM with hyperparameter optimization
    """
    def __init__(self, kernel: str = 'rbf', nu: float = 0.1, gamma: str = 'scale'):
        self.kernel = kernel
        self.nu = nu
        self.gamma = gamma
        self.scaler = StandardScaler()
        self.model = None
        self.best_params_ = None
        
    def fit(self, X: np.ndarray, optimize: bool = False) -> 'OptimizedOneClassSVM':
        """
        Train the model with optimization option
        """
        X_scaled = self.scaler.fit_transform(X)
        
        if optimize:
            self._optimize_hyperparameters(X_scaled)
        else:
            self.model = OneClassSVM(
                kernel=self.kernel,
                nu=self.nu,
                gamma=self.gamma
            )
            self.model.fit(X_scaled)
            
        return self
    
    def _optimize_hyperparameters(self, X: np.ndarray):
        """
        Optimize hyperparameters via cross-validation
        """
        param_grid = {
            'kernel': ['rbf', 'poly', 'sigmoid'],
            'nu': [0.01, 0.05, 0.1, 0.2],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
        }
        
        # Custom scorer (hypersphere volume)
        def volume_scorer(estimator, X):
            n_support = len(estimator.support_vectors_)
            return -n_support / len(X)  # Minimize number of support vectors
        
        grid_search = GridSearchCV(
            OneClassSVM(),
            param_grid,
            scoring=make_scorer(volume_scorer),
            cv=5,
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X)
        self.model = grid_search.best_estimator_
        self.best_params_ = grid_search.best_params_
        
        print(f"Best parameters: {self.best_params_}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.decision_function(X_scaled)
    
    def get_support_vectors(self) -> np.ndarray:
        """
        Return support vectors (decision boundary)
        """
        return self.scaler.inverse_transform(self.model.support_vectors_)

# Usage with optimization
svm_detector = OptimizedOneClassSVM()
svm_detector.fit(X_normal, optimize=True)
predictions = svm_detector.predict(X)
```

#### Local Outlier Factor (LOF) - Advanced Version

**Theory:**

LOF measures the local density of a point relative to its neighbors.

**Calculation in three steps:**

1. **Reachability distance (reach-dist)**: Maximum distance between two points considering the distance to the k-th nearest neighbor. This distance is calculated as the maximum between the k-th neighbor distance of B and the actual distance between A and B.

2. **Local reachability density (lrd)**: Inverse of the average reachability distance from a point to its k nearest neighbors. Higher value means the point is in a dense area.

3. **LOF (Local Outlier Factor)**: Ratio between the average local density of neighbors and the local density of the point. Indicates how isolated a point is relative to its neighbors.

**Interpretation:**
- LOF ≈ 1: similar density to neighbors (normal)
- LOF > 1: lower density (anomaly)
- LOF >> 1: strong anomaly

**Production Implementation:**

```python
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import ParameterGrid

class AdaptiveLOF:
    """
    LOF with dynamic adaptation of number of neighbors
    """
    def __init__(
        self,
        n_neighbors: int = 20,
        contamination: float = 0.1,
        novelty: bool = True,
        metric: str = 'minkowski'
    ):
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.novelty = novelty
        self.metric = metric
        self.scaler = StandardScaler()
        self.model = None
        
    def fit(self, X: np.ndarray) -> 'AdaptiveLOF':
        """
        Train LOF model
        """
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            novelty=self.novelty,
            metric=self.metric,
            n_jobs=-1
        )
        
        if self.novelty:
            self.model.fit(X_scaled)
        else:
            # Non-novelty mode: fit_predict in one go
            self.model.fit_predict(X_scaled)
            
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies (requires novelty=True)
        """
        if not self.novelty:
            raise ValueError("predict() requires novelty=True")
            
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        LOF scores (more negative = more abnormal)
        """
        if not self.novelty:
            raise ValueError("decision_function() requires novelty=True")
            
        X_scaled = self.scaler.transform(X)
        return self.model.decision_function(X_scaled)
    
    def get_lof_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Return raw LOF scores (>1 = anomaly)
        """
        scores = -self.decision_function(X)
        return scores
    
    @staticmethod
    def select_optimal_k(X: np.ndarray, k_range: range = range(5, 51, 5)) -> int:
        """
        Select optimal number of neighbors
        """
        silhouette_scores = []
        
        for k in k_range:
            lof = LocalOutlierFactor(n_neighbors=k, novelty=False)
            labels = lof.fit_predict(X)
            
            # If all points classified the same, skip
            if len(np.unique(labels)) == 1:
                continue
                
            from sklearn.metrics import silhouette_score
            score = silhouette_score(X, labels)
            silhouette_scores.append((k, score))
        
        best_k = max(silhouette_scores, key=lambda x: x[1])[0]
        return best_k

# Example with automatic k selection
optimal_k = AdaptiveLOF.select_optimal_k(X_normal)
print(f"Optimal number of neighbors: {optimal_k}")

lof_detector = AdaptiveLOF(n_neighbors=optimal_k, novelty=True)
lof_detector.fit(X_normal)
predictions = lof_detector.predict(X)
lof_scores = lof_detector.get_lof_scores(X)

print(f"Detected anomalies: {(predictions == -1).sum()}")
print(f"LOF scores (anomalies): min={lof_scores[predictions == -1].min():.2f}, "
      f"max={lof_scores[predictions == -1].max():.2f}")
```

### 3. Deep Learning Methods

#### Autoencoders - Architecture and Theory

**Fundamental Principle:**

An autoencoder learns a compressed representation (encoding) of normal data. Anomalies, not being part of the training distribution, will be poorly reconstructed.

**Architecture:**

The autoencoder consists of two parts:
- **The encoder** transforms input data (x) into a compressed representation (z) in the latent space
- **The decoder** reconstructs the original data (x̂) from this compressed representation

**Loss function:**

The loss function measures reconstruction error by calculating the average squared difference between original and reconstructed data. This error is called Mean Squared Error (MSE).

**Detection:**

An observation is abnormal if reconstruction error (difference between original and reconstruction) exceeds a defined threshold. Anomalies are hard to reconstruct because they weren't present in training data.

**Production-Ready Implementation:**

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, callbacks
import numpy as np
from typing import Tuple, Optional
import matplotlib.pyplot as plt

class AnomalyAutoencoder:
    """
    Sophisticated autoencoder for anomaly detection
    """
    def __init__(
        self,
        input_dim: int,
        encoding_dims: list = [64, 32, 16],
        activation: str = 'relu',
        dropout_rate: float = 0.2,
        l2_reg: float = 1e-5
    ):
        """
        Args:
            input_dim: Input feature dimension
            encoding_dims: List of dimensions for encoding layers
            activation: Activation function
            dropout_rate: Dropout rate for regularization
            l2_reg: L2 regularization coefficient
        """
        self.input_dim = input_dim
        self.encoding_dims = encoding_dims
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        
        self.autoencoder = None
        self.encoder = None
        self.decoder = None
        self.threshold_ = None
        self.history_ = None
        
        self._build_model()
    
    def _build_model(self):
        """
        Build autoencoder architecture
        """
        # Regularization
        regularizer = keras.regularizers.l2(self.l2_reg)
        
        # ===== ENCODER =====
        input_layer = keras.Input(shape=(self.input_dim,), name='input')
        x = input_layer
        
        # Progressive encoding layers
        for i, dim in enumerate(self.encoding_dims):
            x = layers.Dense(
                dim,
                activation=self.activation,
                kernel_regularizer=regularizer,
                name=f'encoder_{i}'
            )(x)
            x = layers.BatchNormalization(name=f'bn_encoder_{i}')(x)
            x = layers.Dropout(self.dropout_rate, name=f'dropout_encoder_{i}')(x)
        
        # Bottleneck (latent space)
        latent_dim = self.encoding_dims[-1]
        latent = layers.Dense(
            latent_dim,
            activation=self.activation,
            name='latent_space'
        )(x)
        
        # ===== DECODER =====
        x = latent
        
        # Decoding layers (mirror of encoder)
        for i, dim in enumerate(reversed(self.encoding_dims[:-1])):
            x = layers.Dense(
                dim,
                activation=self.activation,
                kernel_regularizer=regularizer,
                name=f'decoder_{i}'
            )(x)
            x = layers.BatchNormalization(name=f'bn_decoder_{i}')(x)
            x = layers.Dropout(self.dropout_rate, name=f'dropout_decoder_{i}')(x)
        
        # Output layer (reconstruction)
        output = layers.Dense(
            self.input_dim,
            activation='linear',  # Or 'sigmoid' if data normalized [0,1]
            name='output'
        )(x)
        
        # Models
        self.autoencoder = Model(input_layer, output, name='autoencoder')
        self.encoder = Model(input_layer, latent, name='encoder')
        
        # Separate decoder
        decoder_input = keras.Input(shape=(latent_dim,))
        decoder_layers = decoder_input
        for layer in self.autoencoder.layers[len(self.encoding_dims)+3:]:
            decoder_layers = layer(decoder_layers)
        self.decoder = Model(decoder_input, decoder_layers, name='decoder')
    
    def compile(
        self,
        optimizer: str = 'adam',
        learning_rate: float = 0.001,
        loss: str = 'mse'
    ):
        """
        Compile the model
        """
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
        self.autoencoder.compile(optimizer=opt, loss=loss)
    
    def fit(
        self,
        X_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: int = 1
    ) -> 'AnomalyAutoencoder':
        """
        Train autoencoder on normal data
        """
        # Callbacks
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss' if X_val is not None else 'loss',
            patience=10,
            restore_best_weights=True,
            verbose=verbose
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss' if X_val is not None else 'loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=verbose
        )
        
        # Training
        validation_data = (X_val, X_val) if X_val is not None else None
        
        self.history_ = self.autoencoder.fit(
            X_train, X_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=verbose
        )
        
        # Compute detection threshold
        reconstructions = self.autoencoder.predict(X_train, verbose=0)
        mse = np.mean(np.power(X_train - reconstructions, 2), axis=1)
        self.threshold_ = np.percentile(mse, 95)  # 95th percentile
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies (-1) and normal (1)
        """
        reconstruction_errors = self.reconstruction_error(X)
        return np.where(reconstruction_errors > self.threshold_, -1, 1)
    
    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """
        Compute reconstruction error (MSE)
        """
        reconstructions = self.autoencoder.predict(X, verbose=0)
        mse = np.mean(np.power(X - reconstructions, 2), axis=1)
        return mse
    
    def encode(self, X: np.ndarray) -> np.ndarray:
        """
        Encode data into latent space
        """
        return self.encoder.predict(X, verbose=0)
    
    def decode(self, latent: np.ndarray) -> np.ndarray:
        """
        Decode from latent space
        """
        return self.decoder.predict(latent, verbose=0)
    
    def plot_training_history(self):
        """
        Visualize training history
        """
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.history_.history['loss'], label='Training Loss')
        if 'val_loss' in self.history_.history:
            plt.plot(self.history_.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training History')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.hist(self.history_.history['loss'], bins=50, alpha=0.7)
        plt.axvline(self.threshold_, color='r', linestyle='--', 
                   label=f'Threshold: {self.threshold_:.4f}')
        plt.xlabel('Reconstruction Error')
        plt.ylabel('Frequency')
        plt.legend()
        plt.title('Error Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def save(self, filepath: str):
        """
        Save complete model
        """
        self.autoencoder.save(f'{filepath}_autoencoder.h5')
        np.save(f'{filepath}_threshold.npy', self.threshold_)
    
    @classmethod
    def load(cls, filepath: str, input_dim: int) -> 'AnomalyAutoencoder':
        """
        Load saved model
        """
        instance = cls(input_dim=input_dim)
        instance.autoencoder = keras.models.load_model(f'{filepath}_autoencoder.h5')
        instance.threshold_ = np.load(f'{filepath}_threshold.npy')
        return instance

# Advanced usage example
np.random.seed(42)
tf.random.set_seed(42)

# Training data (normal)
n_train = 5000
n_features = 20
X_train_normal = np.random.randn(n_train, n_features)

# Test data (normal + anomalies)
n_test = 1000
n_anomalies = 100
X_test_normal = np.random.randn(n_test - n_anomalies, n_features)
X_test_anomalies = np.random.uniform(-5, 5, (n_anomalies, n_features))
X_test = np.vstack([X_test_normal, X_test_anomalies])
y_test = np.hstack([np.zeros(n_test - n_anomalies), np.ones(n_anomalies)])

# Validation split
from sklearn.model_selection import train_test_split
X_train, X_val = train_test_split(X_train_normal, test_size=0.2, random_state=42)

# Create and train
ae = AnomalyAutoencoder(
    input_dim=n_features,
    encoding_dims=[64, 32, 16],
    dropout_rate=0.2
)
ae.compile(learning_rate=0.001)

print("Training autoencoder...")
ae.fit(X_train, X_val, epochs=100, batch_size=64, verbose=1)

# Prediction
predictions = ae.predict(X_test)
reconstruction_errors = ae.reconstruction_error(X_test)

# Evaluation
from sklearn.metrics import classification_report, confusion_matrix
print("\nPerformance:")
print(classification_report(y_test, (predictions == -1).astype(int),
                           target_names=['Normal', 'Anomaly']))

# Visualization
ae.plot_training_history()

# Visualization of reconstruction errors
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(range(len(reconstruction_errors)), reconstruction_errors,
           c=['blue' if p == 1 else 'red' for p in predictions], alpha=0.5)
plt.axhline(ae.threshold_, color='black', linestyle='--', label='Threshold')
plt.xlabel('Sample Index')
plt.ylabel('Reconstruction Error')
plt.title('Reconstruction Errors')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(reconstruction_errors[y_test == 0], bins=50, alpha=0.7, label='Normal')
plt.hist(reconstruction_errors[y_test == 1], bins=50, alpha=0.7, label='Anomaly')
plt.axvline(ae.threshold_, color='black', linestyle='--', label='Threshold')
plt.xlabel('Reconstruction Error')
plt.ylabel('Frequency')
plt.legend()
plt.title('Error Distribution by Class')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

#### Variational Autoencoder (VAE)

**Theory:**

VAE (Variational Autoencoders) model the latent space as a probabilistic distribution rather than a deterministic vector. Instead of creating a single point in latent space, they create a probability distribution.

**VAE Loss Function:**

The loss function combines two components:
1. **Reconstruction error**: Measures the difference between input and output (like a classical autoencoder)
2. **KL divergence (Kullback-Leibler)**: Measures how much the learned latent distribution deviates from a standard Gaussian distribution (mean=0, variance=1)

The beta parameter controls the relative importance of these two components. Higher beta forces the latent space to be closer to a standard Gaussian distribution, potentially at the cost of reconstruction quality.

**Implementation:**

```python
class VariationalAutoencoder(keras.Model):
    """
    VAE for anomaly detection
    """
    def __init__(self, input_dim: int, latent_dim: int = 10, beta: float = 1.0):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta
        
        # Encoder
        self.encoder_network = keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu')
        ])
        
        # Latent distribution parameters
        self.mean_layer = layers.Dense(latent_dim, name='z_mean')
        self.log_var_layer = layers.Dense(latent_dim, name='z_log_var')
        
        # Decoder
        self.decoder_network = keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(input_dim, activation='linear')
        ])
        
        self.threshold_ = None
    
    def encode(self, x):
        """
        Encode x into latent distribution
        """
        h = self.encoder_network(x)
        z_mean = self.mean_layer(h)
        z_log_var = self.log_var_layer(h)
        return z_mean, z_log_var
    
    def reparameterize(self, z_mean, z_log_var):
        """
        Reparameterization trick for backprop
        """
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    def decode(self, z):
        """
        Decode from latent space
        """
        return self.decoder_network(z)
    
    def call(self, x, training=None):
        """
        Complete forward pass
        """
        z_mean, z_log_var = self.encode(x)
        z = self.reparameterize(z_mean, z_log_var)
        reconstruction = self.decode(z)
        
        # Compute KL loss
        kl_loss = -0.5 * tf.reduce_mean(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        )
        
        # Add KL loss to metrics
        self.add_loss(self.beta * kl_loss)
        self.add_metric(kl_loss, name='kl_loss')
        
        return reconstruction
    
    def anomaly_score(self, x):
        """
        Compute anomaly score
        """
        reconstruction = self(x, training=False)
        # Reconstruction error
        mse = tf.reduce_mean(tf.square(x - reconstruction), axis=1)
        
        # Combined score with latent probability
        z_mean, z_log_var = self.encode(x)
        log_prob = -0.5 * tf.reduce_sum(
            tf.square(z_mean) + tf.exp(z_log_var), axis=1
        )
        
        # Final score
        return mse - log_prob
    
    def fit_with_threshold(self, X_train, epochs=100, batch_size=32):
        """
        Train and compute threshold
        """
        self.compile(optimizer='adam', loss='mse')
        self.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, verbose=1)
        
        # Compute threshold
        scores = self.anomaly_score(X_train).numpy()
        self.threshold_ = np.percentile(scores, 95)
        
        return self
    
    def predict_anomalies(self, X):
        """
        Predict anomalies
        """
        scores = self.anomaly_score(X).numpy()
        return np.where(scores > self.threshold_, -1, 1)

# Example
vae = VariationalAutoencoder(input_dim=n_features, latent_dim=10, beta=1.0)
vae.fit_with_threshold(X_train, epochs=50)
vae_predictions = vae.predict_anomalies(X_test)
print(f"VAE - Detected anomalies: {(vae_predictions == -1).sum()}")
```

#### LSTM for Time Series - Advanced Version

**Architecture for Temporal Anomaly Detection:**

LSTMs are ideal for time series as they capture long-term dependencies and sequential patterns.

**Production Implementation:**

```python
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Dropout
from tensorflow.keras.models import Sequential
import pandas as pd

class LSTMAnomalyDetector:
    """
    Anomaly detector for time series with LSTM
    """
    def __init__(
        self,
        timesteps: int,
        n_features: int,
        lstm_units: list = [128, 64],
        dropout_rate: float = 0.2
    ):
        """
        Args:
            timesteps: Number of time steps in each sequence
            n_features: Number of features per time step
            lstm_units: List of LSTM units for each layer
            dropout_rate: Dropout rate
        """
        self.timesteps = timesteps
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        
        self.model = None
        self.threshold_ = None
        self.scaler = StandardScaler()
        
        self._build_model()
    
    def _build_model(self):
        """
        Build LSTM autoencoder
        """
        self.model = Sequential(name='LSTM_Autoencoder')
        
        # ===== ENCODER =====
        # First LSTM layer
        self.model.add(LSTM(
            self.lstm_units[0],
            activation='tanh',
            input_shape=(self.timesteps, self.n_features),
            return_sequences=True if len(self.lstm_units) > 1 else False,
            name='encoder_lstm_1'
        ))
        self.model.add(Dropout(self.dropout_rate))
        
        # Additional LSTM layers
        for i, units in enumerate(self.lstm_units[1:], start=2):
            return_seq = i < len(self.lstm_units)
            self.model.add(LSTM(
                units,
                activation='tanh',
                return_sequences=return_seq,
                name=f'encoder_lstm_{i}'
            ))
            self.model.add(Dropout(self.dropout_rate))
        
        # ===== DECODER =====
        # Repeat latent vector
        self.model.add(RepeatVector(self.timesteps, name='repeat_vector'))
        
        # LSTM decoding layers
        for i, units in enumerate(reversed(self.lstm_units), start=1):
            self.model.add(LSTM(
                units,
                activation='tanh',
                return_sequences=True,
                name=f'decoder_lstm_{i}'
            ))
            self.model.add(Dropout(self.dropout_rate))
        
        # Output layer
        self.model.add(TimeDistributed(
            Dense(self.n_features, activation='linear'),
            name='output'
        ))
    
    def compile(self, learning_rate: float = 0.001, loss: str = 'mse'):
        """
        Compile the model
        """
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss=loss)
    
    def create_sequences(
        self,
        data: np.ndarray,
        window_size: int = None
    ) -> np.ndarray:
        """
        Create sliding sequences for training
        """
        if window_size is None:
            window_size = self.timesteps
            
        sequences = []
        for i in range(len(data) - window_size + 1):
            sequences.append(data[i:i + window_size])
        
        return np.array(sequences)
    
    def fit(
        self,
        X_train: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2,
        verbose: int = 1
    ) -> 'LSTMAnomalyDetector':
        """
        Train the model
        """
        # Normalization
        original_shape = X_train.shape
        X_flat = X_train.reshape(-1, self.n_features)
        X_scaled = self.scaler.fit_transform(X_flat)
        X_train_scaled = X_scaled.reshape(original_shape)
        
        # Callbacks
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-7
        )
        
        # Training
        self.model.fit(
            X_train_scaled, X_train_scaled,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stop, reduce_lr],
            verbose=verbose
        )
        
        # Compute threshold
        reconstructions = self.model.predict(X_train_scaled, verbose=0)
        mse = np.mean(np.power(X_train_scaled - reconstructions, 2), axis=(1, 2))
        self.threshold_ = np.percentile(mse, 95)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies
        """
        reconstruction_errors = self.reconstruction_error(X)
        return np.where(reconstruction_errors > self.threshold_, -1, 1)
    
    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """
        Compute reconstruction error
        """
        # Normalization
        original_shape = X.shape
        X_flat = X.reshape(-1, self.n_features)
        X_scaled = self.scaler.transform(X_flat)
        X_scaled = X_scaled.reshape(original_shape)
        
        # Reconstruction
        reconstructions = self.model.predict(X_scaled, verbose=0)
        
        # MSE per sequence
        mse = np.mean(np.power(X_scaled - reconstructions, 2), axis=(1, 2))
        
        return mse
    
    def plot_reconstruction(self, X: np.ndarray, idx: int = 0):
        """
        Visualize reconstruction of a sequence
        """
        X_scaled = self.scaler.transform(
            X[idx].reshape(-1, self.n_features)
        ).reshape(self.timesteps, self.n_features)
        
        reconstruction = self.model.predict(
            X_scaled.reshape(1, self.timesteps, self.n_features),
            verbose=0
        )[0]
        
        fig, axes = plt.subplots(self.n_features, 1, figsize=(12, 3*self.n_features))
        if self.n_features == 1:
            axes = [axes]
        
        for i, ax in enumerate(axes):
            ax.plot(X_scaled[:, i], label='Original', marker='o')
            ax.plot(reconstruction[:, i], label='Reconstruction', marker='x')
            ax.set_title(f'Feature {i}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# Example with time series data
np.random.seed(42)

# Generate synthetic time series
def generate_timeseries(n_sequences=1000, timesteps=50, n_features=3):
    """
    Generate time series with sinusoidal patterns
    """
    X = []
    for _ in range(n_sequences):
        # Sinusoidal pattern with noise
        t = np.linspace(0, 4*np.pi, timesteps)
        series = []
        for f in range(n_features):
            freq = np.random.uniform(0.5, 2.0)
            phase = np.random.uniform(0, 2*np.pi)
            amplitude = np.random.uniform(0.5, 2.0)
            noise = np.random.normal(0, 0.1, timesteps)
            
            signal = amplitude * np.sin(freq * t + phase) + noise
            series.append(signal)
        
        X.append(np.array(series).T)
    
    return np.array(X)

# Training data (normal)
X_train_ts = generate_timeseries(n_sequences=800, timesteps=50, n_features=3)

# Test data (normal + anomalies)
X_test_normal = generate_timeseries(n_sequences=150, timesteps=50, n_features=3)

# Anomalies: series with different pattern
X_test_anomalies = np.random.uniform(-3, 3, (50, 50, 3))

X_test_ts = np.vstack([X_test_normal, X_test_anomalies])
y_test_ts = np.hstack([np.zeros(150), np.ones(50)])

# Training
lstm_detector = LSTMAnomalyDetector(
    timesteps=50,
    n_features=3,
    lstm_units=[128, 64],
    dropout_rate=0.2
)
lstm_detector.compile(learning_rate=0.001)

print("Training LSTM...")
lstm_detector.fit(X_train_ts, epochs=50, batch_size=32, verbose=1)

# Prediction
predictions_ts = lstm_detector.predict(X_test_ts)
errors_ts = lstm_detector.reconstruction_error(X_test_ts)

# Evaluation
print("\nLSTM Performance:")
print(classification_report(y_test_ts, (predictions_ts == -1).astype(int),
                           target_names=['Normal', 'Anomaly']))

# Visualization of a reconstruction
lstm_detector.plot_reconstruction(X_test_ts, idx=0)
```

#### GAN for Anomaly Detection

**GAN-based Anomaly Detection Architecture:**

```python
class AnomalyGAN:
    """
    Generative Adversarial Network for anomaly detection
    """
    def __init__(self, input_dim: int, latent_dim: int = 100):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()
        self.gan = self._build_gan()
    
    def _build_generator(self):
        """
        Generator: latent -> data
        """
        model = Sequential([
            layers.Dense(128, activation='relu', input_dim=self.latent_dim),
            layers.BatchNormalization(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(self.input_dim, activation='tanh')
        ], name='generator')
        return model
    
    def _build_discriminator(self):
        """
        Discriminator: data -> probability
        """
        model = Sequential([
            layers.Dense(256, activation='relu', input_dim=self.input_dim),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ], name='discriminator')
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def _build_gan(self):
        """
        Complete GAN
        """
        self.discriminator.trainable = False
        
        model = Sequential([
            self.generator,
            self.discriminator
        ], name='gan')
        
        model.compile(optimizer='adam', loss='binary_crossentropy')
        return model
    
    def train(self, X_train: np.ndarray, epochs: int = 100, batch_size: int = 128):
        """
        Train the GAN
        """
        for epoch in range(epochs):
            # Select real batch
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            real_data = X_train[idx]
            
            # Generate fake data
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            fake_data = self.generator.predict(noise, verbose=0)
            
            # Train discriminator
            d_loss_real = self.discriminator.train_on_batch(
                real_data, np.ones((batch_size, 1))
            )
            d_loss_fake = self.discriminator.train_on_batch(
                fake_data, np.zeros((batch_size, 1))
            )
            
            # Train generator
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.gan.train_on_batch(noise, np.ones((batch_size, 1)))
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: D_loss={d_loss_real[0]:.4f}, "
                      f"G_loss={g_loss:.4f}")
    
    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly score based on discriminator
        """
        # Discriminator score (probability of being real)
        scores = self.discriminator.predict(X, verbose=0)
        # Inversion: low score = likely anomaly
        return 1 - scores.flatten()
```

## Advanced Evaluation Metrics

### Metrics for Imbalanced Data

In anomaly detection problems, classes are heavily imbalanced (anomalies typically <5%). Accuracy is therefore a misleading metric.

#### 1. Precision, Recall, F1-Score

**Precision** = True Positives / (True Positives + False Positives)
- Proportion of correctly identified anomalies among all anomaly predictions

**Recall** = True Positives / (True Positives + False Negatives)
- Proportion of actual anomalies that were detected

**F1-Score** = 2 × (Precision × Recall) / (Precision + Recall)
- Harmonic mean of precision and recall

**Business Interpretation:**
- **Precision**: Among triggered alerts, what proportion is legitimate? (Cost of false positives)
- **Recall**: Among true anomalies, what proportion is detected? (Cost of false negatives)

#### 2. ROC-AUC and PR-AUC

```python
from sklearn.metrics import (
    roc_auc_score, average_precision_score, 
    precision_recall_curve, roc_curve
)
import matplotlib.pyplot as plt

class AnomalyMetricsEvaluator:
    """
    Complete metrics evaluator for anomaly detection
    """
    def __init__(self, y_true: np.ndarray, y_scores: np.ndarray, threshold: float = None):
        """
        Args:
            y_true: True labels (0=normal, 1=anomaly)
            y_scores: Anomaly scores (higher = more abnormal)
            threshold: Decision threshold (auto-calculated if None)
        """
        self.y_true = y_true
        self.y_scores = y_scores
        self.threshold = threshold or self._find_optimal_threshold()
        self.y_pred = (y_scores >= self.threshold).astype(int)
    
    def _find_optimal_threshold(self) -> float:
        """
        Find optimal threshold via F1-Score
        """
        precision, recall, thresholds = precision_recall_curve(self.y_true, self.y_scores)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        return thresholds[optimal_idx] if optimal_idx < len(thresholds) else thresholds[-1]
    
    def compute_metrics(self) -> dict:
        """
        Compute all important metrics
        """
        from sklearn.metrics import (
            precision_score, recall_score, f1_score, 
            accuracy_score, matthews_corrcoef, cohen_kappa_score
        )
        
        metrics = {
            'accuracy': accuracy_score(self.y_true, self.y_pred),
            'precision': precision_score(self.y_true, self.y_pred, zero_division=0),
            'recall': recall_score(self.y_true, self.y_pred, zero_division=0),
            'f1_score': f1_score(self.y_true, self.y_pred, zero_division=0),
            'roc_auc': roc_auc_score(self.y_true, self.y_scores),
            'pr_auc': average_precision_score(self.y_true, self.y_scores),
            'mcc': matthews_corrcoef(self.y_true, self.y_pred),
            'cohen_kappa': cohen_kappa_score(self.y_true, self.y_pred),
            'threshold': self.threshold
        }
        
        return metrics
    
    def compute_confusion_matrix_costs(
        self, 
        cost_fp: float = 1.0, 
        cost_fn: float = 10.0
    ) -> dict:
        """
        Compute business cost of detection
        
        Args:
            cost_fp: Cost of a false positive (useless alert)
            cost_fn: Cost of a false negative (missed anomaly)
        """
        from sklearn.metrics import confusion_matrix
        
        tn, fp, fn, tp = confusion_matrix(self.y_true, self.y_pred).ravel()
        
        total_cost = cost_fp * fp + cost_fn * fn
        avg_cost_per_sample = total_cost / len(self.y_true)
        
        return {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'total_cost': total_cost,
            'avg_cost_per_sample': avg_cost_per_sample,
            'cost_fp': cost_fp,
            'cost_fn': cost_fn
        }
    
    def plot_roc_curve(self, ax=None):
        """
        Plot ROC curve
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        fpr, tpr, _ = roc_curve(self.y_true, self.y_scores)
        roc_auc = roc_auc_score(self.y_true, self.y_scores)
        
        ax.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})', linewidth=2)
        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_pr_curve(self, ax=None):
        """
        Plot Precision-Recall curve
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        precision, recall, _ = precision_recall_curve(self.y_true, self.y_scores)
        pr_auc = average_precision_score(self.y_true, self.y_scores)
        
        ax.plot(recall, precision, label=f'PR (AUC = {pr_auc:.3f})', linewidth=2)
        baseline = np.sum(self.y_true) / len(self.y_true)
        ax.axhline(baseline, color='k', linestyle='--', label=f'Baseline ({baseline:.3f})')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_score_distribution(self, ax=None):
        """
        Visualize score distribution
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        normal_scores = self.y_scores[self.y_true == 0]
        anomaly_scores = self.y_scores[self.y_true == 1]
        
        ax.hist(normal_scores, bins=50, alpha=0.7, label='Normal', density=True)
        ax.hist(anomaly_scores, bins=50, alpha=0.7, label='Anomaly', density=True)
        ax.axvline(self.threshold, color='red', linestyle='--', 
                  linewidth=2, label=f'Threshold ({self.threshold:.3f})')
        ax.set_xlabel('Anomaly Score')
        ax.set_ylabel('Density')
        ax.set_title('Anomaly Score Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_threshold_impact(self):
        """
        Show threshold impact on metrics
        """
        thresholds = np.linspace(self.y_scores.min(), self.y_scores.max(), 100)
        precisions, recalls, f1_scores = [], [], []
        
        for thresh in thresholds:
            y_pred_temp = (self.y_scores >= thresh).astype(int)
            
            from sklearn.metrics import precision_score, recall_score, f1_score
            precisions.append(precision_score(self.y_true, y_pred_temp, zero_division=0))
            recalls.append(recall_score(self.y_true, y_pred_temp, zero_division=0))
            f1_scores.append(f1_score(self.y_true, y_pred_temp, zero_division=0))
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(thresholds, precisions, label='Precision', linewidth=2)
        ax.plot(thresholds, recalls, label='Recall', linewidth=2)
        ax.plot(thresholds, f1_scores, label='F1-Score', linewidth=2)
        ax.axvline(self.threshold, color='red', linestyle='--', 
                  label=f'Selected Threshold ({self.threshold:.3f})')
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Score')
        ax.set_title('Threshold Impact on Metrics')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
    
    def generate_full_report(self, save_path: str = None):
        """
        Generate complete report with all visualizations
        """
        fig = plt.figure(figsize=(16, 12))
        
        # ROC Curve
        ax1 = plt.subplot(2, 3, 1)
        self.plot_roc_curve(ax1)
        
        # PR Curve
        ax2 = plt.subplot(2, 3, 2)
        self.plot_pr_curve(ax2)
        
        # Score Distribution
        ax3 = plt.subplot(2, 3, 3)
        self.plot_score_distribution(ax3)
        
        # Confusion Matrix
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        ax4 = plt.subplot(2, 3, 4)
        cm = confusion_matrix(self.y_true, self.y_pred)
        ConfusionMatrixDisplay(cm, display_labels=['Normal', 'Anomaly']).plot(ax=ax4)
        ax4.set_title('Confusion Matrix')
        
        # Threshold Impact
        ax5 = plt.subplot(2, 3, (5, 6))
        thresholds = np.linspace(self.y_scores.min(), self.y_scores.max(), 100)
        precisions, recalls, f1_scores = [], [], []
        
        for thresh in thresholds:
            y_pred_temp = (self.y_scores >= thresh).astype(int)
            from sklearn.metrics import precision_score, recall_score, f1_score
            precisions.append(precision_score(self.y_true, y_pred_temp, zero_division=0))
            recalls.append(recall_score(self.y_true, y_pred_temp, zero_division=0))
            f1_scores.append(f1_score(self.y_true, y_pred_temp, zero_division=0))
        
        ax5.plot(thresholds, precisions, label='Precision', linewidth=2)
        ax5.plot(thresholds, recalls, label='Recall', linewidth=2)
        ax5.plot(thresholds, f1_scores, label='F1-Score', linewidth=2)
        ax5.axvline(self.threshold, color='red', linestyle='--', 
                   label=f'Selected Threshold')
        ax5.set_xlabel('Threshold')
        ax5.set_ylabel('Score')
        ax5.set_title('Threshold Impact')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # Text metrics
        metrics = self.compute_metrics()
        costs = self.compute_confusion_matrix_costs()
        
        print("\n" + "="*60)
        print("ANOMALY DETECTION EVALUATION REPORT")
        print("="*60)
        print(f"\n📊 PERFORMANCE METRICS:")
        print(f"  • Accuracy      : {metrics['accuracy']:.4f}")
        print(f"  • Precision     : {metrics['precision']:.4f}")
        print(f"  • Recall        : {metrics['recall']:.4f}")
        print(f"  • F1-Score      : {metrics['f1_score']:.4f}")
        print(f"  • ROC-AUC       : {metrics['roc_auc']:.4f}")
        print(f"  • PR-AUC        : {metrics['pr_auc']:.4f}")
        print(f"  • MCC           : {metrics['mcc']:.4f}")
        print(f"  • Cohen's Kappa : {metrics['cohen_kappa']:.4f}")
        
        print(f"\n🎯 CONFUSION MATRIX:")
        print(f"  • True Positives  : {costs['true_positives']}")
        print(f"  • True Negatives  : {costs['true_negatives']}")
        print(f"  • False Positives : {costs['false_positives']}")
        print(f"  • False Negatives : {costs['false_negatives']}")
        
        print(f"\n💰 COST ANALYSIS:")
        print(f"  • Total Cost        : {costs['total_cost']:.2f}")
        print(f"  • Avg Cost/Sample   : {costs['avg_cost_per_sample']:.4f}")
        
        print(f"\n⚙️  CONFIGURATION:")
        print(f"  • Optimal Threshold : {self.threshold:.4f}")
        print(f"  • Total Samples     : {len(self.y_true)}")
        print(f"  • Anomaly Rate      : {(self.y_true.sum() / len(self.y_true)):.2%}")
        print("="*60 + "\n")

# Complete usage example
evaluator = AnomalyMetricsEvaluator(
    y_true=y_test,
    y_scores=probas  # Or scores from detector
)

# Complete report
metrics = evaluator.compute_metrics()
costs = evaluator.compute_confusion_matrix_costs(cost_fp=1.0, cost_fn=20.0)
evaluator.generate_full_report(save_path='anomaly_detection_report.png')
```

## Visualization of Anomalies

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_anomalies(X, y_pred, title="Anomaly Detection"):
    """
    Visualize detected anomalies
    """
    plt.figure(figsize=(12, 6))
    
    # Normal points
    normal_mask = y_pred == 1
    plt.scatter(X[normal_mask, 0], X[normal_mask, 1], 
                c='blue', label='Normal', alpha=0.5)
    
    # Anomalies
    anomaly_mask = y_pred == -1
    plt.scatter(X[anomaly_mask, 0], X[anomaly_mask, 1], 
                c='red', label='Anomaly', alpha=0.8, marker='x', s=100)
    
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# Usage example
clf = IsolationForest(contamination=0.1, random_state=42)
y_pred = clf.fit_predict(X_train)
plot_anomalies(X_train, y_pred)
```

## Evaluation Metrics

### Metrics for Imbalanced Data

In anomaly detection problems, classes are heavily imbalanced (anomalies typically <5%). Accuracy is therefore a misleading metric.

## Use Cases

### 1. Banking Fraud Detection

```python
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

def detect_fraud(transactions):
    """
    Detect fraudulent transactions
    """
    # Normalization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(transactions)
    
    # Detection
    clf = IsolationForest(contamination=0.01, random_state=42)
    predictions = clf.fit_predict(X_scaled)
    
    # -1 for anomaly, 1 for normal
    fraud_mask = predictions == -1
    return fraud_mask
```

### 2. Network Intrusion Detection

```python
def detect_network_intrusion(network_data):
    """
    Detect network intrusions
    """
    # Select important features
    features = ['packet_size', 'connection_duration', 'bytes_transferred']
    X = network_data[features]
    
    # One-Class SVM
    svm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)
    predictions = svm.fit_predict(X)
    
    return predictions == -1
```

### 3. Industrial Equipment Monitoring

```python
def monitor_equipment(sensor_data, window_size=50):
    """
    Monitor equipment status via sensors
    """
    # Autoencoder for time series
    autoencoder = build_lstm_autoencoder(timesteps=window_size, 
                                         n_features=sensor_data.shape[1])
    
    # Train on normal data
    autoencoder.fit(sensor_data, sensor_data, epochs=100, verbose=0)
    
    # Anomaly detection
    reconstructions = autoencoder.predict(sensor_data)
    mse = np.mean(np.power(sensor_data - reconstructions, 2), axis=1)
    
    threshold = np.percentile(mse, 99)
    return mse > threshold
```

## Best Practices

### 1. Data Preprocessing
- **Normalization**: Standardize features to avoid bias
- **Missing values**: Imputation or removal
- **Feature engineering**: Create relevant features

### 2. Model Selection
- **Tabular data**: Isolation Forest, One-Class SVM, LOF
- **Time series**: LSTM Autoencoder, Prophet
- **Images**: CNN Autoencoder, VAE
- **Unsupervised data**: Clustering (DBSCAN, K-Means)

### 3. Threshold Definition
```python
def find_optimal_threshold(scores, percentile=95):
    """
    Find optimal threshold for classification
    """
    threshold = np.percentile(scores, percentile)
    return threshold
```

### 4. Handling Imbalance
Anomalies are often rare (1-5% of data). Use:
- **Contamination**: Parameter to specify expected anomaly rate
- **Adapted metrics**: Prefer F1-Score and ROC-AUC over accuracy

## Challenges and Limitations

### 1. **False Positives**
Too many alerts can overwhelm operators. Optimize detection threshold.

### 2. **Adaptation to New Patterns**
Models must be regularly retrained to adapt to changes.

### 3. **Interpretability**
Deep learning models are often hard to interpret. Use techniques like SHAP for explainability.

### 4. **Imbalanced Data**
Anomalies are rare by nature. Use data augmentation techniques or robust models.

## Complete Project: Transaction Anomaly Detection

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

class AnomalyDetector:
    def __init__(self, contamination=0.1):
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.model = None
        
    def preprocess(self, data):
        """Data preprocessing"""
        # Normalization
        data_scaled = self.scaler.fit_transform(data)
        return data_scaled
    
    def train(self, X_train):
        """Train the model"""
        X_scaled = self.preprocess(X_train)
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=100
        )
        self.model.fit(X_scaled)
        
    def predict(self, X):
        """Predict anomalies"""
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        return predictions
    
    def get_anomaly_scores(self, X):
        """Get anomaly scores"""
        X_scaled = self.scaler.transform(X)
        scores = self.model.score_samples(X_scaled)
        return scores
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model"""
        predictions = self.predict(X_test)
        # Convert: -1 -> 1 (anomaly), 1 -> 0 (normal)
        y_pred = (predictions == -1).astype(int)
        
        print(classification_report(y_test, y_pred, 
                                    target_names=['Normal', 'Anomaly']))

# Usage example
np.random.seed(42)
n_samples = 1000
n_features = 5

# Normal data
X_normal = np.random.randn(n_samples, n_features)

# Anomalies
n_anomalies = 50
X_anomalies = np.random.uniform(low=-5, high=5, size=(n_anomalies, n_features))

# Combine
X = np.vstack([X_normal, X_anomalies])
y = np.hstack([np.zeros(n_samples), np.ones(n_anomalies)])

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Detector
detector = AnomalyDetector(contamination=0.1)
detector.train(X_train)

# Evaluation
detector.evaluate(X_test, y_test)
```

## Additional Resources

### Python Libraries
- **scikit-learn**: Classical algorithms (Isolation Forest, One-Class SVM, LOF)
- **PyOD**: Library dedicated to anomaly detection
- **TensorFlow/Keras**: Deep learning (autoencoders, LSTM)
- **Prophet**: Anomaly detection in time series

### Installation
```bash
pip install scikit-learn pyod tensorflow pandas numpy matplotlib seaborn
```

### Public Datasets
- **Credit Card Fraud**: Kaggle
- **KDD Cup 99**: Network intrusion detection
- **UNSW-NB15**: Intrusion detection
- **NAB** (Numenta Anomaly Benchmark): Time series

## Conclusion

Anomaly detection is a crucial field of machine learning with numerous practical applications. The choice of method depends on data type, business context, and performance constraints.

**Key takeaways:**
- Understand business context to define what an anomaly is
- Choose algorithm adapted to data type
- Optimize detection threshold to balance precision and recall
- Regularly validate and monitor performance
- Maintain sufficient interpretability for operational action

Anomaly detection continues to evolve with emerging deep learning techniques and hybrid methods, offering increasingly powerful solutions adapted to specific needs of each domain.
