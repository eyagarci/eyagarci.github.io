---
title: "Ensemble Learning Models"
date:   2025-11-30 03:00:00
categories: [Machine Learning]
tags: [Machine-Learning, Ensemble-Learning, Random-Forest, Boosting, Bagging]  
image:
  path: /assets/imgs/headers/ensemble_learning.png
---

## Introduction

Ensemble learning is a powerful machine learning paradigm that combines multiple models to produce better predictions than any individual model. The key principle is that a group of "weak learners" can come together to form a "strong learner," achieving superior performance through diversity and collective decision-making.

## Core Concepts

### Objective

Ensemble learning involves training multiple models and combining their predictions to make final decisions. This approach leverages the wisdom of crowds principle: diverse models make different errors, and by aggregating their predictions, we can reduce overall error.

### Advantages of Ensemble-Based Models

- **Reduced Variance**: Combining predictions smooths out individual model fluctuations
- **Reduced Bias**: Boosting methods can reduce systematic errors
- **Improved Accuracy**: Ensemble methods consistently outperform single models
- **Robustness**: Less sensitive to noise and outliers in training data
- **Handling Complex Relationships**: Different models capture different patterns

## Comparison of Ensemble Methods

### Algorithm Comparison Table

| Algorithm | Training Speed | Prediction Speed | Memory Usage | Accuracy | Overfitting Risk | Hyperparameter Sensitivity | Best Use Case |
|-----------|---------------|------------------|--------------|----------|------------------|---------------------------|---------------|
| **Random Forest** | Fast | Fast | Medium | High | Low | Low | General purpose, feature importance |
| **AdaBoost** | Medium | Fast | Low | Medium-High | Medium | Medium | Imbalanced data, weak learners |
| **Gradient Boosting** | Slow | Fast | Medium | Very High | High | High | Maximum accuracy, small datasets |
| **XGBoost** | Fast | Fast | Medium | Very High | Medium | High | Kaggle competitions, structured data |
| **LightGBM** | Very Fast | Very Fast | Low | Very High | Medium | High | Large datasets, speed critical |
| **CatBoost** | Medium | Fast | Medium | Very High | Low | Low | Categorical features, robust default |
| **Stacking** | Very Slow | Medium | High | Highest | High | Very High | Competitions, maximum performance |
| **Voting** | Medium | Fast | Medium | High | Low | Low | Quick ensemble, model diversity |

### Performance Characteristics

| Method | Parallel Training | Handles Missing Data | Handles Categories | Feature Importance | Interpretability |
|--------|------------------|----------------------|-------------------|-------------------|------------------|
| **Random Forest** | ✅ Yes | ✅ Yes | ⚠️ Manual | ✅ Yes | Medium |
| **AdaBoost** | ❌ No | ❌ No | ❌ No | ✅ Yes | Medium |
| **Gradient Boosting** | ❌ No | ❌ No | ❌ No | ✅ Yes | Low |
| **XGBoost** | ✅ Yes | ✅ Yes | ⚠️ Manual | ✅ Yes | Low |
| **LightGBM** | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes | Low |
| **CatBoost** | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes | Medium |
| **Stacking** | ✅ Yes | Depends | Depends | ❌ No | Very Low |
| **Voting** | ✅ Yes | Depends | Depends | ❌ No | Medium |

### Ensemble Architecture Diagrams

#### Bagging Architecture
```
Training Data
     |
     ├─→ Bootstrap Sample 1 → Model 1 ─┐
     ├─→ Bootstrap Sample 2 → Model 2 ─┤
     ├─→ Bootstrap Sample 3 → Model 3 ─┼→ Aggregate → Final Prediction
     ├─→ Bootstrap Sample 4 → Model 4 ─┤   (Average/Vote)
     └─→ Bootstrap Sample N → Model N ─┘
```

#### Boosting Architecture
```
Training Data → Model 1 → Reweight Errors → Model 2 → Reweight Errors → Model 3 → ... → Model N
                  ↓                           ↓                           ↓              ↓
                Weight α₁                   Weight α₂                   Weight α₃      Weight αₙ
                  ↓                           ↓                           ↓              ↓
                  └───────────────────────────┴───────────────────────────┴──────────────┴→ Weighted Sum → Prediction
```

#### Stacking Architecture
```
                    Training Data
                          |
        ┌─────────────────┼─────────────────┐
        ↓                 ↓                 ↓
    Model 1           Model 2           Model 3         ← Level 0 (Base Models)
    (Random Forest)   (XGBoost)         (SVM)             (Diverse Algorithms)
        |                 |                 |
        └─────────────────┼─────────────────┘
                          ↓
                   [Pred1, Pred2, Pred3]                ← Meta-Features
                          ↓
                    Meta-Model                          ← Level 1 (Meta-Learner)
                  (Logistic Regression)
                          ↓
                  Final Prediction
```

#### Voting Architecture
```
                    Input Data
                         |
        ┌────────────────┼────────────────┐
        ↓                ↓                ↓
    Classifier 1    Classifier 2    Classifier 3
        |                |                |
        ↓                ↓                ↓
    Class A          Class B          Class A
        |                |                |
        └────────────────┼────────────────┘
                         ↓
              Hard Voting: Majority Vote
              Soft Voting: Average Probabilities
                         ↓
                  Final Prediction
```

## Types of Ensemble Methods

### 1. Bagging (Bootstrap Aggregating)

**Concept**: Train multiple models independently on different random subsets of data (with replacement), then average their predictions.

**Key Characteristics**:
- Reduces variance
- Models trained in parallel
- Works well with high-variance models
- Each model sees a different view of the data

**Popular Algorithms**:

#### Random Forest
- Extension of bagging for decision trees
- Adds randomness in feature selection at each split
- Extremely popular for classification and regression
- Handles high-dimensional data well

**Random Forest Architecture Visualization:**

```
                    Original Dataset (N samples)
                            |
        ┌───────────────────┼───────────────────┐
        │                   │                   │
    Bootstrap            Bootstrap          Bootstrap
    Sample 1             Sample 2           Sample N
    (with                (with              (with
    replacement)         replacement)       replacement)
        │                   │                   │
        ▼                   ▼                   ▼
    Decision             Decision           Decision
    Tree 1               Tree 2             Tree N
    (random              (random            (random
    features)            features)          features)
        │                   │                   │
        └───────────────────┼───────────────────┘
                            ▼
                    Majority Vote
                    (Classification)
                       or
                    Average
                    (Regression)
                            ▼
                    Final Prediction
```

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
rf.fit(X_train, y_train)
predictions = rf.predict(X_test)
```

**Advantages**:
- Robust to overfitting
- Handles missing values
- Provides feature importance
- Works well out-of-the-box

### 2. Boosting

**Concept**: Train models sequentially, where each new model focuses on correcting errors made by previous models.

**Key Characteristics**:
- Reduces bias and variance
- Sequential training process
- Models are weighted based on performance
- Focus on hard-to-predict instances

#### AdaBoost (Adaptive Boosting)
- Adjusts weights of misclassified instances
- Subsequent models focus more on difficult cases
- Combines weak learners into strong learner

```python
from sklearn.ensemble import AdaBoostClassifier

ada = AdaBoostClassifier(
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)
ada.fit(X_train, y_train)
```

#### Gradient Boosting
- Uses gradient descent to minimize loss
- Builds trees to predict residual errors
- Highly accurate but can overfit

```python
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
gb.fit(X_train, y_train)
```

#### XGBoost (Extreme Gradient Boosting)
- Optimized gradient boosting implementation
- Regularization to prevent overfitting
- Parallel processing for speed
- Handles missing values automatically

**XGBoost Learning Process:**

```
Iteration 1:  Fit Tree 1 → Predictions₁ → Residuals₁ = Actual - Predictions₁
                ↓
Iteration 2:  Fit Tree 2 on Residuals₁ → Predictions₂ → Residuals₂
                ↓
Iteration 3:  Fit Tree 3 on Residuals₂ → Predictions₃ → Residuals₃
                ↓
              ...
                ↓
Final Model:  Σ(Learning_Rate × Tree_i predictions)

Key Features:
• Parallel Tree Construction within each iteration
• L1 & L2 Regularization
• Sparsity-aware Split Finding
• Cache-aware Block Structure
```

```python
import xgboost as xgb

xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb_model.fit(X_train, y_train)
```

#### LightGBM
- Leaf-wise tree growth strategy
- Faster training speed
- Lower memory usage
- Better accuracy on large datasets

```python
import lightgbm as lgb

lgb_model = lgb.LGBMClassifier(
    n_estimators=100,
    max_depth=-1,
    learning_rate=0.1,
    random_state=42
)
lgb_model.fit(X_train, y_train)
```

#### CatBoost
- Handles categorical features automatically
- Ordered boosting to reduce overfitting
- Robust to hyperparameter tuning
- Built-in cross-validation

```python
from catboost import CatBoostClassifier

cat_model = CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    depth=6,
    random_state=42,
    verbose=False
)
cat_model.fit(X_train, y_train, cat_features=categorical_indices)
```

### 3. Stacking (Stacked Generalization)

**Concept**: Train multiple base models, then use their predictions as features for a meta-model that makes the final prediction.

**Architecture**:
- **Level 0**: Base models (diverse algorithms)
- **Level 1**: Meta-model learns to combine base predictions

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

base_models = [
    ('rf', RandomForestClassifier(n_estimators=10)),
    ('dt', DecisionTreeClassifier()),
    ('svm', SVC(probability=True))
]

stacking_model = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(),
    cv=5
)
stacking_model.fit(X_train, y_train)
```

**Advantages**:
- Leverages strengths of different algorithms
- Can achieve state-of-the-art performance
- Flexible architecture

**Challenges**:
- Computationally expensive
- Risk of overfitting
- Requires careful cross-validation

### 4. Voting

**Concept**: Combine predictions from multiple models using voting (classification) or averaging (regression).

#### Hard Voting
- Each model votes for a class
- Final prediction is the majority vote

#### Soft Voting
- Models predict class probabilities
- Average probabilities for final prediction
- Generally more reliable than hard voting

```python
from sklearn.ensemble import VotingClassifier

voting_clf = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression()),
        ('rf', RandomForestClassifier()),
        ('svm', SVC(probability=True))
    ],
    voting='soft'
)
voting_clf.fit(X_train, y_train)
```

## Additional Ensemble Techniques

### 5. Extra Trees (Extremely Randomized Trees)

**Concept**: Similar to Random Forest but with more randomness in tree construction.

**Key Differences from Random Forest**:
- Splits are chosen randomly (not optimally)
- Uses entire dataset (no bootstrap sampling)
- Faster training due to random splits
- Can reduce variance further

```python
from sklearn.ensemble import ExtraTreesClassifier

# Extra Trees Classifier
extra_trees = ExtraTreesClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    random_state=42,
    n_jobs=-1
)

extra_trees.fit(X_train, y_train)
print(f"Extra Trees Accuracy: {extra_trees.score(X_test, y_test):.4f}")

# For regression
from sklearn.ensemble import ExtraTreesRegressor

extra_trees_reg = ExtraTreesRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
```

**When to Use Extra Trees**:
- Similar use cases as Random Forest
- When you want faster training
- When you have high-dimensional data
- When you want to reduce variance even more

**Comparison: Random Forest vs Extra Trees**

```python
import time

# Compare performance
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Extra Trees': ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)
}

for name, model in models.items():
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    
    accuracy = model.score(X_test, y_test)
    
    print(f"{name}:")
    print(f"  Training Time: {train_time:.3f}s")
    print(f"  Accuracy: {accuracy:.4f}")
```

### 6. Isolation Forest (Anomaly Detection)

**Concept**: Ensemble method specifically designed for anomaly detection using isolation.

**How It Works**:
- Anomalies are easier to isolate than normal points
- Uses random splits to build trees
- Anomalies require fewer splits to isolate
- Assigns anomaly score based on path length

```python
from sklearn.ensemble import IsolationForest
import numpy as np

# Generate dataset with outliers
from sklearn.datasets import make_blobs

X_normal, _ = make_blobs(n_samples=1000, centers=1, random_state=42)
X_outliers = np.random.uniform(low=-10, high=10, size=(50, 2))
X_combined = np.vstack([X_normal, X_outliers])

# Train Isolation Forest
iso_forest = IsolationForest(
    n_estimators=100,
    contamination=0.05,  # Expected proportion of outliers
    random_state=42,
    n_jobs=-1
)

iso_forest.fit(X_combined)

# Predict anomalies (-1 for outliers, 1 for inliers)
predictions = iso_forest.predict(X_combined)
anomaly_scores = iso_forest.score_samples(X_combined)

print(f"Number of detected anomalies: {(predictions == -1).sum()}")
print(f"Expected anomalies: {int(0.05 * len(X_combined))}")

# Visualize results
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(X_combined[:, 0], X_combined[:, 1], 
           c=predictions, cmap='coolwarm', 
           s=50, edgecolors='k', alpha=0.6)
plt.colorbar(label='Prediction (-1: Anomaly, 1: Normal)')
plt.title('Isolation Forest Anomaly Detection')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
```

**Applications**:
- Fraud detection in financial transactions
- Network intrusion detection
- Manufacturing defect detection
- Healthcare anomaly detection

**Advantages**:
- Works well with high-dimensional data
- Linear time complexity
- No need for labeled anomaly data
- Handles large datasets efficiently

### 7. Deep Ensemble Learning (Neural Networks)

**Concept**: Ensemble methods applied to deep neural networks.

#### Simple Neural Network Ensemble

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def create_neural_network(input_dim, dropout_rate=0.3):
    """Create a simple neural network"""
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        layers.Dropout(dropout_rate),
        layers.Dense(64, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Train multiple neural networks with different initializations
n_models = 5
nn_ensemble = []

print("Training Neural Network Ensemble...")
for i in range(n_models):
    print(f"Training model {i+1}/{n_models}...")
    
    # Create model with different random seed
    model = create_neural_network(X_train.shape[1])
    
    # Train with different random initialization
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=0,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        ]
    )
    
    nn_ensemble.append(model)
    test_acc = model.evaluate(X_test, y_test, verbose=0)[1]
    print(f"  Individual model accuracy: {test_acc:.4f}")

# Ensemble predictions (average probabilities)
ensemble_predictions = np.zeros(len(X_test))

for model in nn_ensemble:
    predictions = model.predict(X_test, verbose=0).flatten()
    ensemble_predictions += predictions

ensemble_predictions /= n_models
ensemble_predictions_binary = (ensemble_predictions > 0.5).astype(int)

# Evaluate ensemble
from sklearn.metrics import accuracy_score
ensemble_acc = accuracy_score(y_test, ensemble_predictions_binary)

print(f"\nNeural Network Ensemble Accuracy: {ensemble_acc:.4f}")
```

#### Snapshot Ensemble for Neural Networks

```python
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint

def cosine_annealing(epoch, initial_lr=0.1, epochs_per_cycle=10):
    """Cosine annealing learning rate schedule"""
    cycle = np.floor(epoch / epochs_per_cycle)
    x = np.pi * (epoch % epochs_per_cycle) / epochs_per_cycle
    lr = initial_lr * (np.cos(x) + 1) / 2
    return lr

# Train with cyclic learning rate and save snapshots
model = create_neural_network(X_train.shape[1])

# Callbacks
lr_scheduler = LearningRateScheduler(cosine_annealing)
checkpointer = ModelCheckpoint(
    'snapshot_{epoch:02d}.h5',
    save_best_only=False,
    save_freq='epoch'
)

# Train
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[lr_scheduler],
    verbose=0
)

print("Snapshot ensemble saves models at different learning rate cycles")
print("Creates diverse models from a single training run")
```

#### Monte Carlo Dropout Ensemble

```python
def create_mc_dropout_model(input_dim):
    """Neural network with dropout enabled during inference"""
    inputs = keras.Input(shape=(input_dim,))
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.Dropout(0.5)(x, training=True)  # Keep dropout active during inference
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x, training=True)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Train model
mc_model = create_mc_dropout_model(X_train.shape[1])
mc_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

# Make multiple predictions with dropout enabled
n_iterations = 100
mc_predictions = np.zeros((n_iterations, len(X_test)))

for i in range(n_iterations):
    predictions = mc_model.predict(X_test, verbose=0).flatten()
    mc_predictions[i] = predictions

# Average predictions and compute uncertainty
mean_predictions = mc_predictions.mean(axis=0)
std_predictions = mc_predictions.std(axis=0)

print("Monte Carlo Dropout provides:")
print("- Ensemble predictions without training multiple models")
print("- Uncertainty estimates for each prediction")
print(f"Average prediction std: {std_predictions.mean():.4f}")
```

**Deep Ensemble Best Practices**:
- Use different architectures for diversity
- Vary initialization seeds
- Apply different regularization techniques
- Consider computational cost vs. performance gain

## Advanced Techniques

### Blending
- Similar to stacking but simpler
- Hold-out validation set for meta-model
- Less prone to overfitting than stacking
- Faster to implement

### Cascading
- Models arranged in sequence
- Each level can reject uncertain predictions
- Following levels handle rejected cases
- Balances accuracy and computational cost

### Snapshot Ensembles
- Single training run with cyclic learning rate
- Save model at each learning rate cycle
- Ensemble of models from different local minima
- Memory efficient

## Complete End-to-End Example

Let's walk through a complete ensemble learning project from data loading to final evaluation.

```python
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
import warnings
warnings.filterwarnings('ignore')

# Step 1: Create or Load Dataset
print("Step 1: Creating synthetic dataset...")
X, y = make_classification(
    n_samples=10000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=2,
    weights=[0.7, 0.3],  # Imbalanced classes
    flip_y=0.01,
    random_state=42
)

# Convert to DataFrame for better handling
feature_names = [f'feature_{i}' for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

print(f"Dataset shape: {df.shape}")
print(f"Class distribution:\n{df['target'].value_counts()}")
print(f"\nFirst few rows:\n{df.head()}")

# Step 2: Split the data
print("\nStep 2: Splitting data into train, validation, and test sets...")
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
)

print(f"Training set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")
print(f"Test set: {X_test.shape}")

# Step 3: Feature Scaling
print("\nStep 3: Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Step 4: Train Individual Models
print("\nStep 4: Training individual models...")

# Model 1: Random Forest
print("  Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_score = rf_model.score(X_val, y_val)
print(f"  Random Forest validation accuracy: {rf_score:.4f}")

# Model 2: Gradient Boosting
print("  Training Gradient Boosting...")
gb_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
gb_model.fit(X_train, y_train)
gb_score = gb_model.score(X_val, y_val)
print(f"  Gradient Boosting validation accuracy: {gb_score:.4f}")

# Model 3: Logistic Regression (on scaled data)
print("  Training Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)
lr_score = lr_model.score(X_val_scaled, y_val)
print(f"  Logistic Regression validation accuracy: {lr_score:.4f}")

# Model 4: Support Vector Machine (on scaled data)
print("  Training SVM...")
svm_model = SVC(kernel='rbf', probability=True, random_state=42)
svm_model.fit(X_train_scaled, y_train)
svm_score = svm_model.score(X_val_scaled, y_val)
print(f"  SVM validation accuracy: {svm_score:.4f}")

# Step 5: Create Voting Ensemble
print("\nStep 5: Creating Voting Ensemble...")
voting_clf = VotingClassifier(
    estimators=[
        ('rf', rf_model),
        ('gb', gb_model),
        ('lr', lr_model),
        ('svm', svm_model)
    ],
    voting='soft',
    n_jobs=-1
)

# Note: Voting classifier needs consistent input, so we use a custom prediction
# For simplicity, let's create a version that uses scaled data for LR and SVM
print("  Training Voting Ensemble...")
# We'll train on original data but this is simplified
voting_simple = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42))
    ],
    voting='soft',
    n_jobs=-1
)
voting_simple.fit(X_train, y_train)
voting_score = voting_simple.score(X_val, y_val)
print(f"  Voting Ensemble validation accuracy: {voting_score:.4f}")

# Step 6: Create Stacking Ensemble
print("\nStep 6: Creating Stacking Ensemble...")
stacking_clf = StackingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, random_state=42))
    ],
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5,
    n_jobs=-1
)
stacking_clf.fit(X_train, y_train)
stacking_score = stacking_clf.score(X_val, y_val)
print(f"  Stacking Ensemble validation accuracy: {stacking_score:.4f}")

# Step 7: Final Evaluation on Test Set
print("\nStep 7: Evaluating on test set...")
print("\n" + "="*60)
print("FINAL TEST SET RESULTS")
print("="*60)

models = {
    'Random Forest': rf_model,
    'Gradient Boosting': gb_model,
    'Voting Ensemble': voting_simple,
    'Stacking Ensemble': stacking_clf
}

for name, model in models.items():
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    print(f"\n{name}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  ROC AUC: {auc:.4f}")

# Step 8: Detailed Analysis of Best Model
print("\n" + "="*60)
print("DETAILED ANALYSIS - Stacking Ensemble")
print("="*60)

y_pred_final = stacking_clf.predict(X_test)
y_proba_final = stacking_clf.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred_final))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred_final)
print(cm)

# Step 9: Feature Importance (from Random Forest)
print("\nTop 10 Most Important Features (from Random Forest):")
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance.head(10))

print("\n" + "="*60)
print("End-to-End Example Completed Successfully!")
print("="*60)
```

### Output Example:
```
Step 1: Creating synthetic dataset...
Dataset shape: (10000, 21)
Class distribution:
0    7000
1    3000

Training set: (6400, 20)
Validation set: (1600, 20)
Test set: (2000, 20)

Random Forest validation accuracy: 0.9456
Gradient Boosting validation accuracy: 0.9500
Voting Ensemble validation accuracy: 0.9531
Stacking Ensemble validation accuracy: 0.9562

FINAL TEST SET RESULTS
Random Forest: Accuracy: 0.9435, ROC AUC: 0.9812
Gradient Boosting: Accuracy: 0.9485, ROC AUC: 0.9845
Voting Ensemble: Accuracy: 0.9520, ROC AUC: 0.9868
Stacking Ensemble: Accuracy: 0.9565, ROC AUC: 0.9891
```

### Using Real-World Dataset (Optional)

```python
# Example with Iris dataset
from sklearn.datasets import load_breast_cancer

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

print(f"Dataset: {data.DESCR[:200]}...")
print(f"Features: {data.feature_names}")
print(f"Classes: {data.target_names}")

# Apply the same pipeline as above...
```

## Hyperparameter Tuning for Ensembles

### Key Parameters

**For Random Forest**:
- `n_estimators`: Number of trees (more is generally better)
- `max_depth`: Maximum tree depth
- `max_features`: Features considered for splitting
- `min_samples_split`: Minimum samples to split node

**For Gradient Boosting**:
- `learning_rate`: Step size shrinkage (0.01-0.3)
- `n_estimators`: Number of boosting stages
- `max_depth`: Tree depth (3-10 typical)
- `subsample`: Fraction of samples for fitting

### Tuning Strategies

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3]
}

grid_search = GridSearchCV(
    GradientBoostingClassifier(),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
```

## Practical Considerations

### Performance and Resource Management

#### 1. Training Time Comparison

```python
import time
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Benchmark different algorithms
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, n_jobs=-1, random_state=42),
    'LightGBM': LGBMClassifier(n_estimators=100, n_jobs=-1, random_state=42, verbose=-1),
    'CatBoost': CatBoostClassifier(iterations=100, random_state=42, verbose=False)
}

results = []

for name, model in models.items():
    # Training time
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Prediction time
    start_time = time.time()
    predictions = model.predict(X_test)
    prediction_time = time.time() - start_time
    
    # Accuracy
    accuracy = model.score(X_test, y_test)
    
    results.append({
        'Model': name,
        'Training Time (s)': round(training_time, 3),
        'Prediction Time (s)': round(prediction_time, 4),
        'Accuracy': round(accuracy, 4)
    })

# Display results
df_results = pd.DataFrame(results)
print("\nPerformance Comparison:")
print("="*80)
print(df_results.to_string(index=False))
print("="*80)

# Visualize training time
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.barh(df_results['Model'], df_results['Training Time (s)'])
plt.xlabel('Training Time (seconds)')
plt.title('Training Time Comparison')
plt.tight_layout()

plt.subplot(1, 2, 2)
plt.barh(df_results['Model'], df_results['Accuracy'], color='green')
plt.xlabel('Accuracy')
plt.title('Model Accuracy')
plt.xlim(0.8, 1.0)
plt.tight_layout()
plt.show()
```

**Typical Results (10,000 samples, 20 features):**
```
Model               Training Time (s)    Prediction Time (s)    Accuracy
Random Forest            2.145                0.0234              0.9435
Gradient Boosting        8.732                0.0089              0.9485
AdaBoost                 3.421                0.0156              0.9125
XGBoost                  1.234                0.0067              0.9520
LightGBM                 0.456                0.0045              0.9515
CatBoost                 4.567                0.0098              0.9535
```

#### 2. Memory Usage Analysis

```python
import sys
import psutil
import os

def get_model_memory_size(model):
    """Estimate memory size of a trained model"""
    import pickle
    
    # Serialize model to bytes
    pickled = pickle.dumps(model)
    size_bytes = len(pickled)
    size_mb = size_bytes / (1024 * 1024)
    
    return size_mb

def monitor_memory_usage(func):
    """Decorator to monitor memory usage during training"""
    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())
        
        # Memory before
        mem_before = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Memory after
        mem_after = process.memory_info().rss / (1024 * 1024)  # MB
        mem_used = mem_after - mem_before
        
        print(f"Memory used during training: {mem_used:.2f} MB")
        print(f"Total memory: {mem_after:.2f} MB")
        
        return result
    return wrapper

@monitor_memory_usage
def train_model(model, X, y):
    return model.fit(X, y)

# Test memory usage
print("\nMemory Usage Comparison:")
print("="*60)

for name, model in models.items():
    print(f"\n{name}:")
    trained_model = train_model(model, X_train, y_train)
    model_size = get_model_memory_size(trained_model)
    print(f"Model size on disk: {model_size:.2f} MB")
```

#### 3. Scalability to Large Datasets

##### Using Dask for Distributed Computing

```python
# For very large datasets that don't fit in memory
import dask.dataframe as dd
from dask_ml.ensemble import RandomForestClassifier as DaskRandomForest
from dask_ml.model_selection import train_test_split as dask_train_test_split

# Load large dataset with Dask
# df_large = dd.read_csv('large_dataset.csv')
# X_large = df_large.drop('target', axis=1)
# y_large = df_large['target']

# Train with Dask
# X_train_dask, X_test_dask, y_train_dask, y_test_dask = dask_train_test_split(
#     X_large, y_large, test_size=0.2, random_state=42
# )

# dask_rf = DaskRandomForest(n_estimators=100, random_state=42)
# dask_rf.fit(X_train_dask, y_train_dask)

print("Dask enables training on datasets larger than RAM")
print("Automatically distributes computation across cores/machines")
```

##### Using PySpark MLlib

```python
# For big data with Spark
"""
from pyspark.ml.classification import RandomForestClassifier as SparkRF
from pyspark.ml.classification import GBTClassifier as SparkGBT
from pyspark.sql import SparkSession

# Initialize Spark
spark = SparkSession.builder \
    .appName("EnsembleLearning") \
    .config("spark.executor.memory", "8g") \
    .getOrCreate()

# Load data
df_spark = spark.read.parquet("large_dataset.parquet")

# Train Random Forest
spark_rf = SparkRF(
    featuresCol='features',
    labelCol='label',
    numTrees=100,
    maxDepth=10
)

model = spark_rf.fit(df_spark)
predictions = model.transform(test_df)
"""

print("\nSpark MLlib is ideal for:")
print("- Datasets > 100GB")
print("- Distributed computing clusters")
print("- Production big data pipelines")
```

##### Incremental Learning for Streaming Data

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier

# For data streams, use partial_fit
sgd_clf = SGDClassifier(loss='log_loss', random_state=42)

# Simulate streaming data
batch_size = 1000
for i in range(0, len(X_train), batch_size):
    X_batch = X_train[i:i+batch_size]
    y_batch = y_train[i:i+batch_size]
    
    # Partial fit (incremental learning)
    sgd_clf.partial_fit(X_batch, y_batch, classes=np.unique(y_train))
    
    if (i // batch_size) % 5 == 0:
        accuracy = sgd_clf.score(X_test, y_test)
        print(f"Batch {i//batch_size}: Accuracy = {accuracy:.4f}")

print("\nIncremental learning useful for:")
print("- Online learning scenarios")
print("- Memory-constrained environments")
print("- Continuously arriving data")
```

#### 4. Model Optimization Techniques

##### Feature Selection for Speed

```python
from sklearn.feature_selection import SelectFromModel

# Use feature importance to reduce features
selector = SelectFromModel(
    RandomForestClassifier(n_estimators=100, random_state=42),
    threshold='median'  # Keep features above median importance
)

selector.fit(X_train, y_train)
X_train_reduced = selector.transform(X_train)
X_test_reduced = selector.transform(X_test)

print(f"Original features: {X_train.shape[1]}")
print(f"Selected features: {X_train_reduced.shape[1]}")

# Train on reduced features (faster)
rf_reduced = RandomForestClassifier(n_estimators=100, random_state=42)
rf_reduced.fit(X_train_reduced, y_train)
print(f"Accuracy with reduced features: {rf_reduced.score(X_test_reduced, y_test):.4f}")
```

##### Model Compression

```python
# Reduce model size while maintaining performance

# 1. Reduce number of trees (for Random Forest)
rf_full = RandomForestClassifier(n_estimators=1000, random_state=42)
rf_full.fit(X_train, y_train)

rf_compressed = RandomForestClassifier(n_estimators=50, random_state=42)
rf_compressed.fit(X_train, y_train)

print(f"Full model size: {get_model_memory_size(rf_full):.2f} MB")
print(f"Compressed model size: {get_model_memory_size(rf_compressed):.2f} MB")
print(f"Full model accuracy: {rf_full.score(X_test, y_test):.4f}")
print(f"Compressed model accuracy: {rf_compressed.score(X_test, y_test):.4f}")

# 2. Prune trees (reduce max_depth)
rf_pruned = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,  # Limit depth
    min_samples_leaf=10,  # Require more samples per leaf
    random_state=42
)
rf_pruned.fit(X_train, y_train)
```

##### Parallel Processing

```python
# Leverage multiple cores
from joblib import parallel_backend

# Set number of jobs
with parallel_backend('threading', n_jobs=-1):  # Use all cores
    rf_parallel = RandomForestClassifier(
        n_estimators=100,
        n_jobs=-1,
        random_state=42
    )
    rf_parallel.fit(X_train, y_train)

print("Using n_jobs=-1 enables multi-core training")
print("Especially beneficial for Random Forest and XGBoost")
```

### When to Use Each Method

**Random Forest**:
- Good default choice
- When you need feature importance
- High-dimensional data
- Don't have time for extensive tuning

**XGBoost/LightGBM/CatBoost**:
- Structured/tabular data competitions
- Need maximum accuracy
- Have time for hyperparameter tuning
- Large datasets

**Stacking**:
- Maximum performance required
- Have computational resources
- For competitions or critical applications

**Voting/Blending**:
- Quick ensemble without much tuning
- Combine existing trained models
- When interpretability matters

### Common Pitfalls

1. **Data Leakage**: Ensure proper train/test split before ensemble
2. **Overfitting**: Use cross-validation, regularization
3. **Computational Cost**: Balance performance vs. resources
4. **Diversity**: Ensure base models are sufficiently different
5. **Correlation**: Highly correlated models provide less benefit

### Feature Engineering for Ensembles

- **For Random Forest**: Can handle raw features well
- **For Gradient Boosting**: Benefits from feature engineering
- **For Stacking**: Base models can use different feature sets

## Debugging and Troubleshooting Ensembles

### Common Problems and Solutions

#### Problem 1: Ensemble Performs Worse Than Individual Models

**Symptoms**:
- Ensemble accuracy lower than best individual model
- High correlation between base models

**Diagnosis**:
```python
# Check correlation between model predictions
from scipy.stats import pearsonr

models = [rf_model, gb_model, lr_model]
predictions = np.array([model.predict_proba(X_test)[:, 1] for model in models])

print("Model Prediction Correlations:")
for i in range(len(models)):
    for j in range(i+1, len(models)):
        corr, _ = pearsonr(predictions[i], predictions[j])
        print(f"  Model {i+1} vs Model {j+1}: {corr:.4f}")

# High correlation (>0.9) indicates lack of diversity
```

**Solutions**:
1. **Increase Model Diversity**:
```python
# Use different algorithms
diverse_models = [
    ('rf', RandomForestClassifier(n_estimators=100)),
    ('gb', GradientBoostingClassifier(n_estimators=100)),
    ('svm', SVC(probability=True)),
    ('lr', LogisticRegression()),
    ('knn', KNeighborsClassifier(n_neighbors=5))
]

# Use different feature subsets
from sklearn.feature_selection import SelectKBest, f_classif

selector1 = SelectKBest(f_classif, k=10)
X_train_subset1 = selector1.fit_transform(X_train, y_train)

selector2 = SelectKBest(f_classif, k=15)
X_train_subset2 = selector2.fit_transform(X_train, y_train)
```

2. **Use Different Training Subsets**:
```python
# Train on different data splits
n_samples = len(X_train)
subset_size = int(0.8 * n_samples)

model1 = RandomForestClassifier(random_state=1)
model1.fit(X_train[:subset_size], y_train[:subset_size])

model2 = RandomForestClassifier(random_state=2)
model2.fit(X_train[n_samples-subset_size:], y_train[n_samples-subset_size:])
```

3. **Weighted Voting Based on Performance**:
```python
from sklearn.ensemble import VotingClassifier

# Assign weights based on validation accuracy
voting_weighted = VotingClassifier(
    estimators=[
        ('rf', rf_model),
        ('gb', gb_model),
        ('lr', lr_model)
    ],
    voting='soft',
    weights=[0.4, 0.4, 0.2]  # Based on validation performance
)
```

#### Problem 2: Ensemble is Overfitting

**Symptoms**:
- High training accuracy, low test accuracy
- Large gap between train and validation scores
- Learning curves show divergence

**Diagnosis**:
```python
# Check overfitting
train_score = ensemble_model.score(X_train, y_train)
test_score = ensemble_model.score(X_test, y_test)

print(f"Training Score: {train_score:.4f}")
print(f"Test Score: {test_score:.4f}")
print(f"Gap: {train_score - test_score:.4f}")

if train_score - test_score > 0.1:
    print("⚠️ Model is overfitting!")
```

**Solutions**:
1. **Regularization**:
```python
# For Random Forest
rf_regularized = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,           # Limit tree depth
    min_samples_split=20,  # Require more samples to split
    min_samples_leaf=10,   # Require more samples per leaf
    max_features='sqrt',   # Reduce features per tree
    random_state=42
)

# For Gradient Boosting
gb_regularized = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.01,    # Smaller learning rate
    subsample=0.8,         # Use only 80% of samples
    max_depth=3,           # Shallow trees
    min_samples_leaf=10,
    random_state=42
)
```

2. **Cross-Validation for Stacking**:
```python
# Use cross-validation to prevent overfitting in stacking
stacking_cv = StackingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=50)),
        ('gb', GradientBoostingClassifier(n_estimators=50))
    ],
    final_estimator=LogisticRegression(),
    cv=10,  # Increase cross-validation folds
    n_jobs=-1
)
```

3. **Early Stopping for Boosting**:
```python
# For XGBoost
import xgboost as xgb

xgb_early_stop = xgb.XGBClassifier(
    n_estimators=1000,
    learning_rate=0.1,
    random_state=42
)

xgb_early_stop.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    early_stopping_rounds=50,  # Stop if no improvement for 50 rounds
    verbose=False
)

print(f"Best iteration: {xgb_early_stop.best_iteration}")
```

#### Problem 3: Ensemble Training Takes Too Long

**Symptoms**:
- Training time exceeds acceptable limits
- Need for faster iteration during development

**Solutions**:
1. **Reduce Number of Estimators**:
```python
# Find optimal number of trees
from sklearn.model_selection import learning_curve

train_sizes = [10, 20, 50, 100, 200, 500]
results = []

for n_trees in train_sizes:
    start = time.time()
    rf = RandomForestClassifier(n_estimators=n_trees, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    train_time = time.time() - start
    
    accuracy = rf.score(X_test, y_test)
    results.append((n_trees, train_time, accuracy))

print("\nNumber of Trees vs Performance:")
for n_trees, train_time, acc in results:
    print(f"n_trees={n_trees:3d}: Time={train_time:.2f}s, Accuracy={acc:.4f}")

# Often 100-200 trees is sufficient
```

2. **Use Faster Algorithms**:
```python
# Replace Gradient Boosting with LightGBM
from lightgbm import LGBMClassifier

lgbm_fast = LGBMClassifier(
    n_estimators=100,
    num_leaves=31,
    learning_rate=0.1,
    n_jobs=-1,
    verbose=-1
)

start = time.time()
lgbm_fast.fit(X_train, y_train)
print(f"LightGBM training time: {time.time() - start:.2f}s")
```

3. **Parallel Training**:
```python
# Train base models in parallel for stacking
from joblib import Parallel, delayed

def train_model(model, X, y):
    return model.fit(X, y)

models = [
    RandomForestClassifier(n_estimators=100, random_state=42),
    GradientBoostingClassifier(n_estimators=100, random_state=42),
    LogisticRegression()
]

# Train in parallel
trained_models = Parallel(n_jobs=-1)(
    delayed(train_model)(model, X_train, y_train) for model in models
)
```

#### Problem 4: When to Stop Adding Models

**Decision Framework**:

```python
def evaluate_ensemble_size(base_model_class, max_models=20, step=1):
    """
    Determine optimal number of models in ensemble
    """
    results = {
        'n_models': [],
        'train_score': [],
        'test_score': [],
        'improvement': []
    }
    
    previous_score = 0
    
    for n in range(1, max_models + 1, step):
        # Create ensemble
        if base_model_class == 'rf':
            model = RandomForestClassifier(n_estimators=n, random_state=42, n_jobs=-1)
        elif base_model_class == 'gb':
            model = GradientBoostingClassifier(n_estimators=n, random_state=42)
        
        # Train and evaluate
        model.fit(X_train, y_train)
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        # Calculate improvement
        improvement = test_score - previous_score
        
        results['n_models'].append(n)
        results['train_score'].append(train_score)
        results['test_score'].append(test_score)
        results['improvement'].append(improvement)
        
        previous_score = test_score
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(results['n_models'], results['train_score'], 'b-', label='Train')
    plt.plot(results['n_models'], results['test_score'], 'r-', label='Test')
    plt.xlabel('Number of Models/Trees')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Ensemble Size')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(results['n_models'][1:], results['improvement'][1:], 'g-')
    plt.axhline(y=0.001, color='r', linestyle='--', label='Threshold (0.001)')
    plt.xlabel('Number of Models/Trees')
    plt.ylabel('Improvement over Previous')
    plt.title('Marginal Improvement')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Find optimal point (diminishing returns)
    improvements = np.array(results['improvement'][1:])
    optimal_idx = np.where(improvements < 0.001)[0]
    
    if len(optimal_idx) > 0:
        optimal_n = results['n_models'][optimal_idx[0]]
        print(f"\n✅ Optimal number of models: {optimal_n}")
        print(f"   Test accuracy: {results['test_score'][optimal_idx[0]]:.4f}")
        print(f"   Improvement drops below 0.1% threshold")
    else:
        print(f"\n⚠️ Consider more models, still seeing improvements")
    
    return results

# Use it
results = evaluate_ensemble_size('rf', max_models=200, step=10)
```

**Stopping Criteria**:
1. **Performance Plateau**: Improvement < 0.1% for 3 consecutive additions
2. **Time Budget**: Reached maximum allowed training time
3. **Overfitting**: Test score decreasing while train score increasing
4. **Resource Limits**: Memory or disk space constraints

#### Problem 5: Inconsistent Predictions

**Symptoms**:
- Different predictions on same input
- High prediction variance

**Diagnosis**:
```python
# Check prediction stability
n_runs = 10
predictions_over_runs = []

for i in range(n_runs):
    # Some models have randomness even in prediction
    preds = ensemble_model.predict_proba(X_test)[:, 1]
    predictions_over_runs.append(preds)

predictions_array = np.array(predictions_over_runs)
prediction_std = predictions_array.std(axis=0)

print(f"Average prediction std: {prediction_std.mean():.6f}")
print(f"Max prediction std: {prediction_std.max():.6f}")

# High std indicates instability
if prediction_std.mean() > 0.05:
    print("⚠️ Predictions are unstable!")
```

**Solutions**:
```python
# Fix random seeds everywhere
import random

random.seed(42)
np.random.seed(42)

# For TensorFlow/Keras
import tensorflow as tf
tf.random.set_seed(42)

# Use fixed random_state in all models
rf_stable = RandomForestClassifier(
    n_estimators=100,
    random_state=42,  # Fixed seed
    n_jobs=-1
)
```

### Diagnostic Checklist

Before deploying an ensemble model, verify:

```python
def ensemble_health_check(ensemble_model, X_train, y_train, X_test, y_test):
    """
    Comprehensive health check for ensemble models
    """
    print("="*60)
    print("ENSEMBLE MODEL HEALTH CHECK")
    print("="*60)
    
    # 1. Performance Check
    train_score = ensemble_model.score(X_train, y_train)
    test_score = ensemble_model.score(X_test, y_test)
    gap = train_score - test_score
    
    print(f"\n1. Performance Metrics:")
    print(f"   Train Score: {train_score:.4f}")
    print(f"   Test Score: {test_score:.4f}")
    print(f"   Gap: {gap:.4f}")
    
    if gap > 0.1:
        print("   ⚠️ WARNING: Possible overfitting")
    else:
        print("   ✅ Good generalization")
    
    # 2. Prediction Distribution
    y_pred_proba = ensemble_model.predict_proba(X_test)[:, 1]
    
    print(f"\n2. Prediction Distribution:")
    print(f"   Min: {y_pred_proba.min():.4f}")
    print(f"   Max: {y_pred_proba.max():.4f}")
    print(f"   Mean: {y_pred_proba.mean():.4f}")
    print(f"   Std: {y_pred_proba.std():.4f}")
    
    # 3. Class Balance
    y_pred = ensemble_model.predict(X_test)
    pred_class_dist = np.bincount(y_pred) / len(y_pred)
    true_class_dist = np.bincount(y_test) / len(y_test)
    
    print(f"\n3. Class Distribution:")
    print(f"   True: {true_class_dist}")
    print(f"   Predicted: {pred_class_dist}")
    
    # 4. Model Size
    import pickle
    model_size = len(pickle.dumps(ensemble_model)) / (1024 * 1024)
    print(f"\n4. Model Size: {model_size:.2f} MB")
    
    if model_size > 100:
        print("   ⚠️ WARNING: Large model size")
    
    # 5. Prediction Time
    start = time.time()
    _ = ensemble_model.predict(X_test[:100])
    pred_time = (time.time() - start) / 100 * 1000  # ms per sample
    
    print(f"\n5. Prediction Speed: {pred_time:.2f} ms/sample")
    
    if pred_time > 10:
        print("   ⚠️ WARNING: Slow predictions")
    
    print("\n" + "="*60)

# Run health check
ensemble_health_check(ensemble_model, X_train, y_train, X_test, y_test)
```

## Real-World Applications

### Industry Use Cases

1. **Finance**: Credit scoring, fraud detection, risk assessment
2. **Healthcare**: Disease prediction, patient outcome forecasting
3. **E-commerce**: Recommendation systems, customer churn prediction
4. **Manufacturing**: Predictive maintenance, quality control
5. **Marketing**: Customer segmentation, conversion prediction

### Competition Winners

Ensemble methods dominate machine learning competitions:
- Netflix Prize winner used ensemble of 800+ models
- Kaggle competitions regularly won by ensemble methods
- Most top-scoring solutions use stacking or blending

## Implementation Best Practices

### 1. Cross-Validation Strategy

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    ensemble_model,
    X_train,
    y_train,
    cv=5,
    scoring='accuracy'
)
print(f"CV Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

### 2. Monitor Training

```python
# For gradient boosting
eval_set = [(X_train, y_train), (X_val, y_val)]
model.fit(
    X_train, y_train,
    eval_set=eval_set,
    early_stopping_rounds=10,
    verbose=True
)
```

### 3. Feature Importance Analysis

```python
import matplotlib.pyplot as plt

feature_importance = model.feature_importances_
sorted_idx = feature_importance.argsort()

plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
plt.yticks(range(len(sorted_idx)), X.columns[sorted_idx])
plt.xlabel('Feature Importance')
plt.title('Top Features')
plt.show()
```

### 4. Save and Load Models

```python
import joblib

# Save model
joblib.dump(ensemble_model, 'ensemble_model.pkl')

# Load model
loaded_model = joblib.load('ensemble_model.pkl')
```

## Performance Metrics

### Evaluation

```python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

y_pred = ensemble_model.predict(X_test)
y_pred_proba = ensemble_model.predict_proba(X_test)[:, 1]

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
```

### Advanced Metrics and Diagnostics

#### 1. Confusion Matrix Analysis

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Visualize confusion matrix
fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Class 0', 'Class 1'])
disp.plot(cmap='Blues', ax=ax)
plt.title('Confusion Matrix - Ensemble Model')
plt.show()

# Calculate metrics from confusion matrix
tn, fp, fn, tp = cm.ravel()
print(f"\nTrue Negatives: {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives: {tp}")

# Additional metrics
specificity = tn / (tn + fp)
sensitivity = tp / (tp + fn)
print(f"\nSpecificity: {specificity:.4f}")
print(f"Sensitivity (Recall): {sensitivity:.4f}")
```

#### 2. ROC Curve and AUC

```python
from sklearn.metrics import roc_curve, auc

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()

# Find optimal threshold
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print(f"Optimal Threshold: {optimal_threshold:.4f}")
```

#### 3. Learning Curves for Overfitting Detection

```python
from sklearn.model_selection import learning_curve

def plot_learning_curves(estimator, X, y, title="Learning Curves"):
    """
    Plot learning curves to detect overfitting/underfitting
    """
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
    plt.plot(train_sizes, val_mean, 'o-', color='g', label='Cross-validation score')
    
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='g')
    
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy Score')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Diagnose the issue
    final_train_score = train_mean[-1]
    final_val_score = val_mean[-1]
    gap = final_train_score - final_val_score
    
    print(f"\nDiagnostics:")
    print(f"Final Training Score: {final_train_score:.4f}")
    print(f"Final Validation Score: {final_val_score:.4f}")
    print(f"Train-Val Gap: {gap:.4f}")
    
    if gap > 0.1:
        print("⚠️ High variance (Overfitting) - Consider:")
        print("   - More training data")
        print("   - Regularization")
        print("   - Reduce model complexity")
    elif final_val_score < 0.7:
        print("⚠️ High bias (Underfitting) - Consider:")
        print("   - More complex model")
        print("   - More features")
        print("   - Less regularization")
    else:
        print("✅ Model is well-balanced")

# Use it
plot_learning_curves(ensemble_model, X_train, y_train, "Ensemble Learning Curves")
```

#### 4. Validation Curves for Hyperparameter Tuning

```python
from sklearn.model_selection import validation_curve

def plot_validation_curve(estimator, X, y, param_name, param_range):
    """
    Plot validation curve for a specific hyperparameter
    """
    train_scores, val_scores = validation_curve(
        estimator, X, y,
        param_name=param_name,
        param_range=param_range,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(param_range, train_mean, 'o-', color='r', label='Training score')
    plt.plot(param_range, val_mean, 'o-', color='g', label='Cross-validation score')
    
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
    plt.fill_between(param_range, val_mean - val_std, val_mean + val_std, alpha=0.1, color='g')
    
    plt.xlabel(param_name)
    plt.ylabel('Accuracy Score')
    plt.title(f'Validation Curve - {param_name}')
    plt.legend(loc='best')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# Example: Tune n_estimators for Random Forest
param_range = [10, 50, 100, 200, 300, 500]
plot_validation_curve(
    RandomForestClassifier(random_state=42),
    X_train, y_train,
    'n_estimators',
    param_range
)
```

#### 5. Out-of-Bag (OOB) Score for Random Forest

```python
# Enable OOB score in Random Forest
rf_with_oob = RandomForestClassifier(
    n_estimators=100,
    oob_score=True,
    random_state=42,
    n_jobs=-1
)

rf_with_oob.fit(X_train, y_train)

print(f"OOB Score: {rf_with_oob.oob_score_:.4f}")
print(f"Test Score: {rf_with_oob.score(X_test, y_test):.4f}")

# OOB score is an estimate of generalization performance
# No need for separate validation set
# Useful for quick model evaluation

# Plot OOB error evolution
oob_errors = []
test_errors = []

for n_trees in range(1, 101):
    rf_temp = RandomForestClassifier(
        n_estimators=n_trees,
        oob_score=True,
        random_state=42,
        warm_start=False
    )
    rf_temp.fit(X_train, y_train)
    oob_errors.append(1 - rf_temp.oob_score_)
    test_errors.append(1 - rf_temp.score(X_test, y_test))

plt.figure(figsize=(10, 6))
plt.plot(range(1, 101), oob_errors, label='OOB Error', alpha=0.7)
plt.plot(range(1, 101), test_errors, label='Test Error', alpha=0.7)
plt.xlabel('Number of Trees')
plt.ylabel('Error Rate')
plt.title('OOB Error vs Test Error')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

#### 6. Cross-Validation with Multiple Metrics

```python
from sklearn.model_selection import cross_validate

# Evaluate multiple metrics simultaneously
scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc'
}

cv_results = cross_validate(
    ensemble_model,
    X_train, y_train,
    cv=5,
    scoring=scoring,
    return_train_score=True,
    n_jobs=-1
)

print("\nCross-Validation Results (5-fold):")
print("="*60)
for metric in scoring.keys():
    train_scores = cv_results[f'train_{metric}']
    test_scores = cv_results[f'test_{metric}']
    print(f"{metric.upper()}:")
    print(f"  Train: {train_scores.mean():.4f} (+/- {train_scores.std():.4f})")
    print(f"  Test:  {test_scores.mean():.4f} (+/- {test_scores.std():.4f})")
```

#### 7. Calibration Curve (Probability Calibration)

```python
from sklearn.calibration import calibration_curve, CalibrationDisplay

# Check if predicted probabilities are well-calibrated
prob_true, prob_pred = calibration_curve(
    y_test, y_pred_proba, n_bins=10, strategy='uniform'
)

fig, ax = plt.subplots(figsize=(10, 6))
CalibrationDisplay.from_predictions(
    y_test, y_pred_proba, n_bins=10, ax=ax, name='Ensemble Model'
)
plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
plt.title('Probability Calibration Curve')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# If probabilities are not well-calibrated, use CalibratedClassifierCV
from sklearn.calibration import CalibratedClassifierCV

calibrated_model = CalibratedClassifierCV(ensemble_model, method='isotonic', cv=5)
calibrated_model.fit(X_train, y_train)
```

## Future Directions

### Emerging Trends

1. **Deep Ensembles**: Combining deep neural networks
2. **AutoML with Ensembles**: Automated ensemble construction
3. **Ensemble Pruning**: Selecting optimal subset of models
4. **Online Ensembles**: Adapting to streaming data
5. **Interpretable Ensembles**: Maintaining transparency

### Research Areas

- **Diversity Measures**: Better quantifying model diversity
- **Dynamic Weighting**: Adaptive combination of predictions
- **Transfer Learning**: Using pre-trained ensemble components
- **Federated Ensembles**: Privacy-preserving distributed learning

## Conclusion

Ensemble learning represents one of the most powerful and practical approaches in machine learning. By combining multiple models, we can achieve:

- **Superior Performance**: Consistently better than individual models
- **Robustness**: More reliable predictions across different scenarios
- **Flexibility**: Applicable to various problem types and domains
- **State-of-the-Art Results**: Foundation of winning solutions

### Key Takeaways

1. Start with Random Forest as a strong baseline
2. Use XGBoost/LightGBM/CatBoost for structured data
3. Apply stacking when maximum performance is needed
4. Ensure diversity among base models
5. Use proper cross-validation to prevent overfitting
6. Balance accuracy with computational constraints
7. Monitor and interpret model behavior

Ensemble learning is not just about combining models—it's about strategically leveraging their complementary strengths to build robust, accurate, and reliable machine learning systems.


