---
title: "Scikit Learn Overview"
date:   2024-02-25 22:00:00
categories: [Machine Learning]
tags: [Machine learning,Python]    
image:
  path: /assets/imgs/headers/scikitLearn.png
---

## Introduction:
scikit-learn is a comprehensive machine learning library in Python that offers a wide range of algorithms for various tasks such as classification, regression, clustering, dimensionality reduction, and more. This cheat sheet provides an overview of some commonly used models and techniques in scikit-learn.

## 1. Loading Datasets:

```python
from sklearn import datasets

# Load a dataset
dataset = datasets.load_dataset_name()
X, y = dataset.data, dataset.target
```
**datasets:** scikit-learn provides various built-in datasets for experimentation and practice.

## 2. Preprocessing Data:
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Normalize features
minmax_scaler = MinMaxScaler()
X_train_normalized = minmax_scaler.fit_transform(X_train)
X_test_normalized = minmax_scaler.transform(X_test)
```
**train_test_split:** Splitting data into training and testing sets for model evaluation.

**StandardScaler, MinMaxScaler:** Standardizing and normalizing features to ensure consistency and improve model performance.

## 3. Model Building:
### 3.1 Linear Models:

```python
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Ridge Regression
ridge_reg = Ridge(alpha=1.0)
ridge_reg.fit(X_train, y_train)

# Lasso Regression
lasso_reg = Lasso(alpha=1.0)
lasso_reg.fit(X_train, y_train)
```
**Linear Regression:** For predicting continuous values.

**Logistic Regression:** For binary classification tasks.

**Ridge and Lasso Regression:** For regularization to prevent overfitting.

### 3.2 Tree-Based Models:
```python
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Decision Tree Classifier
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train, y_train)

# Decision Tree Regressor
dt_regressor = DecisionTreeRegressor()
dt_regressor.fit(X_train, y_train)

# Random Forest Classifier
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)

# Random Forest Regressor
rf_regressor = RandomForestRegressor()
rf_regressor.fit(X_train, y_train)
```
**Decision Trees:** Versatile models for both classification and regression tasks.

**Random Forest:** Ensemble method based on decision trees for improved performance and robustness.

### 3.3 Support Vector Machines (SVM):

```python
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR

# Support Vector Classifier
svm_classifier = SVC(kernel='linear', C=1.0)
svm_classifier.fit(X_train_scaled, y_train)

# Support Vector Regressor
svm_regressor = SVR(kernel='linear', C=1.0)
svm_regressor.fit(X_train_scaled, y_train)

# Linear Support Vector Classifier
linear_svm_classifier = LinearSVC(C=1.0)
linear_svm_classifier.fit(X_train_scaled, y_train)

# Linear Support Vector Regressor
linear_svm_regressor = LinearSVR(C=1.0)
linear_svm_regressor.fit(X_train_scaled, y_train)
```
**SVM Classifier:** For classification tasks with linear or non-linear decision boundaries.

**SVM Regressor:** For regression tasks to predict continuous values.

**LinearSVC, LinearSVR:** Linear SVM implementations for large-scale datasets.

### 3.4 Nearest Neighbors:
```python
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# K-Nearest Neighbors Classifier
knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train_scaled, y_train)

# K-Nearest Neighbors Regressor
knn_regressor = KNeighborsRegressor()
knn_regressor.fit(X_train_scaled, y_train)
```
**K-Nearest Neighbors:** Non-parametric method for classification and regression based on proximity to neighboring points.

### 3.5 Naive Bayes:
```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

# Gaussian Naive Bayes
gaussian_nb = GaussianNB()
gaussian_nb.fit(X_train, y_train)

# Multinomial Naive Bayes
multinomial_nb = MultinomialNB()
multinomial_nb.fit(X_train, y_train)

# Bernoulli Naive Bayes
bernoulli_nb = BernoulliNB()
bernoulli_nb.fit(X_train, y_train)
```
**Naive Bayes:** Probabilistic classifiers based on Bayes' theorem with strong independence assumptions between features.

### 3.6 Ensemble Methods:

```python
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor, GradientBoostingClassifier, GradientBoostingRegressor

# AdaBoost Classifier
adaboost_classifier = AdaBoostClassifier()
adaboost_classifier.fit(X_train, y_train)

# AdaBoost Regressor
adaboost_regressor = AdaBoostRegressor()
adaboost_regressor.fit(X_train, y_train)

# Gradient Boosting Classifier
gradientboost_classifier = GradientBoostingClassifier()
gradientboost_classifier.fit(X_train, y_train)

# Gradient Boosting Regressor
gradientboost_regressor = GradientBoostingRegressor()
gradientboost_regressor.fit(X_train, y_train)
```
**AdaBoost:** Adaptive boosting technique for classification and regression.

**Gradient Boosting:** Boosting technique that builds models sequentially to correct errors of previous models.

### 4. Model Evaluation:

```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Predictions
y_pred = svm_classifier.predict(X_test_scaled)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

# Classification report
report = classification_report(y_test, y_pred)

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
```
**accuracy_score:** Evaluating classification accuracy.

**classification_report:** Providing precision, recall, F1-score, and support for each class.

**confusion_matrix:** Visualizing model performance in terms of true positives, true negatives, false positives, and false negatives.

### 5. Hyperparameter Tuning with GridSearchCV:
```python
from sklearn.model_selection import GridSearchCV

# Example: Hyperparameter tuning for SVM
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1]}
grid_search = GridSearchCV(svm_classifier, param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)
best_params = grid_search.best_params_
```
**GridSearchCV:** Searching for the best combination of hyperparameters for improved model performance.

### 6. Saving and Loading Models:
```python
import joblib

# Save model
joblib.dump(svm_classifier, 'svm_classifier.pkl')

# Load model
loaded_model = joblib.load('svm_classifier.pkl')
```
**joblib:** Saving and loading trained models for future use.


## 7. Cross-Validation Functions

```python
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold, ShuffleSplit

# Cross-validation
cv_scores = cross_val_score(model, X, y, cv=5)
cross_val_score: Evaluates model performance using cross-validation.
```

**KFold, StratifiedKFold, ShuffleSplit:** Different cross-validation strategies to split data into train/test sets.

#### Additional Evaluation Metrics

```python
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score

# Mean Squared Error (MSE)
mse = mean_squared_error(y_true, y_pred)

# R-squared (R²)
r2 = r2_score(y_true, y_pred)

# Area under ROC curve (AUC-ROC)
auc_roc = roc_auc_score(y_true, y_pred_proba)
```
**mean_squared_error:** Computes the mean squared error for regression tasks.

**r2_score:** Computes the coefficient of determination for regression tasks.

**roc_auc_score:** Computes the area under the ROC curve for classification tasks.

### Data Transformation

```python

from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, chi2

# Polynomial transformation
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Selecting most important features
selector = SelectKBest(score_func=chi2, k=5)
X_selected = selector.fit_transform(X, y)
```
**PolynomialFeatures:** Transforms features into polynomial features to model nonlinear relationshipsBest: Selects the K best features based on statistical tests like chi-square.

### Pipelines

```python

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Creating a pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=2)),
    ('svm', SVC(kernel='linear'))
])

# Training the model with the pipeline
pipeline.fit(X_train, y_train)

# Predictions
y_pred = pipeline.predict(X_test)
```
**Pipeline:** Chains multiple data processing and learning steps into a single object.

### Clustering

```python
from sklearn.cluster import KMeans, AgglomerativeClustering

# K-Means Clustering
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

# Agglomerative Clustering
agg_clustering = AgglomerativeClustering(n_clusters=3)
agg_clustering.fit(X)
```
**KMeans:** Clustering algorithm based on the k-means method.

**AgglomerativeClustering:** Agglomerative hierarchical clustering method.

### 8. Model Evaluation Metrics

```python
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

# Cosine similarity
cosine_sim = cosine_similarity(X1, X2)

# Euclidean distances
euclidean_dist = euclidean_distances(X1, X2)

```
**cosine_similarity:** Computes cosine similarity between two data sets.

**euclidean_distances:** Computes Euclidean distances between two data sets.

## Model Validation

```python
from sklearn.model_selection import validation_curve

# Validation curves
train_scores, valid_scores = validation_curve(estimator, X, y, param_name, param_range)
```
**validation_curve:** Evaluates model performance on a validation set for different hyperparameter values.

## Dimensionality Reduction

```python
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE

# Principal Component Analysis (PCA)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Truncated Singular Value Decomposition (TruncatedSVD)
svd = TruncatedSVD(n_components=2)
X_svd = svd.fit_transform(X)

# t-Distributed Stochastic Neighbor Embedding (t-SNE)
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)
```
**PCA:** Reduces dimensionality while preserving maximum variance.

**TruncatedSVD:** Dimensionality reduction for sparse matrices.

**t-SNE:** Dimensionality reduction technique for high-dimensional data visualization.

## Missing Data Imputation

```python
from sklearn.impute import SimpleImputer

# Imputing missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
```
**SimpleImputer:** Replaces missing values with statistics like mean, median, or most frequent.

## Ensemble Sampling

```python
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# Bagging Classifier
bagging_classifier = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=10)
bagging_classifier.fit(X_train, y_train)

# Bagging Regressor
bagging_regressor = BaggingRegressor(base_estimator=DecisionTreeRegressor(), n_estimators=10)
bagging_regressor.fit(X_train, y_train)
```
**Bagging:** Ensemble method that aggregates predictions from multiple base models.


## Model Selection:

```python
from sklearn.model_selection import RandomizedSearchCV

# Randomized search for hyperparameters
random_search = RandomizedSearchCV(estimator, param_distributions, n_iter=100, cv=5, random_state=42)
random_search.fit(X_train, y_train)
best_params = random_search.best_params_
```
**RandomizedSearchCV:** Randomized search for hyperparameters to optimize model performance.

## Resampling Methods:

```python
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# SMOTE oversampling
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)

# Random undersampling
rus = RandomUnderSampler()
X_resampled, y_resampled = rus.fit_resample(X, y)
```
**SMOTE (Synthetic Minority Over-sampling Technique):** Technique for oversampling to balance classes by creating synthetic examples of the minority class.

**RandomUnderSampler:** Technique for random undersampling to balance classes by reducing the size of the majority class.

## Density-Based Clustering:

```python
from sklearn.cluster import DBSCAN

# DBSCAN Clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.fit(X)
```
**DBSCAN (Density-Based Spatial Clustering of Applications with Noise):** Density-based clustering algorithm that identifies dense regions of points in the data space.

## Linear Models with Elastic Net Regularization:

```python
from sklearn.linear_model import ElasticNet

# Elastic Net Regression
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_net.fit(X_train, y_train)
```
**ElasticNet:** Linear regression model with a combined l1 and l2 penalty for regularization.

## Model Stability Evaluation:

```python
from sklearn.utils import resample
from sklearn.metrics import accuracy_score

# Bootstrap to evaluate model stability
n_iterations = 1000
scores = []
for _ in range(n_iterations):
    X_boot, y_boot = resample(X_train, y_train)
    model.fit(X_boot, y_boot)
    y_pred = model.predict(X_test)
    scores.append(accuracy_score(y_test, y_pred))
```
**Bootstrap:** Resampling method to evaluate model stability by calculating the distribution of performance scores over multiple samples.

## Semi-Supervised Learning:

```python
from sklearn.semi_supervised import LabelPropagation, LabelSpreading

# Label Propagation
label_propagation = LabelPropagation()
label_propagation.fit(X_train, y_train)

# Label Spreading
label_spreading = LabelSpreading()
label_spreading.fit(X_train, y_train)
```
**Label Propagation:** Semi-supervised learning algorithm that propagates labels from a small set of known labels to the entire dataset.

**Label Spreading:** Similar to Label Propagation but with a smoother label propagation process.

## Regression Evaluation:

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Mean Absolute Error (MAE)
mae = mean_absolute_error(y_true, y_pred)

# Mean Squared Error (MSE)
mse = mean_squared_error(y_true, y_pred)
```
**mean_absolute_error:** Computes the mean absolute error for regression tasks.

**mean_squared_error:** Computes the mean squared error for regression tasks.

## Specialized Preprocessing Functions:

```python
from sklearn.preprocessing import RobustScaler, PowerTransformer, QuantileTransformer

# Robust Scaler
robust_scaler = RobustScaler()
X_robust_scaled = robust_scaler.fit_transform(X)

# Power Transformer
power_transformer = PowerTransformer(method='yeo-johnson')
X_power_transformed = power_transformer.fit_transform(X)

# Quantile Transformer
quantile_transformer = QuantileTransformer(output_distribution='normal')
X_quantile_transformed = quantile_transformer.fit_transform(X)
```
**RobustScaler:** Robust scaling of features to reduce the impact of outliers.

**PowerTransformer:** Power transformation to stabilize variance and make data more Gaussian-like.

**QuantileTransformer:** Transformation based on quantiles to uniformize or normalize feature distributions.

## Feature Selection Methods:

```python
from sklearn.feature_selection import RFE, SelectFromModel

# Recursive Feature Elimination (RFE)
rfe = RFE(estimator, n_features_to_select=5)
X_rfe_selected = rfe.fit_transform(X, y)

# Selection from a model
selector = SelectFromModel(estimator)
X_selected = selector.fit_transform(X, y)
```
**RFE (Recursive Feature Elimination):** Iterative method for selecting the most important features by eliminating those with the least impact on the model.

**SelectFromModel:** Selects features based on importance attributed by a given model.

# Conclusion:
scikit-learn provides a vast array of models and tools for machine learning tasks. By leveraging its functionalities, you can build, evaluate, and fine-tune models efficiently. This cheat sheet serves as a guide to help you navigate through the various components of scikit-learn and accelerate your machine learning projects.

