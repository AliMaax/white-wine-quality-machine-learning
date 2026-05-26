import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score

 
# SECTION 3.1.1 BASELINE MODEL
# Ali Maaz (Registration No: 2023-EE-21)


# Reference: Linear Regression (Normal Equation / closed-form solution)
# https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/supervised_learning/regression.py
# https://github.com/rushter/MLAlgorithms/blob/master/mla/linear_models.py
class LinearRegression:
    def __init__(self):
        self.weights = None

    # Reference: For solving the normal equation (X^T X)^-1 X^T y 
    # https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/supervised_learning/regression.py#L48
    def fit(self, X, y):
        X_vals = X.values if isinstance(X, pd.DataFrame) else X
        y_vals = y.values if isinstance(y, pd.Series) else y
        
        # Add bias term (column of 1s)
        X_b = np.c_[np.ones((X_vals.shape[0], 1)), X_vals]
        self.weights = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y_vals

    def predict(self, X):
        X_vals = X.values if isinstance(X, pd.DataFrame) else X
        X_b = np.c_[np.ones((X_vals.shape[0], 1)), X_vals]
        return X_b @ self.weights



# SECTION 3.1.2 MAIN MODEL
# Muhammad Hashim Butt (Registration No: 2023-EE-27)


# Reference: Decision Tree node structure 
# https://github.com/python-engineer/MLfromscratch/blob/master/mlfromscratch/decision_tree.py#L1
# https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/supervised_learning/decision_tree.py
class TreeNode:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, *, value=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        
    def is_leaf(self):
        return self.value is not None

# Reference: Decision Tree classifier built from scratch using Gini impurity 
# https://github.com/python-engineer/MLfromscratch/blob/master/mlfromscratch/decision_tree.py
# https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/supervised_learning/decision_tree.py
class DecisionTree:
    def __init__(self, max_depth=12, min_samples_split=4):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        self.root = self._grow_tree(X, y)

    # Reference: Recursive tree growing with random feature subsampling (as used in Random Forest)
    # https://github.com/python-engineer/MLfromscratch/blob/master/mlfromscratch/decision_tree.py#L50
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        if depth >= self.max_depth or len(np.unique(y)) == 1 or n_samples < self.min_samples_split:
            return TreeNode(value=self._most_common_class(y))
            
        feat_idxs = np.random.choice(n_features, int(np.sqrt(n_features)), replace=False)
        best_feat, best_thresh = self._best_split(X, y, feat_idxs)
        
        if best_feat is None:
            return TreeNode(value=self._most_common_class(y))
            
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)
        return TreeNode(best_feat, best_thresh, left, right)

    # Reference: Best split search by iterating unique thresholds per feature 
    # https://github.com/python-engineer/MLfromscratch/blob/master/mlfromscratch/decision_tree.py#L30
    def _best_split(self, X, y, feat_idxs):
        best_gain, split_idx, split_thresh = -1, None, None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            for threshold in np.unique(X_column):
                gain = self._information_gain(y, X_column, threshold)
                if gain > best_gain:
                    best_gain, split_idx, split_thresh = gain, feat_idx, threshold
        return split_idx, split_thresh

    def _information_gain(self, y, X_column, threshold):
        # Info gain = Gini(parent) - weighted average Gini(left child, right child)
        left_idxs, right_idxs = self._split(X_column, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        n = len(y)
        child_gini = (len(left_idxs)/n) * self._gini(y[left_idxs]) + (len(right_idxs)/n) * self._gini(y[right_idxs])
        return self._gini(y) - child_gini

    def _split(self, X_column, split_thresh):
        return np.argwhere(X_column <= split_thresh).flatten(), np.argwhere(X_column > split_thresh).flatten()

    def _gini(self, y):
        # Reference: Gini impurity formula 
        # https://github.com/python-engineer/MLfromscratch/blob/master/mlfromscratch/decision_tree.py#L12
        proportions = np.bincount(y) / len(y)
        return 1 - np.sum([p**2 for p in proportions if p > 0])

    def _most_common_class(self, y):
        counts = Counter(y)
        return counts.most_common(1)[0][0] if counts else 0

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    # Reference: Tree traversal for inference 
    # https://github.com/python-engineer/MLfromscratch/blob/master/mlfromscratch/decision_tree.py#L75
    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature_idx] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

# Reference: Random Forest using bootstrap aggregation (bagging) over Decision Trees 
# https://github.com/python-engineer/MLfromscratch/blob/master/mlfromscratch/random_forest.py
# https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/supervised_learning/random_forest.py
class RandomForest:
    def __init__(self, n_estimators=15, max_depth=12, min_samples_split=4):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    def fit(self, X, y):
        X_vals = X.values if isinstance(X, pd.DataFrame) else X
        y_vals = y.values if isinstance(y, pd.Series) else y
        self.trees = []
        for _ in range(self.n_estimators):
            tree = DecisionTree(self.max_depth, self.min_samples_split)
            X_samp, y_samp = self._bootstrap_sample(X_vals, y_vals)
            tree.fit(X_samp, y_samp)
            self.trees.append(tree)

    # Reference: Bootstrap sampling with replacement 
    # https://github.com/python-engineer/MLfromscratch/blob/master/mlfromscratch/random_forest.py#L28
    def _bootstrap_sample(self, X, y):
        idxs = np.random.choice(X.shape[0], X.shape[0], replace=True)
        return X[idxs], y[idxs]

    # Reference: Majority vote aggregation across all trees 
    # https://github.com/python-engineer/MLfromscratch/blob/master/mlfromscratch/random_forest.py#L40
    def predict(self, X):
        X_vals = X.values if isinstance(X, pd.DataFrame) else X
        tree_preds = np.array([tree.predict(X_vals) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        return np.array([Counter(pred).most_common(1)[0][0] for pred in tree_preds])
try:
    df = pd.read_csv("winequality-white.csv", sep=';')
    print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} columns\n")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

X = df.drop(columns=['quality'])
y = df['quality']

# Reference: Stratified train/test split to preserve class distribution 
# https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/model_selection/_split.py
X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# Reference: StratifiedKFold cross-validation 
# https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/model_selection/_split.py#L1353
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = {
    'Baseline (LR)':   {'rmse': [], 'acc': [], 'f1': []},
    'Main Model (RF)': {'rmse': [], 'acc': [], 'f1': []},
    'Flexible (SVM)':  {'rmse': [], 'acc': [], 'f1': []}
}

print("Running 5-Fold Stratified Cross-Validation...\n")

for fold, (train_idx, val_idx) in enumerate(skf.split(X_dev, y_dev), start=1):
    X_train, X_val = X_dev.iloc[train_idx].copy(), X_dev.iloc[val_idx].copy()
    y_train, y_val = y_dev.iloc[train_idx].copy(), y_dev.iloc[val_idx].copy()
      
    # 3.1.1 BASELINE PIPELINE PROCESSING [Ali Maaz, 2023-EE-21]
    drop_cols = ['density', 'pH', 'citric acid', 'sulphates']
    X_train_lr = X_train.drop(columns=drop_cols)
    X_val_lr = X_val.drop(columns=drop_cols)

    for col in ['chlorides', 'residual sugar']:
        X_train_lr[col] = np.log1p(X_train_lr[col])
        X_val_lr[col] = np.log1p(X_val_lr[col])

    # Reference: StandardScaler for zero-mean unit-variance normalization 
    # https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/preprocessing/_data.py#L517
    scaler_lr = StandardScaler()
    X_train_lr_scaled = scaler_lr.fit_transform(X_train_lr)
    X_val_lr_scaled = scaler_lr.transform(X_val_lr)

    lr = LinearRegression()
    lr.fit(X_train_lr_scaled, y_train)
    y_pred_cont = lr.predict(X_val_lr_scaled)
    y_pred_class = np.clip(np.round(y_pred_cont), 3, 9).astype(int)

    results['Baseline (LR)']['rmse'].append(np.sqrt(mean_squared_error(y_val, y_pred_cont)))
    results['Baseline (LR)']['acc'].append(accuracy_score(y_val, y_pred_class))
    results['Baseline (LR)']['f1'].append(f1_score(y_val, y_pred_class, average='macro'))

    # 3.1.2 MAIN MODEL ENSEMBLE PROCESSING [Muhammad Hashim Butt, 2023-EE-27]

    rf = RandomForest(n_estimators=15, max_depth=12, min_samples_split=4)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_val)

    results['Main Model (RF)']['rmse'].append(np.sqrt(mean_squared_error(y_val, y_pred_rf.astype(float))))
    results['Main Model (RF)']['acc'].append(accuracy_score(y_val, y_pred_rf))
    results['Main Model (RF)']['f1'].append(f1_score(y_val, y_pred_rf, average='macro'))

    # 3.1.3 FLEXIBLE COMPARISON PIPELINE PROCESSING [Ali Maaz, 2023-EE-21]

    # Reference: RobustScaler (median/IQR scaling, less sensitive to outliers) 
    # https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/preprocessing/_data.py#L1007
    scaler_svm = RobustScaler()
    X_train_svm = scaler_svm.fit_transform(X_train)
    X_val_svm = scaler_svm.transform(X_val)

    # Reference: SVC with RBF kernel and balanced class weights 
    # https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/svm/_classes.py#L475
    svm = SVC(kernel='rbf', C=5.0, class_weight='balanced', gamma='scale', random_state=42)
    svm.fit(X_train_svm, y_train)
    y_pred_svm = svm.predict(X_val_svm)

    results['Flexible (SVM)']['rmse'].append(np.sqrt(mean_squared_error(y_val, y_pred_svm.astype(float))))
    results['Flexible (SVM)']['acc'].append(accuracy_score(y_val, y_pred_svm))
    results['Flexible (SVM)']['f1'].append(f1_score(y_val, y_pred_svm, average='macro'))

print(f"{'CROSS-VALIDATION RESULTS (mean ± std)':^60}")
for model_name, scores in results.items():
    print(f"\n  {model_name}")
    print(f"    RMSE:      {np.mean(scores['rmse']):.3f}  ±  {np.std(scores['rmse']):.3f}")
    print(f"    Accuracy:  {np.mean(scores['acc'])*100:.1f}%  ±  {np.std(scores['acc'])*100:.1f}%")
    print(f"    Macro F1:  {np.mean(scores['f1']):.3f}  ±  {np.std(scores['f1']):.3f}")