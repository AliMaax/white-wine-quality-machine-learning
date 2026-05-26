# EE 439: White Wine Quality Prediction — Data First, Models Second

An end-to-end machine learning project analyzing the structural characteristics and quality classification of the **UCI White Wine Quality dataset**. This repository evaluates how engineering a highly overlapping feature space impacts global boundary classifiers versus local tree-based partitioning models.

In compliance with the course **DIY policy**, the Baseline (Linear Regression) and Main (Random Forest) models were built entirely from scratch using pure Python and NumPy.

---

## 📊 1. Exploratory Data Analysis (EDA) Insights

The dataset was split into an **80/20 Development/Test** configuration ($3,918$ development samples, $980$ isolated test samples) with a fixed seed of `random_state=42`. 

### Key Structural Findings:
* **The Alcohol-Quality Axis:** Correlation exploration confirms that alcohol content is the strongest positive linear driver of wine quality ($\rho = 0.43$).
* **Physical Multicollinearity:** A severe physical dependency exists between alcohol content and fluid density ($\rho = -0.77$). Because ethanol is less dense than water, higher alcohol naturally depresses density. 
* **Statistical Malformations:** Features like `chlorides`, `total sulfur dioxide`, and `residual sugar` contain extreme, skewed statistical outliers concentrated heavily within the mid-tier ($5$ and $6$) quality ratings.
* **Low-Signal Noise:** Scatter plot analyses of `pH`, `citric acid`, and `sulphates` form uniform vertical clouds with near-zero correlation to the target variable, acting as background noise in their raw forms.
* **High-Dimensional Inseparability:** 2D Principal Component Analysis (PCA) projections and Parallel Coordinates plots reveal a single, dense, heavily superimposed cluster showing zero linear separability across quality tiers.

---

## 🛠️ 2. Tailored Preprocessing Pipelines

To address these unique data characteristics, distinct preprocessing pipelines were designed for each model layout:

| Model | Preprocessing Architecture | Justification |
| :--- | :--- | :--- |
| **Linear Regression** | • Drop `pH`, `citric acid`, `sulphates`<br>• Drop `density` to remove multicollinearity<br>• Standard Scaling | Prevents redundant feature overlap from destabilizing closed-form matrix coefficients. |
| **Random Forest** | • No feature scaling or scaling modifications applied (Original 11 features kept) | Tree-based models are naturally immune to monotonic scale differences and extreme distribution outliers. |
| **Support Vector Machine** | • Robust Scaling (`RobustScaler`) | Protects distance-based margin operations from being warped by extreme mid-tier outliers. |

---

## 💻 3. Model Implementation Details

### 📐 Baseline: Ordinary Least Squares (OLS) Linear Regression *(From Scratch)*
* **Library Constraints:** Pure `numpy` matrix math.
* **Optimization:** Closed-form global solution computed directly via the Moore-Penrose pseudo-inverse matrix equation:
$$\beta = (X^T X)^{-1} X^T y$$
* **Hyperparameters:** None.

### 🌲 Main Model: Random Forest Ensemble *(From Scratch)*
* **Library Constraints:** Pure Python and `numpy`.
* **Core Logic:** Custom implementation of recursive Gini impurity splitting bounds for standalone Decision Trees combined via bootstrap aggregation (bagging).
* **Hyperparameters:** `n_estimators=15`, `max_depth=12`, `min_samples_split=4`. These structural limits maintain computational viability while keeping individual trees from over-memorizing noisy mid-tier fluctuations.

### 🔮 Flexible Comparison: Radial Basis Function (RBF) SVM *(Scikit-Learn)*
* **Library Constraints:** `sklearn.svm.SVC`.
* **Core Logic:** Maps overlapping low-dimensional samples into higher dimensions to discover non-linear margin separations.
* **Hyperparameters:** `kernel='rbf'`, Soft-margin penalty `C=5.0`, `gamma='scale'`, and `class_weight='balanced'` to actively counteract severe class sparsity at the tail ends (ratings $3$ and $9$).

---

## 📈 4. Cross-Validation Performance Metrics

Evaluations were gathered using a **5-fold Stratified Cross-Validation** scheme over the $3,918$ development samples. Due to target integer distributions, performance is tracked across both geometric regression and discrete classification scales.

### Performance Matrix Summary

| Model Framework | RMSE (⬇️) | Classification Accuracy (⬆️) | Macro F1-Score (⬆️) |
| :--- | :---: | :---: | :---: |
| **Linear Regression (Baseline)** | $0.755 \pm 0.006$ | $51.8\% \pm 1.3\%$ | $0.219 \pm 0.016$ |
| **Random Forest (Main Model)** | **$0.732 \pm 0.021$** | **$62.1\% \pm 1.6\%$** | **$0.319 \pm 0.024$** |
| **SVM RBF Kernel (Flexible)** | $0.980 \pm 0.022$ | $48.1\% \pm 2.0\%$ | $0.313 \pm 0.039$ |

### Core Evaluation Takeaways:
1. **The Random Forest Won Globally:** Locally scoped tree partitioning handled the dense class overlaps far better than global boundary approaches. Bootstrap bagging minimized the impact of noisy mid-tier data points.
2. **The OLS Paradox:** Linear Regression preserved a reasonably low RMSE but suffered a poor F1-score ($0.219$). Because OLS minimizes squared error globally, it pulls predictions safely toward the dataset mean ($\approx 5.8$)—making it completely blind to minority wine profiles at the extreme tails.
3. **The SVM Failure:** The RBF SVM yielded the weakest accuracy ($48.1\%$). Forcing a high margin penalty ($C=5.0$) alongside `balanced` class weights forced the classifier to prioritize highly sparse minority boundaries, completely degrading its accuracy over the dominant mid-tier samples.

---

## 👥 5. Statement of Contributions

* **Muhammad Hashim Butt (2023-EE-27):**
  * Designed data partitioning and the 5-fold cross-validation architecture.
  * Conducted non-linear trend evaluations, PCA projections, and parallel coordinate plots.
  * Developed and coded the recursive Gini impurity splitting parameters and bootstrap ensemble logic for the custom `RandomForest` class from scratch.
* **Ali Maaz (2023-EE-21):**
  * Conducted feature distribution analysis and outlier exploration during EDA.
  * Built the feature-pruning cleaning pipeline and multicollinearity removal mechanisms.
  * Implemented the baseline OLS Linear Regression model from scratch via matrix math.
  * Built the comparison scikit-learn RBF SVM pipeline and generated the performance visualization scripts.

---

