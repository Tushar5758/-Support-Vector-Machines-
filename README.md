# Support Vector Machines (SVM) – Breast Cancer Classification

## Objective  
The goal of this task is to apply **Support Vector Machines (SVMs)** for both **linear** and **non-linear (RBF)** classification on the **Breast Cancer dataset**, visualize decision boundaries, and tune hyperparameters for improved performance.

---

## Tools & Libraries  
- Python 3.x  
- NumPy – numerical operations  
- Pandas – dataset handling  
- Matplotlib – visualization  
- Scikit-learn – SVM, preprocessing, evaluation, cross-validation  

---

## Dataset  
The dataset used is the **Breast Cancer Wisconsin (Diagnostic) dataset**.  
It contains the following key features:  

- **Target**: `diagnosis` (M = Malignant, B = Benign)  
- **Features**: `radius_mean`, `texture_mean`, `perimeter_mean`, `area_mean`, `smoothness_mean`, `compactness_mean`, `concavity_mean`, `concave points_mean`, `symmetry_mean`, `fractal_dimension_mean`, … up to worst-case values.  

Total **30 features** are considered for classification.  

---

## Steps Implemented  

### 1. Data Preprocessing  
- Removed unnecessary columns (`id`).  
- Encoded `diagnosis` column into binary labels (M = 1, B = 0).  
- Standardized features using **StandardScaler**.  
- Split data into training and testing sets (80-20 split).  

### 2. Model Training  
- **Linear SVM** trained with `kernel='linear'`.  
- **RBF SVM** trained with `kernel='rbf'`.  

### 3. Evaluation  
- Used **confusion matrix**, **classification report**, and **accuracy score**.  
- Compared performance of **linear vs RBF kernels**.  

### 4. Visualization  
- Applied **PCA (2D projection)** to reduce features for visualization.  
- Plotted **decision boundaries** for both Linear and RBF kernels.  

### 5. Hyperparameter Tuning  
- Used **GridSearchCV** to optimize hyperparameters:  
  - `C` → Regularization parameter  
  - `gamma` → Kernel coefficient for RBF  
- Selected best parameters using **cross-validation**.  

### 6. Cross-Validation  
- Applied **10-fold cross-validation** for robust evaluation.  
- Reported average accuracy across folds.  

---

## Results  

- **Linear SVM**: Good performance, but limited in handling non-linear separation.  
- **RBF SVM**: Better generalization and higher accuracy after tuning.  
- **GridSearchCV** identified the optimal values of **C** and **gamma**.  
- **Cross-validation** confirmed model stability with high average accuracy.  

---

## Visualization Samples  

- **Decision Boundary (Linear SVM with PCA)**  
- **Decision Boundary (RBF SVM with PCA)**  

These plots help understand how SVM separates benign and malignant cases in 2D space after dimensionality reduction.  

---

