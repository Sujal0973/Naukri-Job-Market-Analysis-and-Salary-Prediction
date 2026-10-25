# 🤖 Machine Learning Modeling & Benchmark Report

This document details the architecture, feature processing pipelines, experimental benchmarks, evaluation metrics, and error analyses for both the **Job Role Classification Model** and the **Salary Estimation Regressor**.

---

## 1. Problem Formulations

### Primary Task: Multi-Class Job Role Classification
Given candidate requirements, experience, location, company review scores, and skill tags, predict which of the 6 data disciplines the job belongs to:
- `AI Engineer`
- `Business Analyst`
- `Data Analyst`
- `Data Engineer`
- `Data Scientist`
- `Machine Learning Engineer`

### Secondary Task: Salary Estimation Regression
Predict the estimated annual compensation (in Lacs Per Annum - LPA) for job profiles where compensation was disclosed by employers.

---

## 2. Feature Engineering & Preprocessing Pipeline

```text
Input Features
├── Numerical Features: ['Average Experience', 'Rating', 'Reviews', 'Skill Count', 'Posted Days']
│   └── SimpleImputer(strategy='median')
├── Categorical Features: ['Primary Location']
│   ├── SimpleImputer(strategy='constant', fill_value='Unknown')
│   └── OneHotEncoder(handle_unknown='ignore')
└── Text Feature: ['Skills']
    └── TfidfVectorizer(ngram_range=(1, 2), max_features=500, min_df=2, stop_words='english')
```

The feature sets are concatenated using `scipy.sparse.hstack` into a unified feature matrix of **677 dimensions** for 3,308 training samples and 828 stratified test samples (80/20 train/test split).

---

## 3. Multi-Class Classification Benchmark

All models were evaluated on the held-out 20% test dataset ($N = 828$) with stratified sampling:

| Model Architecture | Test Accuracy | Weighted Precision | Weighted Recall | Weighted F1 Score | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **XGBoost Classifier (Champion)** | **65.22%** | **66.68%** | **65.22%** | **65.66%** | Best overall balance across all 6 classes |
| **Random Forest Classifier** | 65.10% | 66.60% | 65.10% | 65.26% | Robust ensemble baseline; slightly slower inference |
| **Gradient Boosting Classifier** | 61.47% | 63.24% | 61.47% | 62.03% | Solid performance on structured tabular features |
| **Decision Tree Classifier** | 59.30% | 60.15% | 59.30% | 59.64% | Prone to overfitting on high-dimensional TF-IDF |
| **Logistic Regression (SAGA/LBFGS)** | 33.82% | 25.80% | 33.82% | 27.73% | Struggled with non-linear feature interactions |

---

## 4. Champion Model Deep Dive (XGBoost)

### Hyperparameters
- `n_estimators`: 400
- `max_depth`: 6
- `learning_rate`: 0.05
- `subsample`: 0.8
- `colsample_bytree`: 0.8
- `objective`: `"multi:softprob"`
- `eval_metric`: `"mlogloss"`

### Classification Report
```text
                           precision    recall  f1-score   support

              AI Engineer       0.72      0.48      0.58        64
         Business Analyst       0.94      0.82      0.87        88
             Data Analyst       0.83      0.82      0.82       168
            Data Engineer       0.90      0.89      0.90       132
           Data Scientist       0.48      0.55      0.51       200
Machine Learning Engineer       0.40      0.41      0.40       176

                 accuracy                           0.65       828
                macro avg       0.71      0.66      0.68       828
             weighted avg       0.67      0.65      0.66       828
```

### Stratified 5-Fold Cross Validation
- **Fold Accuracies**: `[64.95%, 68.13%, 68.28%, 67.17%, 67.02%]`
- **Mean Accuracy**: **67.11%** ($\pm 1.19\%$)
- Demonstrates exceptional stability across data partitions without data leakage.

### Error Analysis & Confusion Matrix Insights
1. **High Distinctiveness**: `Business Analyst` (0.94 Precision) and `Data Engineer` (0.90 Precision) exhibit sharp boundaries due to distinctive vocabularies (`Power BI`, `Excel`, `Tableau` vs. `SQL`, `Spark`, `ETL`, `Hadoop`, `Airflow`).
2. **Domain Overlap**: The primary source of misclassification occurred between `Data Scientist` and `Machine Learning Engineer`. Both job categories frequently request identical stacks (`Python`, `Scikit-learn`, `PyTorch`, `TensorFlow`, `Deep Learning`), leading to semantic ambiguity in raw employer postings.

---

## 5. Salary Prediction Regressor Benchmark

Trained strictly on verified disclosed salary listings ($N = 462$) using 80/20 train/test evaluation:

| Regressor Architecture | $R^2$ Score | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) |
| :--- | :---: | :---: | :---: |
| **Ridge Regression (Champion)** | **0.4072** | **6.02 LPA** | **7.57 LPA** |
| **Gradient Boosting Regressor** | 0.4010 | 5.83 LPA | 7.61 LPA |
| **Random Forest Regressor** | 0.3935 | 6.03 LPA | 7.66 LPA |
| **Decision Tree Regressor** | 0.1730 | 6.54 LPA | 8.94 LPA |

### Key Drivers of Salary
1. **Average Experience**: Strongest linear predictor ($r \approx 0.58$).
2. **Job Role Type**: Data Engineer and Machine Learning Engineer command an estimated 20-35% premium over entry-level Data Analyst positions.
3. **Primary Location**: Bengaluru and Hyderabad exhibit the highest base medians.

---

## 6. Serialized Artifacts

All models, encoders, and transformers are pre-fitted and stored in `models/` for immediate production inference:
- `models/role_classifier_bundle.joblib`: Contains the fitted XGBoost classifier, column transformer, TF-IDF vectorizer, and label encoder.
- `models/salary_predictor_bundle.joblib`: Contains the fitted Ridge/RandomForest regressor and feature transformers.
