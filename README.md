
# Loan Approval Prediction using Machine Learning

This project aims to predict whether a loan application should be **approved or rejected** using supervised machine learning techniques. It uses a structured dataset of 4269 loan applications and applies advanced data preprocessing, feature engineering, model evaluation, and hyperparameter tuning to build a robust prediction pipeline.

## Key Highlights

* **Precision-Focused Approach**: In the context of financial risk, we prioritized **precision** to reduce false positives — ensuring most approved loans are high quality.
* **Extra Validation Layer**: In addition to the traditional train-test split, we added a validation set to tune hyperparameters and a test set for final unbiased evaluation.
* **Model Performance Transparency**: Every model's performance was benchmarked using multiple metrics including:

  * Accuracy
  * F1 Score
  * Precision
  * Recall
  * ROC AUC Score
* **5-Fold Cross Validation**: Added for all optimized models to ensure reliability and consistency across folds.

---

## Dataset Overview

* **Size**: 4269 records, 13 features
* **Feature Types**:

  * Numerical: income, loan amount, asset values, etc.
  * Categorical: education, self-employment status, number of dependents, etc.
* **Target**: `loan_status` (Approved / Rejected)

---

## Data Preprocessing

* Removed redundant ID columns (`loan_id`)
* Handled categorical variables using One-Hot Encoding
* Normalized numerical features with StandardScaler
* Verified absence of missing data via heatmap
* Split into training (64%), validation (16%), and test (20%) sets

---

## Models Evaluated

| Model                  | Validation Accuracy | Validation Precision | Validation F1 Score | ROC AUC |
| ---------------------- | ------------------- | -------------------- | ------------------- | ------- |
| Logistic Regression    | 92.5%               | 88.8%                | 90.3%               | 92.4%   |
| K-Nearest Neighbors    | 89.8%               | 85.6%                | 86.6%               | 89.3%   |
| Decision Tree          | 95.8%               | 93.2%                | 94.5%               | 95.8%   |
| Random Forest          | 95.0%               | 90.6%                | 93.6%               | 95.4%   |
| Support Vector Machine | 93.9%               | 90.9%                | 91.9%               | 93.7%   |
| XGBoost (Best)         | 97.1%               | 95.8%                | 96.1%               | 97.0%   |
| Naive Bayes            | 93.3%               | 91.1%                | 91.1%               | 92.8%   |

---

## Final Model: XGBoost

The XGBoost model was selected for its highest precision and ROC AUC scores.

**Hyperparameters**:

* `learning_rate`: 0.1
* `max_depth`: 3
* `n_estimators`: 400

**Final Test Set Evaluation**:

* Accuracy: 97.8%
* Precision: 97.2%
* F1 Score: 97.1%
* Recall: 96.9%
* ROC AUC: 97.6%

---

## Visualizations

* Feature distribution histograms and boxplots
* Correlation heatmaps
* ROC curves
* Confusion matrices
* Model performance bar charts

---

## Tech Stack

* Python (Pandas, NumPy, Matplotlib, Seaborn)
* Scikit-Learn
* XGBoost
* Joblib (for model persistence)

---

## Output

* Trained Model: `loan_approval_model.pkl`
* Evaluation Summary: Stored in a Pandas DataFrame
* Visualization Plots: Inline via Matplotlib/Seaborn

---

## Takeaways

This project showcases how rigorous model validation, combined with business-aligned evaluation metrics like precision, can produce meaningful results in high-stakes domains such as financial lending.

---

## Future Work

* Use SHAP values for explainability
* Real-time deployment with Flask or FastAPI
* Continuous model monitoring and retraining
* Advanced feature importance analysis

---

## Author

**Ishaan Gupta**


---

Let me know if you'd like to add badges, a `requirements.txt`, or deployment instructions next.

