Loan Approval Prediction using Machine Learning

This project predicts whether a loan application should be approved or rejected using supervised machine learning. It is based on a structured dataset of 4269 loan records and applies advanced preprocessing, feature engineering, model evaluation, and hyperparameter tuning to build a robust prediction pipeline.

Key Highlights

Precision-Focused Approach: The primary focus is on improving precision to minimize false positives, ensuring that approved loans are of high quality.
Extra Validation Layer: In addition to a traditional train-test split, a separate validation set was used for hyperparameter tuning, and the test set was reserved for final evaluation.
Comprehensive Model Evaluation: Each model was evaluated using multiple metrics including Accuracy, F1 Score, Precision, Recall, and ROC AUC.
Cross-Validation: All models underwent 5-fold cross-validation to ensure robustness and consistency of results.
Dataset Overview

Total Records: 4269
Features: 13
Target Variable: loan_status (Approved / Rejected)
Feature Types:
Numerical: income, loan amount, asset values, etc.
Categorical: education, self-employment status, number of dependents, loan term
Data Preprocessing

Removed ID column (loan_id)
Handled categorical variables using One-Hot Encoding
Standardized numerical features using StandardScaler
Verified absence of missing values using a heatmap
Split data into:
Training set (64%)
Validation set (16%)
Test set (20%)
Models Evaluated

Model	Validation Accuracy	Validation Precision	Validation F1 Score	ROC AUC
Logistic Regression	92.5%	88.8%	90.3%	92.4%
K-Nearest Neighbors	89.8%	85.6%	86.6%	89.3%
Decision Tree	95.8%	93.2%	94.5%	95.8%
Random Forest	95.0%	90.6%	93.6%	95.4%
Support Vector Machine	93.9%	90.9%	91.9%	93.7%
XGBoost (Best Model)	97.1%	95.8%	96.1%	97.0%
Naive Bayes	93.3%	91.1%	91.1%	92.8%

Final Model: XGBoost

The XGBoost model was selected based on its superior precision and ROC AUC score.

Best Hyperparameters:

learning_rate: 0.1
max_depth: 3
n_estimators: 400
Test Set Performance:

Accuracy: 97.8%
Precision: 97.2%
F1 Score: 97.1%
Recall: 96.9%
ROC AUC: 97.6%
Visualizations

Feature distribution histograms and boxplots
Correlation heatmap
ROC curve
Confusion matrix
Bar plots of model performance metrics
Tech Stack

Python
Pandas, NumPy
Matplotlib, Seaborn
Scikit-Learn
XGBoost
Joblib (for model serialization)
Output

Trained model: loan_approval_model.pkl
Evaluation metrics saved in a DataFrame
Visual plots generated using Matplotlib and Seaborn
Takeaways

This project demonstrates the importance of using multiple validation layers and precision-focused evaluation in high-risk domains like financial services. With careful model selection and robust metrics tracking, it's possible to deliver highly accurate and reliable predictive models.

Future Work

Integrating SHAP for feature importance and explainability
Building a real-time API with Flask or FastAPI
Deploying the model to cloud infrastructure
Implementing continuous monitoring and retraining

Author

Ishaan Gupta
Data Scientist & Machine Learning Enthusiast
Email: gupta03ishaan@gmail.com
