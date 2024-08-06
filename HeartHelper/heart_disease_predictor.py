import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import joblib

# Load the dataset from the URL
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
data = pd.read_csv(url, header=None)

# Assign column names based on the dataset description
data.columns = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]

# Data Preprocessing
# Replace missing values represented by '?' with NaN
data.replace('?', np.nan, inplace=True)
data = data.astype(float)  # Convert all columns to float

# Replace missing values with the median value of the column
data.fillna(data.median(), inplace=True)

# Additional Feature Engineering
# Create polynomial features
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
poly_features = poly.fit_transform(data.drop(columns='target'))
poly_feature_names = poly.get_feature_names(data.columns[:-1])
poly_data = pd.DataFrame(poly_features, columns=poly_feature_names)

# Combine original and polynomial features
combined_data = pd.concat([data, poly_data], axis=1)

# Split the dataset into features and target variable
X = combined_data.drop(columns='target')
y = combined_data['target']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Feature Selection using Recursive Feature Elimination (RFE)
model = LogisticRegression(solver='liblinear', max_iter=1000)
rfe = RFE(model, n_features_to_select=20)  # Select top 20 features
fit = rfe.fit(X_scaled, y)
selected_features = fit.support_
X_selected = X_scaled[:, selected_features]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Hyperparameter Tuning using Grid Search for Logistic Regression
param_grid_lr = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2']
}
grid_lr = GridSearchCV(LogisticRegression(solver='liblinear', max_iter=1000), param_grid_lr, cv=5)
grid_lr.fit(X_train, y_train)

# Best parameters and estimator for Logistic Regression
best_params_lr = grid_lr.best_params_
best_estimator_lr = grid_lr.best_estimator_

# Cross-Validation for Logistic Regression
cv_scores_lr = cross_val_score(best_estimator_lr, X_train, y_train, cv=5)

# Train the Logistic Regression model with best parameters
best_estimator_lr.fit(X_train, y_train)

# Hyperparameter Tuning using Grid Search for Decision Tree
param_grid_dt = {
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}
grid_dt = GridSearchCV(DecisionTreeClassifier(), param_grid_dt, cv=5)
grid_dt.fit(X_train, y_train)

# Best parameters and estimator for Decision Tree
best_params_dt = grid_dt.best_params_
best_estimator_dt = grid_dt.best_estimator_

# Cross-Validation for Decision Tree
cv_scores_dt = cross_val_score(best_estimator_dt, X_train, y_train, cv=5)

# Train the Decision Tree model with best parameters
best_estimator_dt.fit(X_train, y_train)

# Make predictions with both models
y_pred_lr = best_estimator_lr.predict(X_test)
y_pred_prob_lr = best_estimator_lr.predict_proba(X_test)

y_pred_dt = best_estimator_dt.predict(X_test)
y_pred_prob_dt = best_estimator_dt.predict_proba(X_test)

# Evaluate Logistic Regression model
accuracy_lr = accuracy_score(y_test, y_pred_lr)
conf_matrix_lr = confusion_matrix(y_test, y_pred_lr)
class_report_lr = classification_report(y_test, y_pred_lr, zero_division=0)

# ROC AUC Score for multi-class classification
try:
    roc_auc_lr = roc_auc_score(y_test, y_pred_prob_lr, multi_class='ovr')
except ValueError as e:
    print(f"ROC AUC Score for Logistic Regression could not be computed: {e}")
    roc_auc_lr = None

# Evaluate Decision Tree model
accuracy_dt = accuracy_score(y_test, y_pred_dt)
conf_matrix_dt = confusion_matrix(y_test, y_pred_dt)
class_report_dt = classification_report(y_test, y_pred_dt, zero_division=0)

# ROC AUC Score for multi-class classification
try:
    roc_auc_dt = roc_auc_score(y_test, y_pred_prob_dt, multi_class='ovr')
except ValueError as e:
    print(f"ROC AUC Score for Decision Tree could not be computed: {e}")
    roc_auc_dt = None

# Print results for Logistic Regression
print("Logistic Regression Results")
print(f"Best Parameters: {best_params_lr}")
print(f"Cross-Validation Scores: {cv_scores_lr}")
print(f"Mean CV Accuracy: {cv_scores_lr.mean():.2f}")
print(f"Accuracy: {accuracy_lr:.2f}")
print("Confusion Matrix:")
print(conf_matrix_lr)
print("Classification Report:")
print(class_report_lr)
if roc_auc_lr is not None:
    print(f"ROC AUC Score: {roc_auc_lr:.2f}")

# Print results for Decision Tree
print("\nDecision Tree Results")
print(f"Best Parameters: {best_params_dt}")
print(f"Cross-Validation Scores: {cv_scores_dt}")
print(f"Mean CV Accuracy: {cv_scores_dt.mean():.2f}")
print(f"Accuracy: {accuracy_dt:.2f}")
print("Confusion Matrix:")
print(conf_matrix_dt)
print("Classification Report:")
print(class_report_dt)
if roc_auc_dt is not None:
    print(f"ROC AUC Score: {roc_auc_dt:.2f}")

# Plot ROC Curves (only for binary classification)
num_classes = len(np.unique(y))
if num_classes == 2:
    plt.figure()
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_prob_lr[:, 1])
    fpr_dt, tpr_dt, _ = roc_curve(y_test, y_pred_prob_dt[:, 1])
    plt.plot(fpr_lr, tpr_lr, color='blue', lw=2, label='Logistic Regression (area = %0.2f)' % roc_auc_lr)
    plt.plot(fpr_dt, tpr_dt, color='red', lw=2, label='Decision Tree (area = %0.2f)' % roc_auc_dt)
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    # Plot Precision-Recall Curves
    plt.figure()
    precision_lr, recall_lr, _ = precision_recall_curve(y_test, y_pred_prob_lr[:, 1])
    precision_dt, recall_dt, _ = precision_recall_curve(y_test, y_pred_prob_dt[:, 1])
    plt.plot(recall_lr, precision_lr, color='blue', lw=2, label='Logistic Regression')
    plt.plot(recall_dt, recall_dt, color='red', lw=2, label='Decision Tree')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.show()

# Save the trained models
joblib.dump(best_estimator_lr, 'heart_disease_model_lr.pkl')
joblib.dump(best_estimator_dt, 'heart_disease_model_dt.pkl')

# Load the models and make a prediction (for demonstration)
loaded_model_lr = joblib.load('heart_disease_model_lr.pkl')
loaded_model_dt = joblib.load('heart_disease_model_dt.pkl')
sample_data = X_test[0].reshape(1, -1)
sample_prediction_lr = loaded_model_lr.predict(sample_data)
sample_prediction_prob_lr = loaded_model_lr.predict_proba(sample_data)
sample_prediction_dt = loaded_model_dt.predict(sample_data)
sample_prediction_prob_dt = loaded_model_dt.predict_proba(sample_data)

print("\nSample Prediction (Logistic Regression):")
print(f"Prediction: {sample_prediction_lr[0]}")
print(f"Prediction Probability: {sample_prediction_prob_lr}")

print("\nSample Prediction (Decision Tree):")
print(f"Prediction: {sample_prediction_dt[0]}")
print(f"Prediction Probability: {sample_prediction_prob_dt}")
