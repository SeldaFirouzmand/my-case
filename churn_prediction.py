
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve


data_path = 'credit_card_default.csv'

# Load the dataset into a pandas DataFrame and set the index to the "ID" column
ccd = pd.read_csv(data_path, index_col="ID")
print(ccd.head())

# Understand the data (uncomment to explore)
print(ccd.describe())

# Initial Processing
# Clean PAY_ features
pay_features = ['PAY_' + str(i) for i in range(1, 7)]
for x in pay_features:
    # Transform -1 and -2 values to 0 (assuming these represent non-defaulted payments)
    ccd.loc[ccd[x] <= 0, x] = 0

# Rename 'default payment next month' to 'default' for brevity
ccd.rename(columns={'default payment next month': 'default'}, inplace=True)

# Define feature categories
categorical_features = ['SEX', 'EDUCATION', 'MARRIAGE']
numerical_features = ['LIMIT_BAL', 'AGE', 'PAY_1', 'PAY_2', 'PAY_3', 'PAY_4',
                      'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3',
                      'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2',
                      'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
target_feature = ['default']

# Standardize Numeric Features and Create Dummy Variables
scaler = StandardScaler()
ccd[numerical_features] = scaler.fit_transform(ccd[numerical_features])
ccd = pd.get_dummies(ccd, columns=categorical_features, drop_first=True)

# Splitting Data into Train and Test
X = ccd.drop(columns=['default'])
y = ccd['default']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TODO: Build Models
# 1. Random Forest Classifier
# 2. Decision Tree Classifier
# 3. Logistic Regression
# For each model:
# - Fit the model on the training data
# - Predict on both training and testing data
# - Compute probabilities for the test set (for AUROC)

# TODO: Model Performance Comparison
# - Calculate training accuracy, testing accuracy, and AUROC scores for each model
# - Create a DataFrame to display the results
# - Print the comparison table

# Example structure for results (to be completed):
# models = ['Random Forest', 'Decision Tree', 'Logistic Regression']
# results_df = pd.DataFrame({
#     'Model': models,
#     'Training Accuracy': [...],
#     'Testing Accuracy': [...],
#     'AUROC Score': [...]
# })
# print("Model Performance Comparison:")
# print(results_df)
