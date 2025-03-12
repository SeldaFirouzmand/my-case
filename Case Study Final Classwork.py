import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve


data_path = 'data\credit_card_default.csv'

# Load the dataset into a pandas DataFrame and set the index to the "ID" column
ccd = pd.read_csv(data_path, index_col="ID")

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

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

# Load the dataset into a pandas DataFrame and set the index to the "ID" column
ccd = pd.read_csv(data_path, index_col="ID")
# Understand the data
ccd.describe()

# Clean PAY_ features
pay_features = ['PAY_' + str(i) for i in range(1,7)]
# Loop through each 'pay_' feature
for x in pay_features:
# Transform the -1 and -2 values to 0, assuming these represent non-defaulted payments
    ccd.loc[ccd[x] <= 0, x] = 0
# Rename the 'default payment next month' column to 'default' for brevity
ccd.rename(columns={'default payment next month':'default'}, inplace=True)
categorical_features = ['SEX', 'EDUCATION', 'MARRIAGE']
numerical_features = ['LIMIT_BAL', 'AGE', 'PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', \
'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', \
'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4',\
'PAY_AMT5', 'PAY_AMT6']
target_feature = ['default']

scaler = StandardScaler()
ccd[numerical_features] = scaler.fit_transform(ccd[numerical_features])
ccd = pd.get_dummies(ccd, columns=categorical_features, drop_first=True)

X = ccd.drop(columns=['default'])
y = ccd['default']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_train_pred = rf_model.predict(X_train)
rf_test_pred = rf_model.predict(X_test)
rf_test_proba = rf_model.predict_proba(X_test)[:, 1]
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_train_pred = dt_model.predict(X_train)
dt_test_pred = dt_model.predict(X_test)
dt_test_proba = dt_model.predict_proba(X_test)[:, 1]
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_train_pred = lr_model.predict(X_train)
lr_test_pred = lr_model.predict(X_test)
lr_test_proba = lr_model.predict_proba(X_test)[:, 1]

models = ['Random Forest', 'Decision Tree', 'Logistic Regression']
train_accuracies = [
accuracy_score(y_train, rf_train_pred),
accuracy_score(y_train, dt_train_pred),
accuracy_score(y_train, lr_train_pred)
]
test_accuracies = [
accuracy_score(y_test, rf_test_pred),accuracy_score(y_test, dt_test_pred),
accuracy_score(y_test, lr_test_pred)
]
auroc_scores = [
roc_auc_score(y_test, rf_test_proba),
roc_auc_score(y_test, dt_test_proba),
roc_auc_score(y_test, lr_test_proba)
]
results_df = pd.DataFrame({
'Model': models,
'Training Accuracy': [f"{x:.4f}" for x in train_accuracies],
'Testing Accuracy': [f"{x:.4f}" for x in test_accuracies],
'AUROC Score': [f"{x:.4f}" for x in auroc_scores]
})
# Print the table
print("Model Performance Comparison:")
print(results_df)

plt.figure(figsize=(12, 8))
plot_tree(dt_model, max_depth=2, class_names=['No Default', 'Default'],filled=True)
plt.show()

# Calculate ROC curves
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_test_proba)
dt_fpr, dt_tpr, _ = roc_curve(y_test, dt_test_proba)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_test_proba)
# ax3 = plt.plot(1, 1, 1)
plt.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {auroc_scores[0]:.3f})', color='blue')
plt.plot(dt_fpr, dt_tpr, label=f'Decision Tree (AUC = {auroc_scores[1]:.3f})', color='orange')
plt.plot(lr_fpr, lr_tpr, label=f'Logistic Regression (AUC = {auroc_scores[2]:.3f})', color='green')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

 # Create probability distribution plots
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
 # Random Forest
sns.histplot(rf_test_proba, bins=20, kde=True, ax=axes[0], color='blue')
axes[0].set_title('Random Forest Probability Distribution')
axes[0].set_xlabel('Predicted Probability')
axes[0].set_ylabel('Frequency')
 # Decision Tree
sns.histplot(dt_test_proba, bins=20, kde=True, ax=axes[1], color='orange')

axes[1].set_title('Decision Tree Probability Distribution')
axes[1].set_xlabel('Predicted Probability')
 # Logistic Regression
sns.histplot(lr_test_proba, bins=20, kde=True, ax=axes[2], color='green')
axes[2].set_title('Logistic Regression Probability Distribution')
axes[2].set_xlabel('Predicted Probability')
plt.tight_layout()
plt.show()