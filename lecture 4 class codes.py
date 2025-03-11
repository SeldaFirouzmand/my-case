## Data Loading
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

data_path = 'credit_card_default.csv'
# Load the dataset into a pandas DataFrame and set the index to the "ID" column
ccd = pd.read_csv(data_path, index_col="ID")
# Understand the data
print(ccd.describe())


 #Clean PAY_ features
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
# Building Models
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

 # Model Performance Comparison
models = ['Random Forest', 'Decision Tree', 'Logistic Regression']
train_accuracies = [
accuracy_score(y_train, rf_train_pred),
accuracy_score(y_train, dt_train_pred),
accuracy_score(y_train, lr_train_pred)
]
test_accuracies = [accuracy_score(y_test, rf_test_pred),
accuracy_score(y_test, dt_test_pred),
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