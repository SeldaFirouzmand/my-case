# Case Study: Credit Card Default Dataset

## Problem Statement

When a customer accepts a credit card from a bank or issuer, they agree to terms and conditions, including making minimum payments by the due dates listed on their credit card statements. Failure to meet these payment deadlines results in the issuer marking the credit card as default. Consequences may include penalties, reduced credit limits, or, in cases of serious delinquency, account closure.

In pursuit of market share, some credit card-issuing banks issue cards to unqualified clients without sufficient information about their repayment ability. When cardholders overuse their cards for goods and services beyond their financial capacity, they accumulate significant debt. In consumer finance, it is critical for banks to estimate the likelihood of a cardholder defaulting. This analysis supports risk assessment and informs decisions on credit card approvals.

This case study focuses on analyzing a dataset to understand and predict credit card defaults based on customer demographics, credit limits, payment history, and billing information.

## Dataset Description

The dataset contains information about credit card holders and their payment behavior over a six-month period (April 2005 to September 2005). Below is a description of the key variables:

- **SEX**: Gender
  - 1 = Male
  - 2 = Female

- **EDUCATION**: Education level
  - 1 = Graduate school
  - 2 = University
  - 3 = High school
  - 4 = Others

- **MARRIAGE**: Marital status
  - 1 = Married
  - 2 = Single
  - 3 = Others

- **AGE**: Age (in years)

- **LIMIT_BAL**: Amount of given credit (New Taiwan Dollar)
  - Includes individual consumer credit and supplementary family credit.

- **PAY_1 - PAY_6**: History of past payment (April 2005 to September 2005)
  - PAY_1 = Repayment status in September 2005
  - PAY_2 = Repayment status in August 2005
  - ...
  - PAY_6 = Repayment status in April 2005
  - Scale:
    - -1 = Paid on time (duly)
    - 1 = Payment delayed by 1 month
    - 2 = Payment delayed by 2 months
    - ...
    - 8 = Payment delayed by 8 months
    - 9 = Payment delayed by 9 months or more

- **BILL_AMT1 - BILL_AMT6**: Amount of bill statement (New Taiwan Dollar)
  - BILL_AMT1 = Bill amount in September 2005
  - BILL_AMT2 = Bill amount in August 2005
  - ...
  - BILL_AMT6 = Bill amount in April 2005

- **PAY_AMT1 - PAY_AMT6**: Amount of previous payment (New Taiwan Dollar)
  - PAY_AMT1 = Payment made in September 2005
  - PAY_AMT2 = Payment made in August 2005
  - ...
  - PAY_AMT6 = Payment made in April 2005

## Tasks
For this assignment, you will analyze the Credit Card Default Dataset to predict whether a cardholder will default on their next payment. Your task is to:

- Load and preprocess the dataset as outlined in the starter code.
- Clean the `PAY_` features by converting values of -1 and -2 to 0 (indicating no payment delay).
- Standardize numerical features and create dummy variables for categorical features (`SEX`, `EDUCATION`, `MARRIAGE`).
- Split the data into training (80%) and testing (20%) sets.
- Build and train three models: Random Forest Classifier, Decision Tree Classifier, and Logistic Regression.
- Evaluate the models using training accuracy, testing accuracy, and AUROC (Area Under the ROC Curve) scores.
- Complete the provided starter code to output a performance comparison table.

The dataset is located in the `data` folder as `credit_card_default.csv`. Use the starter code and submit your completed `starter_code.py` file via GitHub Classroom, ensuring it passes all tests in the test script.
