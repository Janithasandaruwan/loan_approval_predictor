import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


# Load the data
data = pd.read_csv("loan_data.csv")

# Handle missing values
data['AnnualIncome'] = data['AnnualIncome'].fillna(data['AnnualIncome'].median())

# Normalize continuous variables
for col in ['AnnualIncome', 'CreditScore', 'DebtRatio']:
    data[col] = (data[col] - data[col].mean()) / data[col].std()

# Convert categorical variables to dummy variables if needed
data = pd.get_dummies(data, columns=['MaritalStatus'], drop_first=True)

# Split data into features (X) and target (y)
X = data[['CreditScore', 'Income', 'DebtRatio']]
y = data['LoanApproved']

# Train-test split (70/30 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)



# Add constant for the intercept
X_train_const = sm.add_constant(X_train)

# Fit the Probit model
probit_model = sm.Probit(y_train, X_train_const).fit()

# Display model summary
print(probit_model.summary())
