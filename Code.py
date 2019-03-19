import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

data = pd.read_csv('/Users/nagarjuna/PycharmProjects/Loan_prediction_analysis/loan_data_set.csv')

train, test = train_test_split(data, test_size = 0.2, random_state = 0)

#Over view of data
train.describe()
train.head(10)
train.sample()

train_original = train.copy()
test_original = test.copy()

train.columns # displays columns

train.dtypes  #type of each column


dataset = pd.concat([train,test]) #to join both the datasets

train['Loan_Status'].value_counts()
train['Loan_Status'].value_counts(normalize = True) #Percentage

train['Loan_Status'].value_counts().plot.bar() #Bar graph

cross = pd.crosstab(train['Gender'], train['Loan_Status'])

train.isnull().sum()

#Filling the empty cells for train data

train.isnull().sum() #Checks for empty cells

#Filling empty cells fobinary variables with most frequent ones

train['Gender'].fillna(train['Gender'].mode()[0], inplace = True)
train['Married'].fillna(train['Married'].mode()[0], inplace = True)
train['Dependents'].fillna(train['Dependents'].mode()[0], inplace = True)
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace = True)

#Checking for Loan_Amount, Loan_Amount term and credit history

train['LoanAmount'].fillna(train['LoanAmount'].median(), inplace = True)
train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].median(), inplace = True)
train['Credit_History'].fillna(train['Credit_History'].median(), inplace = True)


#Filling the empty cells for test data

test.isnull().sum()
test['Self_Employed'].fillna(test['Self_Employed'].mode()[0], inplace = True)
train['LoanAmount'].fillna(test['LoanAmount'].median(), inplace = True)
test['Loan_Amount_Term'].fillna(test['Loan_Amount_Term'].mode()[0], inplace = True)
test['Credit_History'].fillna(test['Credit_History'].mode()[0], inplace = True)

#Logistic Regression

train = train.drop('Loan_ID',axis =1)
test = test.drop('Loan_ID',axis =1)

train_r = train['Loan_Status']

train = train.drop('Loan_Status', axis = 1)

test_r = test['Loan_Status']

test = test.drop('Loan_Status', axis = 1)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = LogisticRegression()
model.fit(train, train_r)




