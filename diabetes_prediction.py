import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

#loading the diabetes dataset to a pandas Dataframe

diabetes_dataset  = pd.read_csv(r'C:\Users\Skylink\Desktop\Diabetes Prediction\diabetes.csv')

#pd.read_csv?

# print(diabetes_dataset.head())

#number of rows and columns in this dataset
# print(diabetes_dataset.shape)

#getting the statistical measures of the data
# print(diabetes_dataset.describe())

#total outcome values in this dataset
print(diabetes_dataset['Outcome'].value_counts())

# 0 - Non-Diabetic
# 1 - Diabetic

print(diabetes_dataset.groupby('Outcome').mean())