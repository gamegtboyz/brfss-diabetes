# import  libraries
import numpy as np
import pandas as pd
import scipy
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# import balanced data from .csv files
data_balanced = pd.read_csv("BRFSS23_diabetes_balanced.csv")

# split the data into train and test datasets
x = data_balanced[['GENHLTH', 'BPHIGH6', '_BMI5', '_AGEG5YR', 'TOLDHI3']]
y = data_balanced['DIABETE4']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

# tune the parameters
c_inverse = [0.01, 0.1, 1, 10, 100]
max_iter = range(100, 1000, 200)
solver = ['lbfgs', 'liblinear', 'saga']
penalty = ['l2', 'l1']

result = []

for c in c_inverse:
    for m in max_iter:
        for s in solver:
            for p in penalty:
                try:
                    print(f"running on c = {c}, m = {m}, s = {s}, p = {p}")
                    LR_accuracy = cross_val_score(
                        LogisticRegression(C = c, max_iter = m, solver = s, penalty = p),
                        x_train,
                        y_train,
                        cv=StratifiedKFold(n_splits=10),
                        scoring='accuracy',
                        error_score='raise'
                    )                    
                    result.append([c, m, s, p , LR_accuracy.mean()])
                
                except Warning as w:
                    continue
                
                except Exception as e:
                    print(e)

# create dataframe from the list
results_df = pd.DataFrame(result, columns = ['c_inverse', 'max_inter', 'solver', 'penalty', 'accuracy'])

# sort the accuracy, then export as .csv files
results_df.sort_values('accuracy', ascending=False).head(10)
results_df.to_csv('parameter-tuning-results.csv')