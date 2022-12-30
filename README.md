
GRIP : The Sparks Foundation
Data Science & Business Analytics
#GRIPDEC22
Name : Aya Abdelghaffar
Task 1 : predicting using Supervised ML
predict the percentage of a student based on the no. of study hours
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression  
from sklearn import metrics 
from sklearn.metrics import r2_score
# Reading The Data
data=pd.read_csv("data.csv")
data.head()
Hours	Scores
0	2.5	21
1	5.1	47
2	3.2	27
3	8.5	75
4	3.5	30
data.shape
(25, 2)
data.info() 

RangeIndex: 25 entries, 0 to 24
Data columns (total 2 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   Hours   25 non-null     float64
 1   Scores  25 non-null     int64  
dtypes: float64(1), int64(1)
memory usage: 528.0 bytes
data.describe()
Hours	Scores
count	25.000000	25.000000
mean	5.012000	51.480000
std	2.525094	25.286887
min	1.100000	17.000000
25%	2.700000	30.000000
50%	4.800000	47.000000
75%	7.400000	75.000000
max	9.200000	95.000000
X=data.drop(columns='Scores',axis=1)
y=data['Scores']
plt.figure(figsize=(10,10),dpi=80)
plt.scatter(X,y)
plt.title("Hours Vs Score") 
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

We see a linear relation between hours and score so we can use linear regression.
# Split the data 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=82 )
# Model
regressor=LinearRegression()
regressor.fit(X_train,y_train)
LinearRegression()
# Draw Best Fit Line
line=regressor.predict(X)
plt.figure(figsize=(10,10),dpi=80)
plt.title("Best fit line") 
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.scatter(X,y)
plt.plot (X,line)
plt.show

Predict the score if the student study 9.25 hours'
x=[[9.25 ]]
print(regressor.predict(x))
[94.46128383]
/home/aya/snap/jupyter/common/lib/python3.7/site-packages/sklearn/base.py:451: UserWarning: X does not have valid feature names, but LinearRegression was fitted with feature names
  "X does not have valid feature names, but"
Model Evaluation
y_pred=regressor.predict(X_test)
print("Mean Absolute Error : ",metrics.mean_absolute_error(y_test,y_pred))
print("R2 Score : ",r2_score(y_test,y_pred))
Mean Absolute Errror :  3.6707965195714243
R2 Score :  0.9810815915350791
 
