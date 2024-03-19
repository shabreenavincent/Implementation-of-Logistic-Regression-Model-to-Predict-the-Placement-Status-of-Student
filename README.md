# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Get the data and use label encoder to change all the values to numeric
2. Classify the training data and the test data
3. Calculate the accuracy score, confusion matrix and classification report
4. 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: SHABREENA VINCENT
RegisterNumber: 212222230141

import pandas as pd
data=pd.read_csv('/content/Placement_Data (1).csv')
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1) #removes the specified row or cols
data1.head() 

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x
y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver="liblinear") 
lr.fit(x_train,y_train)
y_pred= lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1= classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
*/
```

## Output:
#### Placement data
![Screenshot 2023-05-10 092032](https://github.com/Yamunaasri/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/115707860/a277624a-9c25-419a-973c-347d9c6d9c71)


#### Salary data
![Screenshot 2023-05-10 093619](https://github.com/Yamunaasri/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/115707860/4407941c-a7e9-4489-a204-e7d33511c2c1)


#### isnull()
![Screenshot 2023-05-10 092302](https://github.com/Yamunaasri/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/115707860/d9faca11-18b1-446b-b7a2-77b261a746c3)


#### Checking for duplicates
![Screenshot 2023-05-10 092346](https://github.com/Yamunaasri/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/115707860/33d7fe4b-668c-4ddb-8f7f-0e223421787c)


#### Print data
![Screenshot 2023-05-10 093925](https://github.com/Yamunaasri/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/115707860/0f5771a8-3104-446d-b43e-67ebbecfecc6)


#### Data status
![Screenshot 2023-05-10 093930](https://github.com/Yamunaasri/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/115707860/9f0d528c-d3a2-4030-8a01-b4b142192ef9)


#### y_prediction array
![Screenshot 2023-05-10 092601](https://github.com/Yamunaasri/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/115707860/49e74f57-fb75-4540-9f7e-362526c0d419)


#### Accuracy score
![Screenshot 2023-05-10 092606](https://github.com/Yamunaasri/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/115707860/1fee2a32-d47d-4b92-8ee0-17aac937ccd7)


#### Confusion Matrix
![Screenshot 2023-05-10 092613](https://github.com/Yamunaasri/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/115707860/2dd5d307-0eea-44b3-8cd0-9234ee940b05)


#### Classification report
![Screenshot 2023-05-10 092624](https://github.com/Yamunaasri/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/115707860/1d3616f3-4dff-4c0c-8891-84c9c73426a1)


#### Prediction of LR
![Screenshot 2023-05-11 154321](https://github.com/Yamunaasri/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/115707860/07c86990-d7e6-4aa1-952a-5b7c464d51c5)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
