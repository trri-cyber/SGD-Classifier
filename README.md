# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Dataset Loading & Preparation: The Iris dataset is loaded using sklearn.datasets, converted into a DataFrame, and the target column (species labels) is added.

2.Feature and Target Splitting: Features (x) and labels (y) are separated, then split into training and testing sets using an 80-20 ratio.

3.Model Initialization & Training: A SGDClassifier is initialized with a maximum of 1000 iterations and trained using the training data.

4.Prediction: The model predicts species labels on the test set.

5.Evaluation: Model performance is evaluated using accuracy, confusion matrix, and classification report (precision, recall, f1-score).

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: Rishab p doshi
RegisterNumber:  212224240134
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
iris=load_iris()
df=pd.DataFrame(iris.data,columns=iris.feature_names)
df['target']=iris.target
print(df.head())

X=df.drop('target',axis=1)
y=df['target']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/2,random_state=42)
from sklearn.linear_model import SGDClassifier
sgd_clf=SGDClassifier()
sgd_clf.fit(X_train,y_train)
y_pred=sgd_clf.predict(X_test)
from sklearn.metrics import accuracy_score,confusion_matrix
acc=accuracy_score(y_test,y_pred)
print(acc)
con=confusion_matrix(y_test,y_pred)
print(con)
```
## Output:
![image](https://github.com/user-attachments/assets/f7b8c801-0a62-4dd6-9613-aa7db2ee0326)



## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
