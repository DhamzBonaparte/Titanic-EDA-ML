import pandas as pd
import numpy as np

df=pd.read_csv('data/tested.csv')

df.info()

df.drop(columns=['PassengerId','Name','Cabin','Fare','Ticket'],inplace=True)
df.info()

from sklearn.model_selection import train_test_split
xTrain,xTest,yTrain,yTest=train_test_split(df.drop(columns=['Survived']),df['Survived'],test_size=0.2,random_state=0)

xTrain.shape

xTrain.isnull().sum()
xTrain.head()

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

ct1=ColumnTransformer([
    ('age',SimpleImputer(strategy='mean'),[2])
],remainder='passthrough')

from sklearn.preprocessing import OneHotEncoder

ct2=ColumnTransformer([
    ('ohe_sex_embark',OneHotEncoder(sparse_output=False,handle_unknown='ignore'),[3,5])
],remainder='passthrough')

from sklearn.preprocessing import StandardScaler

ct3=ColumnTransformer([
    ('scale',StandardScaler(),slice(0,9))
])

from sklearn.tree import DecisionTreeClassifier

dct=DecisionTreeClassifier()

from sklearn.pipeline import Pipeline

pipe=Pipeline([
    ('ct1',ct1),
    ('ct2',ct2),
    ('ct3',ct3),
    ('dct',dct),
])

pipe.fit(xTrain,yTrain)

pipe.named_steps

prediction=pipe.predict(xTest)
prediction

from sklearn.metrics import accuracy_score
acc=accuracy_score(yTest,prediction)
print('Accuracy:',acc)

from sklearn.model_selection import cross_val_score
cross_val=cross_val_score(pipe, xTrain, yTrain, cv=5, scoring='accuracy').mean()
print('Cross-Validation Accuracy:', cross_val)
