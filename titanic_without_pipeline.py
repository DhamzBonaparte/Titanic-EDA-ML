import numpy as np
import pandas as pd

df=pd.read_csv('data/tested.csv')

df.info()
df.drop(columns=['PassengerId','Name','Cabin','Fare','Ticket'],inplace=True)

from sklearn.model_selection import train_test_split
trainX,testX,trainY,testY=train_test_split(df.drop(columns=['Survived']),df['Survived'],test_size=0.2,random_state=0)

df.isnull().sum()

from sklearn.impute import SimpleImputer
ageImputer=SimpleImputer(strategy='mean')
ageImputer.fit(trainX[['Age']])
trainX['Age']=ageImputer.transform(trainX[['Age']])
testX['Age']=ageImputer.transform(testX[['Age']])
testX

from sklearn.preprocessing import StandardScaler
ageScaler=StandardScaler()
ageScaler.fit(trainX[['Age']])
trainX['Age']=ageScaler.transform(trainX[['Age']])
testX['Age']=ageScaler.transform(testX[['Age']])
np.round(trainX['Age'].describe())

from sklearn.preprocessing import OneHotEncoder
Sexohe=OneHotEncoder(sparse_output=False,handle_unknown='ignore')
Embohe=OneHotEncoder(sparse_output=False,handle_unknown='ignore')

trainsex=Sexohe.fit_transform(trainX[['Sex']])
trainemb=Embohe.fit_transform(trainX[['Embarked']])

testsex=Sexohe.transform(testX[['Sex']])
testemb=Embohe.transform(testX[['Embarked']])

trainX.head(2)

trainX_rem=trainX.drop(columns=['Sex','Embarked'])
testX_rem=testX.drop(columns=['Sex','Embarked'])

trainXTransformed=np.concatenate((trainX_rem,trainsex,trainemb),axis=1)
testXTransformed=np.concatenate((testX_rem,testsex,testemb),axis=1)

trainXTransformed.shape

from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier()
clf.fit(trainXTransformed,trainY)

prediction=clf.predict(testXTransformed)

from sklearn.metrics import accuracy_score
acc=accuracy_score(testY,prediction)
print("The accuracy is: "+str(acc)+'/1.0')

