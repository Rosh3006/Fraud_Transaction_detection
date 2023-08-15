import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df=pd.read_csv("creditcard.csv")
pd.set_option('display.max_columns',None)
df=df.drop(columns =['Time'])
print(df.head())
print(df.isnull().sum())
print(df.groupby('Class').value_counts())

legit = df[df.Class == 0]
fraud= df[df.Class == 1]
print(legit.shape)
print(fraud.shape)
print(legit.Amount.describe())
print(fraud.Amount.describe())

print(df.groupby('Class').mean())

#Under sampling
legit_sample=legit.sample(n=492)
new_df=pd.concat([legit_sample,fraud],axis=0)
print(new_df.head())
print(new_df['Class'].value_counts())
print(new_df.groupby('Class').mean())


X=new_df.drop(columns='Class',axis=1)
Y=new_df["Class"]


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.30)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit(x_train)
x_train= sc.transform(x_train)
x_test= sc.transform(x_test)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
print("Accuracy of Linear Regression:",lr.score(x_test,y_test)*100)
from sklearn.linear_model import LogisticRegression
lor=LogisticRegression()
lor.fit(x_train,y_train)
print("Accuracy of Logistic Regression:",lor.score(x_test,y_test)*100)
from sklearn.neighbors import KNeighborsClassifier
knc=KNeighborsClassifier()
knc.fit(x_train,y_train)
print("Accuracy of KNC:",knc.score(x_test,y_test)*100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)
print("Accuracy of Decision Tree:",dt.score(x_test,y_test)*100)
from sklearn import svm
classifer=svm.SVC(kernel='linear')
classifer.fit(x_train,y_train)
print("Accuracy of SVC:",classifer.score(x_test,y_test)*100)

from sklearn import svm
regression=svm.SVR(kernel='linear')
regression.fit(x_train,y_train)
print("Accuracy of SVR:",regression.score(x_test,y_test)*100)

from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=100,random_state=0)
regressor.fit(x_train,y_train)
print("Accuracy of Random Forest Regressor:",regressor.score(x_test,y_test)*100)

from sklearn.ensemble import RandomForestClassifier
RandomForest = RandomForestClassifier(oob_score=True)
RandomForest.fit(x_train,y_train)
print(RandomForest.oob_score_)
print("Accuracy of Random Forest Classifier:",RandomForest.score(x_test,y_test)*100)

from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(x_train,y_train)
print("Accuracy of Naive Bayes Classifier:",classifier.score(x_test,y_test)*100)

import pickle
filename="saveModelfraud.sav"
pickle.dump(RandomForest,open(filename,'wb'))
load_model=pickle.load(open(filename,'rb'))
y=load_model.predict([[-0.997800020905509,1.0999501830901,1.43856577993145,-1.36996217198801,0.364836284400598,-0.41923590434811,0.929916694490226,-0.471805090190641,1.58804771927219,-0.41675287541644,0.0888689721223715,-2.17711417342942,3.17147432446441,0.694401672332279,-0.247944447007421,0.643571289378497,-0.463262712791586,-0.0317459609743358,-0.594996980278217,0.208877184138451,-0.346407044793429,-0.399724872772842,-0.139499563278151,-0.446227832449058,-0.185711278122289,0.683184041588875,-0.0542561670724449,-0.00661477381074577,37.96]])
print(y)