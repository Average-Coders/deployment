import array
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
plt.rcParams['figure.figsize'] = (5, 5)
data=pd.read_csv("healthcare-dataset-stroke-data.csv")


corr_matrix = data.corr()
plt.figure(figsize=(20,15))
sns.heatmap(corr_matrix, annot=True)




data.isnull().sum()

data['bmi'].fillna(data['bmi'].mean(),inplace=True)

data.drop('id',axis=1,inplace=True)

from matplotlib.pyplot import figure
figure(num=None, figsize=(8, 6), dpi=800, facecolor='w', edgecolor='k')
data.plot(kind='box')


from sklearn.preprocessing import LabelEncoder
enc=LabelEncoder()

gender=enc.fit_transform(data['gender'])

smoking_status=enc.fit_transform(data['smoking_status'])

work_type=enc.fit_transform(data['work_type'])
Residence_type=enc.fit_transform(data['Residence_type'])
ever_married=enc.fit_transform(data['ever_married'])
data['work_type']=work_type

data['ever_married']=ever_married
data['Residence_type']=Residence_type
data['smoking_status']=smoking_status
data['gender']=gender
data

X=data.drop('stroke',axis=1)
Y=data['stroke']


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.5,random_state=10)




from sklearn.preprocessing import StandardScaler
std=StandardScaler()
X_train_std=std.fit_transform(X_train)
X_test_std=std.transform(X_test)
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=5,criterion="entropy")
rf.fit(X_train_std,Y_train)
Y_pred=rf.predict(X_train_std)
print(Y_pred)

from sklearn.metrics import accuracy_score

ac_rf=accuracy_score(Y_test,Y_pred)


from sklearn.metrics import confusion_matrix
confusion_matrix(Y_test,Y_pred)
tp, fp, fn, tn = confusion_matrix(Y_test, Y_pred).ravel()
(tp, fp, fn, tn)
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score




from sklearn.tree import DecisionTreeClassifier

model=DecisionTreeClassifier()
model.fit(X_train_std,Y_train)
  

   
    
Y_pred_lr=model.predict(X_test_std)
#Y_pred_lr
ac_lr=accuracy_score(Y_test,Y_pred_lr)




pickle.dump(model, open('model.pkl','wb'))
print(data)
# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
