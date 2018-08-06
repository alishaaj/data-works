# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 12:11:53 2018

@author: alish
"""
# Classification models for traffic rank prediction

#Importing libraries
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('traffic rank_data.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, 14].values

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, :])
X[:, :] = imputer.transform(X[:, :])

# Encoding the Dependent Variable
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
X = sc_X.transform(X)


#k-fold cross validation splitting and classification
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

#Applying KNN Classifier
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors =5, metric = 'minkowski', p=1 )  #p=1 manhattan, p=2 #eucledian
kf = KFold(n_splits=10)
sum = 0
for train, test in kf.split(X):
     X_train, X_test = X[train], X[test]
     y_train, y_test = y[train], y[test]
     classifier.fit(X_train,y_train)
     y_pred = classifier.predict(X_test)
     sum += accuracy_score(y_test, y_pred)
average = sum/10
score_knn5m = average
print("Accuracy:KNN 5 Manhattan ",average)

classifier = KNeighborsClassifier(n_neighbors =5, metric = 'minkowski', p=2 )  #p=1 manhattan, p=2 #eucledian
kf = KFold(n_splits=10)
sum = 0
for train, test in kf.split(X):
     X_train, X_test = X[train], X[test]
     y_train, y_test = y[train], y[test]
     classifier.fit(X_train,y_train)
     y_pred = classifier.predict(X_test)
     sum += accuracy_score(y_test, y_pred)
average = sum/10
score_knn5e = average
print("Accuracy:KNN 5 Euclidean ",average)

classifier = KNeighborsClassifier(n_neighbors =11, metric = 'minkowski', p=1 )  #p=1 manhattan, p=2 #eucledian
kf = KFold(n_splits=10)
sum = 0
for train, test in kf.split(X):
     X_train, X_test = X[train], X[test]
     y_train, y_test = y[train], y[test]
     classifier.fit(X_train,y_train)
     y_pred = classifier.predict(X_test)
     sum += accuracy_score(y_test, y_pred)
average = sum/10
score_knn11m = average
print("Accuracy:KNN 11 Manhattan ",average)

classifier = KNeighborsClassifier(n_neighbors =11, metric = 'minkowski', p=2 )  #p=1 manhattan, p=2 #eucledian
kf = KFold(n_splits=10)
sum = 0
for train, test in kf.split(X):
     X_train, X_test = X[train], X[test]
     y_train, y_test = y[train], y[test]
     classifier.fit(X_train,y_train)
     y_pred = classifier.predict(X_test)
     sum += accuracy_score(y_test, y_pred)
average = sum/10
score_knn11e = average
print("Accuracy:KNN 11 Euclidean ",average)

#Applying SVC Classifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
classifier= SVC(kernel='linear', random_state=0)
kf = KFold(n_splits=10)
sum = 0
for train, test in kf.split(X):
     X_train, X_test = X[train], X[test]
     y_train, y_test = y[train], y[test]
     classifier.fit(X_train,y_train)
     y_pred = classifier.predict(X_test)
     sum += accuracy_score(y_test, y_pred)
average = sum/10
score_svcl = average
print("Accuracy:SVC_Linear",average)

classifier= SVC(kernel='rbf', random_state=0)
kf = KFold(n_splits=10)
sum = 0
for train, test in kf.split(X):
     X_train, X_test = X[train], X[test]
     y_train, y_test = y[train], y[test]
     classifier.fit(X_train,y_train)
     y_pred = classifier.predict(X_test)
     sum += accuracy_score(y_test, y_pred)
average = sum/10
score_svcr = average
print("Accuracy:SVC_RBF ",average)

#Applying Decision tree Classifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
kf = KFold(n_splits=10)
sum = 0
for train, test in kf.split(X):
     X_train, X_test = X[train], X[test]
     y_train, y_test = y[train], y[test]
     classifier.fit(X_train,y_train)
     y_pred = classifier.predict(X_test)
     sum += accuracy_score(y_test, y_pred)
average = sum/10
score_dt = average
print("Accuracy:DT ",average)

#Applying Random Forest classifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
kf = KFold(n_splits=10)
sum = 0
for train, test in kf.split(X):
     X_train, X_test = X[train], X[test]
     y_train, y_test = y[train], y[test]
     classifier.fit(X_train,y_train)
     y_pred = classifier.predict(X_test)
     sum += accuracy_score(y_test, y_pred)
average = sum/10
score_rf10 = average
print("Accuracy:RF10 ",average)

classifier=RandomForestClassifier(n_estimators=20,criterion='entropy',random_state=0)
kf = KFold(n_splits=10)
sum = 0
for train, test in kf.split(X):
     X_train, X_test = X[train], X[test]
     y_train, y_test = y[train], y[test]
     classifier.fit(X_train,y_train)
     y_pred = classifier.predict(X_test)
     sum += accuracy_score(y_test, y_pred)
average = sum/10
score_rf20 = average
print("Accuracy:RF20 ",average)


#Summary of accuracy measures of all classifiers
new=pd.DataFrame({
"pred_accuracy":[score_knn5m,score_knn5e,score_knn11m,score_knn11e,score_svcl,score_svcr,score_dt,score_rf10,score_rf20],
"classifier":['KNN-5-manhattan','KNN-5-euclidean','KNN-11-manhattan','KNN-11-euclidean','SVC-linear','SVC-radialBiasFn','DT','RF-10','RF-20']
})
print("Summary: \n",new)   

