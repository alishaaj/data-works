# -*- coding: utf-8 -*-
"""
Created on Tue May  1 18:17:04 2018

@author: alish
"""
# Natural Language Processing for Video Title prediction

# Importing the libraries
import pandas as pd
import re
import nltk

# Importing the dataset
dataset = pd.read_csv('video category data_usa.csv', encoding='latin1')

# Cleaning the texts
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Generating corpus for Title
corpus_title = []
for i in range(0, 23358):
    title = re.sub('[^a-zA-Z]', ' ', dataset['title'][i])
    title = title.lower()
    title = title.split()
    ps = PorterStemmer()
    title = [ps.stem(word) for word in title if not word in set(stopwords.words('english'))]
    title = ' '.join(title)
    corpus_title.append(title)

# Creating the Bag of Words model using Count Vector
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus_title).toarray()
y = dataset.iloc[:, 5].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# import accuracy metrics for classifier evaluation
from sklearn.metrics import accuracy_score

# Applying classifiers to Bag of Words Model

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifierNB = GaussianNB()
classifierNB.fit(X_train, y_train)
# Predicting the Test set results
y_predNB = classifierNB.predict(X_test)
# Calculating accuracy score
score_nb = accuracy_score(y_test, y_predNB)


# Fitting Decision Tree Classifier to the Training set
from sklearn.tree import DecisionTreeClassifier
classifierDT = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifierDT.fit(X_train,y_train)
# Predicting the Test set results
y_predDT = classifierDT.predict(X_test)
# Calculating accuracy score
score_dt=accuracy_score(y_test, y_predDT)


# Fitting Decision Tree Classifier to the Training set
from sklearn.ensemble import RandomForestClassifier
classifierRF=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
classifierRF.fit(X_train,y_train)
# Predicting the Test set results
y_predRF = classifierRF.predict(X_test)
# Calculating accuracy score
score_rf=accuracy_score(y_test, y_predRF)

# Summary of accuracy measures of all classifiers
summary=pd.DataFrame({
"pred_accuracy":[score_nb,score_dt,score_rf],
"classifier":['Naive Bayes','Decision Tree','Random Forest']
})
print("Summary: \n",summary)