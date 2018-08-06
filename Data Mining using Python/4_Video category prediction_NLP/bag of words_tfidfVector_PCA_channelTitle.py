# -*- coding: utf-8 -*-
"""
Created on Wed May  2 17:13:21 2018

@author: alish
"""


# Natural Language Processing for Video Title prediction

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk

# Importing the dataset
dataset = pd.read_csv('video category data_usa.csv', encoding='latin1')

# Cleaning the texts
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Generating corpus for Channel title
corpus_ch_title = []
for i in range(0, 23358):
    ch_title = re.sub('[^a-zA-Z]', ' ', dataset['channel_title'][i])
    ch_title = ch_title.lower()
    ch_title = ch_title.split()
    ps = PorterStemmer()
    ch_title = [ps.stem(word) for word in ch_title if not word in set(stopwords.words('english'))]
    ch_title = ' '.join(ch_title)
    corpus_ch_title.append(ch_title)

# Creating the Bag of Words model using TF-IDF Vector
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer()
X = cv.fit_transform(corpus_ch_title).toarray()
y = dataset.iloc[:, 5].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# import accuracy metrics for classifier evaluation
from sklearn.metrics import accuracy_score
"""
# Applying PCA and identifyng the optimal PCA

# Commented after identifying, and re-run the code with optimal PC count
from sklearn.decomposition import PCA
pca = PCA(n_components = None)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

#identify the PCA components by cumulative sum of variances
var=np.cumsum(np.round(explained_variance, decimals=3)*100)
plt.ylabel('% Variance Explained')
plt.xlabel('# of Features')
plt.title('PCA Analysis')
plt.ylim(30,100.5)
plt.plot(var)"""

from sklearn.decomposition import PCA
pca = PCA(n_components = 550)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

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