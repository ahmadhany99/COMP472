import json as js
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

#Part 2.1
f = open('goemotions.json')
file = js.load(f)
split = np.array(file)
x = split[:, 0]
y1 = split[:, 1]
y2 = split[:, 2]

# Part 2.2
x1_train, x1_test, y1_train, y1_test = train_test_split(x, y1, train_size=0.8, test_size=0.2)
x2_train, x2_test, y2_train, y2_test = train_test_split(x, y2, train_size=0.8, test_size=0.2)

# Part 2.3.5
print("Using a better performing Naive Bayes Classifier found using GridSearchCV.")
vec=CountVectorizer()

x1_train = vec.fit_transform(x1_train)
x1_test = vec.transform(x1_test)
nb_classifier = MultinomialNB()
param_dict={"alpha":[0.5, 0, 1, 10.0]}
grid1 = GridSearchCV(estimator=nb_classifier,param_grid=param_dict,n_jobs=-1,error_score='raise')
grid1.fit(x1_train, y1_train)
y1_pred = grid1.predict(x1_test)
print("For emotions : \n")
print(classification_report(y1_test, y1_pred))
print("Confusion Matrix: \n", confusion_matrix(y1_test, y1_pred))

x2_train = vec.fit_transform(x2_train)
x2_test = vec.transform(x2_test)
grid2=GridSearchCV(estimator=nb_classifier,param_grid=param_dict,n_jobs=-1,error_score='raise')
grid2.fit(x2_train, y2_train)
y2_pred = grid2.predict(x2_test)
print("For sentiments : \n")
print(classification_report(y2_test, y2_pred))
print("Confusion Matrix: \n", confusion_matrix(y2_test, y2_pred))
f.close()

