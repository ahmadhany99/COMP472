import json as js
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
import pickle
from sklearn.metrics import classification_report, confusion_matrix

f = open('goemotions.json')
file = js.load(f)
# Part 2.1
posts = [item[0] for item in file]
vec = CountVectorizer()
vec.fit(posts)
print(len(vec.vocabulary_))
split = np.array(file)
x = split[:, 0]
y1 = split[:, 1]
y2 = split[:, 2]
# Part 2.2
x1_train, x1_test, y1_train, y1_test = train_test_split(x, y1, train_size=0.8, test_size=0.2)
x2_train, x2_test, y2_train, y2_test = train_test_split(x, y2, train_size=0.8, test_size=0.2)
# Part 2.3.2
x1_train = vec.fit_transform(x1_train)
x1_test = vec.transform(x1_test)
dtc = DecisionTreeClassifier()
dtc.fit(x1_train, y1_train)
y1_pred = dtc.predict(x1_test)
print("For emotions : \n")
print(classification_report(y1_test, y1_pred))
print("Confusion Matrix: \n", confusion_matrix(y1_test, y1_pred))
x2_train = vec.fit_transform(x2_train)
x2_test = vec.transform(x2_test)
dtc1 = DecisionTreeClassifier()
dtc1.fit(x2_train, y2_train)
y2_pred = dtc1.predict(x2_test)
print("For sentiments : \n")
print(classification_report(y2_test, y2_pred))
print("Confusion Matrix: \n", confusion_matrix(y2_test, y2_pred))
f.close()
