import json as js
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

#Opening and loading the dataset
file = open('goemotions.json')
fLoaded = js.load(file)
#print(fLoaded)

#PART 2.1
posts = [item[0] for item in fLoaded]
vec = CountVectorizer()
vec.fit(posts)
print(len(vec.vocabulary_))

#PART 2.2
emotions = [item[1] for item in fLoaded]
sentiments = [item[2] for item in fLoaded]

X_train, X_test, yEmotion_train, yEmotion_test, ySent_train, ySent_test= train_test_split(posts, emotions, sentiments,train_size=0.80,
                                                                     shuffle=True, test_size=0.20)
#Part 2.3.1
X_train = vec.fit_transform(X_train)
X_test = vec.transform(X_test)

classifier = MultinomialNB()
model = classifier.fit(X_train, yEmotion_train)

y_pred = model.predict(X_test)
print("For emotions : \n")
print(classification_report(yEmotion_test, y_pred, zero_division= 1))
print("Confusion Matrix: \n", confusion_matrix(yEmotion_test, y_pred))

model = classifier.fit(X_train, ySent_train)
y_pred = model.predict(X_test)
print("For sentiments : \n")
print(classification_report(ySent_test, y_pred, zero_division= 1))
print("Confusion Matrix: \n", confusion_matrix(ySent_test, y_pred))
file.close()