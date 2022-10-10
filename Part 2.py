import json as js
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
f = open('goemotions.json')
y = js.load(f)
emotions = {}
sentiment={}
for i in y:
    if i[1] not in emotions:
        emotions[i[1]] = 1
    else:
        emotions[i[1]] += 1
    if i[2] not in sentiment:
        sentiment[i[2]] = 1
    else:
        sentiment[i[2]] += 1
a=np.concatenate(y)
vec=CountVectorizer()
vec.fit(a)
print (len(vec.vocabulary_))
train, test = train_test_split(y,train_size=0.8, shuffle=True, test_size=0.2)