import json as js
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
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
size = {"fontsize":0}
plt.pie(emotions.values(), labels=emotions.keys(),textprops=size)
plt.legend(loc='center left',bbox_to_anchor=(-0.4,0.5),fontsize=8)
plt.show()
plt.pie(sentiment.values(),labels=sentiment.keys())
plt.show()
a=np.concatenate(y)
vec=CountVectorizer()
vec.fit(a)
print (len(vec.vocabulary_))
train, test = train_test_split(y,train_size=0.8, shuffle=True, test_size=0.2)
f.close()
