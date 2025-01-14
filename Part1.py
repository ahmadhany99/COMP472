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
size = {"fontsize":12}
plt.pie(emotions.values(), labels=emotions.keys(),textprops=size, autopct='%1.0f%%')
plt.legend(loc='center left',bbox_to_anchor=(-0.4,0.5),fontsize=8)
plt.show()
plt.pie(sentiment.values(),labels=sentiment.keys(), autopct='%1.2f%%')
plt.show()
f.close()
