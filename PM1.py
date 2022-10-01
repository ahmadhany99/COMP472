import json as js
import numpy as np
import matplotlib.pyplot as plt
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
f.close()
