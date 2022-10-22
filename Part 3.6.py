import json as js
import numpy as np
import gensim.downloader as api
import nltk 
from nltk import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
#Part 3.1

GoogleWord2Vec = api.load('word2vec-google-news-300')
f = open('goemotions.json')
file = js.load(f)
split = np.array(file)
posts = split[:, 0]
y1 = split[:, 1]
y2 = split[:, 2]

# Part 3.2

postsEmbed = [word_tokenize(post) for post in posts]

# Part 3.3

AveragePostEmbed = []
NumberOfWordsFound = 0
NumberOfWordsNotFound = 0
for i in range(len(postsEmbed)):
    NumberOfWordsHit = 0
    EmbedOfWords = 0.0
    for word in postsEmbed[i]:
        if word in GoogleWord2Vec:
            EmbedOfWords += GoogleWord2Vec[word]
            NumberOfWordsFound += 1
            NumberOfWordsHit += 1
        else:
            NumberOfWordsNotFound += 1
    if NumberOfWordsHit == 0:
        AveragePostEmbed.append(np.zeros(len(GoogleWord2Vec['hello'])))
    else:
        AveragePostEmbed.append(EmbedOfWords / NumberOfWordsHit)

x1_train, x1_test, y1_train, y1_test = train_test_split(AveragePostEmbed, y1, train_size=0.8, test_size=0.2)
x2_train, x2_test, y2_train, y2_test = train_test_split(AveragePostEmbed, y2, train_size=0.8, test_size=0.2)

#Part 3.6

print("Using a better performing Multi-Layered Perceptron")
mlp = MLPClassifier(max_iter=1, activation="identity", solver="adam")
mlp.fit(x1_train, y1_train)
y1_pred = mlp.predict(x1_test)
print("For emotions : \n")
print(classification_report(y1_test, y1_pred))
mlp1 = MLPClassifier(max_iter=1, activation="identity", solver="adam")
mlp1.fit(x2_train, y2_train)
y1_pred = mlp1.predict(x2_test)
print("For sentiments : \n")
print(classification_report(y2_test, y1_pred))