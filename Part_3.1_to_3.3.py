import json as js
import numpy as np
import gensim.downloader as api
from nltk import word_tokenize


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
