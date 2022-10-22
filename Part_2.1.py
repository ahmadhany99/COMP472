import json as js
from sklearn.feature_extraction.text import CountVectorizer

#Part1
f = open('goemotions.json')
file = js.load(f)
posts = [item[0] for item in file]
vec = CountVectorizer()
vec.fit(posts)
print(len(vec.vocabulary_))
f.close()
