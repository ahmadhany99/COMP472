import json as js
import numpy as np
from sklearn.model_selection import train_test_split

f = open('goemotions.json')
file = js.load(f)
split = np.array(file)
x = split[:, 0]
y1 = split[:, 1]
y2 = split[:, 2]

# Part 2.2
x1_train, x1_test, y1_train, y1_test = train_test_split(x, y1, train_size=0.8, test_size=0.2)
x2_train, x2_test, y2_train, y2_test = train_test_split(x, y2, train_size=0.8, test_size=0.2)
