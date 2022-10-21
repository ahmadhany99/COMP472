import json as js
import json as js
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

f = open('goemotions.json')
file = js.load(f)
split = np.array(file)
x = split[:, 0]
y1 = split[:, 1]
y2 = split[:, 2]

# Part 2.2
x1_train, x1_test, y1_train, y1_test = train_test_split(x, y1, train_size=0.8, test_size=0.2)

#PART 2.3.6 MLP
vec=CountVectorizer()


x1_train = vec.fit_transform(x1_train)
x1_test = vec.transform(x1_test)

mlp = MLPClassifier(max_iter=1)

parameters= {
    'hidden_layer_sizes': [(2,30, 50)],
    'activation': ['logistic', 'tanh', 'relu', 'identity'],
    'solver': ['adam', 'sgd'],
    }
clf = GridSearchCV(estimator = mlp, param_grid=parameters, n_jobs=-1,error_score='raise')
sc = StandardScaler(with_mean=False)
scaler = sc.fit(x1_train)
x1_train_scaled = scaler.transform(x1_train)


clf.fit(x1_train_scaled, y1_train)
y_pred= clf.predict(x1_test)
print('accuracy: {:.2f}'.format(accuracy_score(y1_test, y_pred)))

