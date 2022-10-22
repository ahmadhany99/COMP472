import json as js
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix
f = open('goemotions.json')
file = js.load(f)
split = np.array(file)
x = split[:, 0]
y1 = split[:, 1]
y2 = split[:, 2]
# Part 2.2
x1_train, x1_test, y1_train, y1_test = train_test_split(x, y1, train_size=0.8, test_size=0.2)
x2_train, x2_test, y2_train, y2_test = train_test_split(x, y2, train_size=0.8, test_size=0.2)
# Part 2.3.5
print("Using a better performing Decision Tree found using GridSearchCV. ")
vec=CountVectorizer()
x1_train = vec.fit_transform(x1_train)
x1_test = vec.transform(x1_test)
dtc = DecisionTreeClassifier()
param_dict={"criterion":["entropy"], "max_depth": [2,11],"min_samples_split": [12,40,6]}
grid=GridSearchCV(estimator=dtc,param_grid=param_dict,n_jobs=-1,error_score='raise')
grid.fit(x1_train, y1_train)
y1_pred = grid.predict(x1_test)
print("For emotions : \n")
print(classification_report(y1_test, y1_pred))
print("Confusion Matrix: \n", confusion_matrix(y1_test, y1_pred))
print("\n The best estimator across ALL searched params:\n", grid.best_estimator_)
print("\n The best score across ALL searched params:\n", grid.best_score_)
print("\n The best parameters across ALL searched params:\n", grid.best_params_)
x2_train = vec.fit_transform(x2_train)
x2_test = vec.transform(x2_test)
dtc1 = DecisionTreeClassifier()
grid1=GridSearchCV(estimator=dtc1,param_grid=param_dict,n_jobs=-1,error_score='raise')
grid1.fit(x2_train, y2_train)
y2_pred = grid1.predict(x2_test)
print("For sentiments : \n")
print(classification_report(y2_test, y2_pred))
print("Confusion Matrix: \n", confusion_matrix(y2_test, y2_pred))
print("\n The best estimator across ALL searched params:\n", grid1.best_estimator_)
print("\n The best score across ALL searched params:\n", grid1.best_score_)
print("\n The best parameters across ALL searched params:\n", grid1.best_params_)
f.close()
