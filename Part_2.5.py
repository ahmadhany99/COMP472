import json as js
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

#Part 2.5
f = open('goemotions.json')
file = js.load(f)
split = np.array(file)
x = split[:, 0]
y1 = split[:, 1]
y2 = split[:, 2]

x1_train, x1_test, y1_train, y1_test = train_test_split(x, y1, train_size=0.6, test_size=0.4)
x2_train, x2_test, y2_train, y2_test = train_test_split(x, y2, train_size=0.6, test_size=0.4)

#Part 2.3.1
print("Using Naive Bayes Model with default parameters")
vec = CountVectorizer()
x1_train = vec.fit_transform(x1_train)
x1_test = vec.transform(x1_test)
x2_train = vec.fit_transform(x2_train)
x2_test = vec.transform(x2_test)
classifier = MultinomialNB()

model = classifier.fit(x1_train, y1_train)
y_pred = model.predict(x1_test)
print("For emotions : \n")
print(classification_report(y1_test, y_pred))
print("Confusion Matrix: \n", confusion_matrix(y1_test, y_pred))

model = classifier.fit(x2_train, y2_train)
y_pred = model.predict(x2_test)
print("For sentiments : \n")
print(classification_report(y2_test, y_pred))
print("Confusion Matrix: \n", confusion_matrix(y2_test, y_pred))

# Part 2.3.2
print("Using DTC model with default parameters")
dtc = DecisionTreeClassifier()
dtc.fit(x1_train, y1_train)
y1_pred = dtc.predict(x1_test)
print("For emotions : \n")
print(classification_report(y1_test, y1_pred))
print("Confusion Matrix: \n", confusion_matrix(y1_test, y1_pred))
dtc1 = DecisionTreeClassifier()
dtc1.fit(x2_train, y2_train)
y2_pred = dtc1.predict(x2_test)
print("For sentiments : \n")
print(classification_report(y2_test, y2_pred))
print("Confusion Matrix: \n", confusion_matrix(y2_test, y2_pred))

#Part 2.3.3
print("Using MLP model with default parameters")
mlp_clf = MLPClassifier(max_iter=1)
mlp_clf.fit(x1_train, y1_train)
y1_pred = mlp_clf.predict(x1_test)
print("For emotions : \n")
print(classification_report(y1_test, y1_pred))
print("Confusion Matrix: \n", confusion_matrix(y1_test, y1_pred))
mlp2 = MLPClassifier(max_iter=1)
mlp2.fit(x2_train, y2_train)
y2_pred = mlp2.predict(x2_test)
print("For sentiments : \n")
print(classification_report(y2_test, y2_pred))
print("Confusion Matrix: \n", confusion_matrix(y2_test, y2_pred))

#Part 2.3.4
print("Using a better performing Naive Bayes CLassifier using GridSearchCV. ")
nb_classifier = MultinomialNB()
param_dict={"alpha":[0.5, 0, 1, 10.0]}
grid1 = GridSearchCV(estimator=nb_classifier,param_grid=param_dict,n_jobs=-1,error_score='raise')
grid1.fit(x1_train, y1_train)
y1_pred = grid1.predict(x1_test)
print("For emotions : \n")
print(classification_report(y1_test, y1_pred))
print("Confusion Matrix: \n", confusion_matrix(y1_test, y1_pred))

grid2=GridSearchCV(estimator=nb_classifier,param_grid=param_dict,n_jobs=-1,error_score='raise')
grid2.fit(x2_train, y2_train)
y2_pred = grid2.predict(x2_test)
print("For sentiments : \n")
print(classification_report(y2_test, y2_pred))
print("Confusion Matrix: \n", confusion_matrix(y2_test, y2_pred))

#Part 2.3.5
print("Using a better performing Decision Tree found using GridSearchCV. ")
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

#Part 2.3.6
print("Using a better performing MLP found using GridSearchCV. ")
mlp = MLPClassifier(early_stopping=True, verbose=True, max_iter=1)
sc = StandardScaler(with_mean=False)
scaler_x = sc.fit(x1_train)
x1_train_scaled = scaler_x.transform(x1_train)
scaler_y = sc.fit(y1_train)
y1_train_scaled = scaler_y.transform(y1_train)

parameters= {
    'hidden_layer_sizes': ((5,5), (5,10)),
    'activation': ['logistic', 'tanh', 'relu', 'identity'],
    'solver': ['adam', 'sgd'],
    }
clf = GridSearchCV(estimator = mlp, param_grid=parameters, n_jobs=-1,error_score='raise')

clf.fit(x1_train_scaled, y1_train)
y1_pred= clf.predict(x1_test)
print('accuracy: {:.2f}'.format(accuracy_score(y1_test, y1_pred)))
print("For emotions : \n")
print(classification_report(y1_test, y1_pred,zero_division=1))
print("Confusion Matrix: \n", confusion_matrix(y1_test, y1_pred))
x2_train = vec.fit_transform(x2_train)
x2_test = vec.transform(x2_test)
y2_train = vec.transform(y2_train)
y2_test = vec.transform(y2_test)
scaler_x2 = sc.fit(x2_train)
x2_train_scaled = scaler_x2.transform(x2_train)
scaler_y2 = sc.fit(y2_train)
y2_train_scaled = scaler_y2.transform(y2_train)

mlp1 = MLPClassifier(max_iter=1)
grid1=GridSearchCV(estimator=mlp1,param_grid=parameters,n_jobs=-1,error_score='raise')
grid1.fit(x2_train_scaled, y2_train)
y2_pred = grid1.predict(x2_test)
print("For sentiments : \n")
print(classification_report(y2_test, y2_pred, zero_division=1))
print("Confusion Matrix: \n", confusion_matrix(y2_test, y2_pred))

f.close()