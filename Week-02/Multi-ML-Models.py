# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 14:58:11 2019

@author: alexl
"""
import pandas as pd

# Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

data = pd.read_csv('Data/export_all.csv')
X = data.drop(labels='Survived',axis=1)
y = data['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)

names = ['LogisticRegression',
         'LinearGradientDecent',
          'RandomForest',
          'K-NNeighbor',
          'GaussianNB',
          'Perceptron',
          'LinearSVC',
          'DecisionTree',
          'SVC',
          'SGDClassifier'
        ]

classifiers = [LogisticRegression(),
               SGDClassifier(),
               RandomForestClassifier(),
               KNeighborsClassifier(),
               GaussianNB(),
               Perceptron(),
               LinearSVC(),
               DecisionTreeClassifier(),
               SVC(),
               SGDClassifier()
]
#print(LogisticRegression().get_params().keys())

parameters = [{'C':[0.1,0.5,0.8,1],'solver':['newton-cg', 'lbfgs'],'class_weight':['auto','balanced']},
               {'alpha':[0.0001,0.001,0.01],'learning_rate':['optimal']},
               {'class_weight':['balanced'],'n_estimators':[10,100,1000],'criterion':['gini','entropy']},
               {'n_neighbors':[3,5,7,9,12],'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute']},
               {},
               {},
               {},
               {'criterion':['gini','entropy'],'splitter':['best','random'],'class_weight':['balanced']},
               {},
               {}
               ]

answer = pd.DataFrame(columns=['Name','Score'])#,'TP','TN','FP','FN','AUC'])

for name, classifier, parameter in zip(names, classifiers, parameters):
    clf = GridSearchCV(classifier,parameter)
    
    m = clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    answer = answer.append({'Name':name, 'Score':score}, ignore_index=True)
    
print(answer.sort_values(by = 'Score', ascending = False))
