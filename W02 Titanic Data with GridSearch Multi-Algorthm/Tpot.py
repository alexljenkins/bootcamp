# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 18:57:30 2019
@author: alexl
"""
import pandas as pd

from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split

data = pd.read_csv('Data/export_all.csv')

X = data.drop(labels='Survived',axis=1)
y = data['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

pipeline_optimizer = TPOTClassifier(generations=2, population_size=50, cv=5,
                                    random_state=42, verbosity=2)
pipeline_optimizer.fit(X_train, y_train)

print(pipeline_optimizer.score(X_test, y_test))
pipeline_optimizer.export('tpot_exported_pipeline_X.py')
