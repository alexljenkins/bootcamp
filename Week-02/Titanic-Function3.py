# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 12:08:36 2019

@author: alexl
"""
import pandas as pd
import matplotlib.pyplot as plt
import FeatureEngineerer as fe

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_score, recall_score

data = pd.read_csv('Data/train.csv', index_col=0)
test = pd.read_csv('Data/test.csv', index_col=0)

df_all = pd.concat([data,test])

def share_results(a,b):
    iterater = 0
    for i in a[0]:
        print(f'coefficient of {X.columns[iterater]}: {round(i,4)}')
        iterater +=1
    print(f'constant is: {b[0]}')
    

def build_model(model, X, y):
    #Fit model
    model.fit(X, y)
    print(f'Model score was: {model.score(X,y)}')
    share_results(model.coef_, model.intercept_)
    
    return model.predict(X)
    

def predict_results(X):
    #predict the test set results
    y_predict = model.predict(X)
#    print(model.score(X_test,y_test))
#    X_df['Survived?'] = y_predict
    return y_predict


def export_predictions(df, y_predict, model_name):
    #explorting data for kagale
    export = pd.DataFrame(y_predict, columns = ['Survived'], index = test.index) ##################
    export.to_csv(f'Data/export_{model_name}.csv')
    
#-------------------------- EVALUATING CLASSIFIERS --------------------------#


def Prediction_Evaluator(y, y_predict, model_name):
    print(f'-------------- Logistic Regression --------------')
    c = confusion_matrix(y, y_predict)
    cm = pd.DataFrame(c, index = ['Actual-NO','Actual-YES'], columns = ['Pred-NO','Pred-Yes'])
    print(cm)
    
    recall_score(y,y_predict), precision_score(y, y_predict)

    precision, recall, threshold = roc_curve(y, y_predict)
    
    plt.plot(precision, recall, marker = 'x')
    plt.xlabel('False-Pos Rate or Precision')
    plt.ylabel('False-Neg rate or Recall')
    plt.show()
    
    print(auc(precision, recall)) #area under curve

#clean dataframes
data = fe.FeatureEngineer(data)
test = fe.FeatureEngineer(test)

#select model and set hyperparameters
model = LogisticRegression(C = 1, random_state = 42, max_iter = 10000, solver='lbfgs') #solver='lbfgs' actually less accurate
#model = RandomForestClassifier(n_estimators = 100 ,criterion='gini')
#select model features
X = data.loc[:,['Individual_fare', 'Mr','Mrs','Rare','Age_group', 'Fam', 'C', 3]]

""" VARIABLES
'Sex'
'Age_group', 'Age'
'Master', 'Miss', 'Mr', 'Mrs', 'Rare'
'Fam', 'SibSp', 'Parch'
'Pclass', 'Q', 'S', 'C'
'Name', 'Ticket', 'Fare', 'Cabin', 'Individual_fare'
"""

y = data.iloc[:,0].values
X_test = test.loc[:,X.columns] #X_test if not train-test-split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42) #comment out

#scale data
X = fe.FeatureScaler(X)
X_test = fe.FeatureScaler(X_test)

#model_name, model = ModelSelector()
pred = build_model(model, X, y)
y_predict = predict_results(X) #make X_test
#Prediction_Evaluator(y, y_predict, model)

#Produce evaluations or export results depending if data was split or not
#if y.shape == y_predict.shape:
#    Prediction_Evaluator(y, y_predict, model)
#else:
#    export_predictions(X_test, y_predict, model) #PLAY WITH C VALUE

#data.to_csv(f'Data/export.csv')

#applying FE and scaling to the complete dataset
df_all = fe.FeatureEngineer(df_all)
df_all = fe.FeatureScaler(df_all)
df_all.to_csv('Data/export_alla.csv')
