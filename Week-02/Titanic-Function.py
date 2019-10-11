# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 12:08:36 2019

@author: alexl
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('Data/train.csv', index_col=0)
test = pd.read_csv('Data/test.csv', index_col=0)

#-------------------------- GLOBAL FEATURE VARIABLES --------------------------#

#calculate the average age of survivals vs drowned vs overall
mean_age = data['Age'].append(test['Age']).mean()
survived_average_age = data[data['Survived'] == 1]['Age'].mean()
drowned_average_age = data[data['Survived'] == 0]['Age'].mean()
mean_fare = data[data['Pclass'] == 3]['Fare'].append(test[test['Pclass'] == 3]['Fare']).mean()

#-------------------------- FEATURE ENGINEERING --------------------------#


def FeatureEngineer(df):
    """
    Takes in a dataframe, returning a feature engineered dataframe.
    """

    #replace "Age" column NAs with average age based on survived or not if known, else with mean.
    if 'Survived' in df.columns:
        df.loc[df['Survived'] == 1, 'Age'] = df.loc[df['Survived'] == 1, 'Age'].fillna(value = survived_average_age)
        df.loc[df['Survived'] == 0, 'Age'] = df.loc[df['Survived'] == 0, 'Age'].fillna(value = drowned_average_age)
    else:
        df['Age'].fillna(mean_age, inplace = True)

    #replace female with 0, male with 1
    if 'Sex' in df.columns:
        df['Sex'] = pd.factorize(df['Sex'], sort=True)[0]
    
    #fill "fare" column NAs
    if 'Fare' in df.columns and df.Fare.isna().sum() > 0:
        df['Fare'].fillna(mean_fare, inplace = True)
    
    #one-hot encode on Embarked data
    if 'Embarked' in df.columns:
        ohe_embarked = pd.get_dummies(df['Embarked'])
        #merge and drop useless columns
        df = pd.concat((df, ohe_embarked), axis=1)
        df.drop(['Embarked'], axis=1, inplace=True)
    
    #one-hot encode on Pclass data
    if 'Pclass' in df.columns:
        ohe_pclass = pd.get_dummies(df['Pclass'])
        #merge and drop useless columns
        df = pd.concat((df, ohe_pclass), axis=1)
        df.drop(['Pclass'], axis=1, inplace=True)
        
#    if 'SibSp' in df.columns:
#        df.groupby

    return df
#print(data.Pclass)
data = FeatureEngineer(data)
test = FeatureEngineer(test)

#siblings = []
#siblings['Survived'] = data[data['Survived'] == 1].groupby('SibSp')['SibSp'].count()
#siblings['Total'] = data.groupby('SibSp')['SibSp'].count()
#
#parch = []
#parch['Survived'] = data[data['Survived'] == 1].groupby('Parch')['Parch'].count()
#parch['Total'] = data.groupby('Parch')['Parch'].count()


#-------------------------- BUILDING THE MODEL --------------------------#

#select model

model = LogisticRegression(C = 0.5, random_state = 42, solver = 'lbfgs', max_iter = 1000)

#Select Data Fields
X = data.loc[:,['Age','Sex','SibSp', 1,3,'S']] # 'Pclass', 'Name' 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin' 'Q', 'S'
y = data.iloc[:,0].values
X_test = test.loc[:,X.columns]


#-------------------------- BUILDING THE MODEL --------------------------#

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=50)


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
    


#predict the test set results
def predict_results(X):
    y_predict = model.predict(X)
#    print(model.score(X_test,y_test))
#    X_df['Survived?'] = y_predict
    return y_predict

def export_predictions():
    #explorting data for kagale
    export = pd.DataFrame(test['Survived'], index = test.index)
    export.to_csv('Data/export.csv')



#-------------------------- EVALUATING CLASSIFIERS --------------------------#
from sklearn.metrics import confusion_matrix
#from sklearn.metrics import accuracy_score

def Prediction_Evaluator():
    c = confusion_matrix(y, y_predict)
    cm = pd.DataFrame(c, index = ['Actual-NO','Actual-YES'], columns = ['Pred-NO','Pred-Yes'])
    print(cm)
    from sklearn.metrics import precision_score, recall_score
    recall_score(y,y_predict), precision_score(y, y_predict)
    
    from sklearn.metrics import roc_curve, auc
    precision, recall, threshold = roc_curve(y,y_predict)
    
    plt.plot(precision, recall, marker = 'x')
    plt.xlabel('False-Pos Rate or Precision')
    plt.ylabel('False-Neg rate or Recall')
    plt.show()
    
    print(auc(precision, recall)) #area under curve
    

pred = build_model(model, X, y)
y_predict = predict_results(X)