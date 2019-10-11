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
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
#from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_score, recall_score


# Algorithms
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB


data = pd.read_csv('Data/train.csv', index_col=0)
test = pd.read_csv('Data/test.csv', index_col=0)

#-------------------------- GLOBAL FEATURE VARIABLES --------------------------#

#calculate the average age of survivals vs drowned vs overall
#mean_age = data['Age'].append(test['Age']).mean()
#survived_average_age = data[data['Survived'] == 1]['Age'].mean()
#drowned_average_age = data[data['Survived'] == 0]['Age'].mean()
#mean_fare = data[data['Pclass'] == 3]['Fare'].append(test[test['Pclass'] == 3]['Fare']).mean()

#-------------------------- FEATURE ENGINEERING --------------------------#


def FeatureEngineer(df):
    """
    Takes in a dataframe, returning a feature engineered dataframe.
    """

    #replace "Age" column NAs with average age based on survived or not if known, else with mean.
#    if 'Survived' in df.columns:
#        df.loc[df['Survived'] == 1, 'Age'] = df.loc[df['Survived'] == 1, 'Age'].fillna(value = survived_average_age)
#        df.loc[df['Survived'] == 0, 'Age'] = df.loc[df['Survived'] == 0, 'Age'].fillna(value = drowned_average_age)
#    else:
#        df['Age'].fillna(mean_age, inplace = True)
#    df['Age'].fillna(mean_age, inplace = True)
    
    
    #replace female with 0, male with 1
    if 'Sex' in df.columns:
        df['Sex'] = pd.factorize(df['Sex'], sort=True)[0]
    
    if 'Cabin' in df.columns:
        df['Cabin'].fillna(0, inplace = True)
        df[df['Cabin'] != 0] = 1
        
#    if 'SibSp' and 'Survived' in df.columns:
#        df['SibSp'] = LinearEncoder(df, 'SibSp')
#        
#    if 'Parch' and 'Survived' in df.columns:
#        df['Parch'] = LinearEncoder(df, 'Parch')

    if 'Parch' and 'SibSp' in df.columns:
        #determine if passenger is alone or not
        df['Fam'] = df['SibSp'] + df['Parch']
        #creates Fam as boolean
#        df['Fam'].apply(lambda x: 1 if x > 0 else 0)
        
    #fill "fare" column NAs and calc fare per person
    if 'Fare' in df.columns and df.Fare.isna().sum() > 0:
        mean_fare = df[df['Pclass'] == 3]['Fare'].mean()
        df['Fare'].fillna(mean_fare, inplace = True)
        
    df['Individual_fare'] = df['Fare']/(df['Fam']+1)
    df['Individual_fare'] = df['Individual_fare'].astype(int)
    df['Individual_fare'] = pd.qcut(df['Individual_fare'], 4, labels = [0, 1, 2, 3])
    df['Individual_fare'].fillna(value = 3, inplace = True)

    if 'Name' in df.columns:
        #Extract title and one-hot encode
        for name in df:
            df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    
        for title in df['Title']:
            df['Title'].replace(['Lady', 'Countess','Capt', 'Col',
         	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare', inplace = True)
        
            df['Title'].replace('Mlle', 'Miss', inplace = True)
            df['Title'].replace('Ms', 'Miss', inplace = True)
            df['Title'].replace('Mme', 'Mrs', inplace = True)
        
    if 'Age' in df.columns:
        #calculates mean age for titles and applys to missing age values
        titles = df['Title'].unique()
        
        for title in titles:
            mean_age = df[df['Title'] == title]['Age'].mean()
            df.loc[df['Title'] == title, 'Age'] = df.loc[df['Title'] == title, 'Age'].fillna(value = mean_age)
        
        #group age into 3 categories 'Child', 'Adult', 'Senior' as ints
        bins = [-100, 11, 18, 22, 27, 33, 40, 100]
        labels = [0, 1, 2, 3, 4, 5, 6]
        df['Age_group'] = pd.cut(df["Age"], bins, labels = labels)

        #draw a bar plot of age vs. survival
        if 'Survived' in df.columns:
            sns.barplot(x="Age_group", y="Survived", data=df)
            plt.show()
            
    if 'Title' in df.columns:
        #one-hot encode on title data
        df = OneHotEncoder(df, 'Title')
        
    if 'Pclass' in df.columns:
        #one-hot encode on Pclass data
        df = OneHotEncoder(df, 'Pclass')
        
    if 'Embarked' in df.columns:
        #one-hot encode on Embarked data
        df = OneHotEncoder(df, 'Embarked')
        
    return df


def OneHotEncoder(df, column):
    ohencoder = pd.get_dummies(df[column])
    #merge and drop useless columns
    df = pd.concat((df, ohencoder), axis=1)
    df.drop([column], axis=1, inplace=True)

    return df


#def LinearEncoder(df, column):
#    """
#    Currently not being used
#    """
#    
#    encoder = pd.DataFrame()
#    encoder['Total'] = df.groupby(column)[column].count()
#    encoder['Survived'] = df[df['Survived'] == 1].groupby(column)[column].count()
#    encoder['Survived'] = encoder['Survived'].fillna(0)
#    encoder['Percentage'] = encoder['Survived'] / encoder['Total']
#    encoder.sort_values('Percentage', axis=0, ascending=False, inplace=True)
#
#    mapped = df[column].map(encoder['Percentage'])
#    return mapped
#
#LinearEncoder(data,'SibSp')

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

def FeatureScaler(df):
    #Feature scaling = normalizing for 0-1
    scaler = MinMaxScaler()
    for column in df.columns:
        df[column] = scaler.fit_transform(df[[column]])
    return df
    
#-------------------------- EVALUATING CLASSIFIERS --------------------------#


def Prediction_Evaluator(y, y_predict, model_name):
    print(f'--------------{model_name}--------------')
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

def ModelSelector(i):
    
    model_names = ['Logistic Regression','Linear Gradient Decent','Random Forest','K-N Neighbor','GaussianNB','Perceptron','LinearSVC','DecisionTree','SGDClassifier','LinearSVC']
    model_codes = [LogisticRegression(C = .5, random_state = 42, max_iter = 10000, solver='lbfgs'),
                   linear_model.SGDClassifier(max_iter=5, tol=None),
                   RandomForestClassifier(n_estimators=100),
                   KNeighborsClassifier(n_neighbors = 3),
                   GaussianNB(),
                   Perceptron(max_iter=5),
                   LinearSVC(),
                   DecisionTreeClassifier(),
                   SGDClassifier(),
                   LinearSVC()
    ]
    models = pd.DataFrame(model_names, model_codes)
#    models = {'Logistic Regression': LogisticRegression(C = .5, random_state = 42, max_iter = 10000, solver='lbfgs'),
#             'Linear Gradient Decent': linear_model.SGDClassifier(max_iter=5, tol=None),
#             'Random Forest': RandomForestClassifier(n_estimators=100),
#             'K-N Neighbor': KNeighborsClassifier(n_neighbors = 3),
#             'GaussianNB': GaussianNB(),
#             'Perceptron': Perceptron(max_iter=5),
#             'LinearSVC': LinearSVC(),
#             'DecisionTree': DecisionTreeClassifier()
#             }
    print(models.iloc[0,i], models.iloc[1,i])
    return models.iloc[0,i], models.iloc[1,i]

#clean dataframes
data = FeatureEngineer(data)
test = FeatureEngineer(test)

#select model and set hyperparameters
model = LogisticRegression(C = 1, random_state = 42, max_iter = 10000, solver='lbfgs') #solver='lbfgs' actually less accurate

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
X = FeatureScaler(X)
X_test = FeatureScaler(X_test)

#model_name, model = ModelSelector()
pred = build_model(model, X, y)
y_predict = predict_results(X)
Prediction_Evaluator(y, y_predict, model)
#for i in range(8):
#    model_name, model = ModelSelector(i)
#    pred = build_model(model, X, y)
#    y_predict = predict_results(X) #make X_test
#    #Produce evaluations or export results depending if data was split or not
#    if y.shape == y_predict.shape:
#        Prediction_Evaluator(y, y_predict, model_name)
#    else:
#        export_predictions(X_test, y_predict, model_name) #PLAY WITH C VALUE

#data.to_csv(f'Data/exported_data.csv')