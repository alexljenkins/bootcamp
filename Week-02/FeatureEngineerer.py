# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 12:08:36 2019

@author: alexl
"""
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def FeatureEngineer(df):
    """
    Takes in a dataframe, returning a feature engineered dataframe.
    """

    #replace female with 0, male with 1
    if 'Sex' in df.columns:
        df['Sex'] = pd.factorize(df['Sex'], sort=True)[0]
    
    if 'Cabin' in df.columns:
        df['Cabin'].fillna(0, inplace = True)
        df[df['Cabin'] != 0] = 1

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
    """
    Encodes a column into multiple binary columns.
    Requires DF and column name
    """
    ohencoder = pd.get_dummies(df[column])
    #merge and drop useless columns
    df = pd.concat((df, ohencoder), axis=1)
    df.drop([column], axis=1, inplace=True)

    return df

def FeatureScaler(df):
    """
    Scales features to values between 0-1
    Requires DF
    """
    scaler = MinMaxScaler()
    for column in df.columns:
        try:
            df[column] = scaler.fit_transform(df[[column]])
        except:
            continue
            
    return df
    

print('I am the best')






















