# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 13:19:15 2019
@author: alexl
"""
#TSNE for smushing things into two demensions so we can visualise it

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data = pd.read_csv('Data/train.csv', index_col=0)
test = pd.read_csv('Data/test.csv', index_col=0)
print(data.columns)

#Exploring who survived

#shows the percentage of people who survived
print(data['Survived'].sum() /
      data['Survived'].count())

#shows bar graphs of didn't survive (0) vs survived (1) by Pclass
sns.countplot(data['Survived'], hue = data['Pclass'])
plt.show()

#shows the percentage of upper class passengers who survived
print(data[data['Pclass'] == 1]['Survived'].sum() /
      data[data['Pclass'] == 1]['Survived'].count())

#shows men and women by class
sns.countplot(data['Sex'], hue = data['Pclass'])
plt.show()

#men vs women survival rate
men = data.loc[data['Sex'] == 'male']["Survived"]
rate_men = round(sum(men)/len(men)*100,2)

women = data.loc[data.Sex == 'female']["Survived"]
rate_women = round(sum(women)/len(women)*100,2)

print(f"{rate_women}% of women survived")
print(f"{rate_men}% of men survived")

#visualise age distribution
sns.distplot(data['Age'].dropna(),kde=False,bins=30)
plt.show()
#visualise age in which class
sns.boxplot(x='Pclass',y='Age',data=data)
plt.show()
#visualise if age effects survival
plt.scatter(x='Survived',y='Age',data=data)
plt.show()

#calculate the average age of survivals vs drowned vs overall
mean_age = data['Age'].append(test['Age']).mean()
survived_average_age = data[data['Survived'] == 1]['Age'].mean()
drowned_average_age = data[data['Survived'] == 0]['Age'].mean()
print(survived_average_age)
print(drowned_average_age)
#null values heatmap
sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='coolwarm')
plt.show()

one_hot_encoded = pd.get_dummies(data['Embarked'])
print(one_hot_encoded)
pd.concat(data, one_hot_encoded, axis = 1)

#or in one line
data[['S','Q','C']] = pd.get_dummies(data['Embarked'])
### Original way to replace age na data:
#data['Age'].fillna(200, inplace = True)
#
#for i in data.index:
#    if data.loc[i,'Age'] == 200 and data.loc[i,'Survived'] == 1:
#        data.loc[i,'Age'] = survived_average_age
#    if data.loc[i,'Age'] == 200 and data.loc[i,'Survived'] == 0:
#        data.loc[i,'Age'] = drowned_average_age
###

##replace NA with average age based on survived or not
data.loc[data['Survived'] == 1, 'Age'] = data.loc[data['Survived'] == 1, 'Age'].fillna(value = survived_average_age)
data.loc[data['Survived'] == 0, 'Age'] = data.loc[data['Survived'] == 0, 'Age'].fillna(value = drowned_average_age)
test['Age'].fillna(mean_age, inplace = True)

#show null heatmap again to ensure all NAs are gone
sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='coolwarm')
plt.show()


#covert male/female into boolean
data['Sex'] = data['Sex'].map({'male':0,'female':1})

pd.factorize(data['Sex'], sort = True)[0]
#train data
data['Sex'].replace(to_replace = 'male', value = '1',inplace = True)
data['Sex'].replace(to_replace = 'female', value = '0',inplace = True)
#test data
test['Sex'].replace(to_replace = 'male', value = '1',inplace = True)
test['Sex'].replace(to_replace = 'female', value = '0',inplace = True)

#replace NAs in test data "fare" column with mean of fares for that class
mean_fare = data[data['Pclass'] == 3]['Fare'].append(test[test['Pclass'] == 3]['Fare']).mean()
test['Fare'].fillna(mean_fare, inplace = True)


#visualise correlation of the data
sns.heatmap(data.corr())
plt.show()

#null values heatmap
sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='coolwarm')
plt.show()

#Correlation and heatmap
sns.heatmap(data.corr(), cmap ='Greys', annot=True)
plt.figure(figsize=(10,10))
plt.show()



#--------------------------BUILDING THE MODEL--------------------------#

#Grab the relevant data fields
X = data.loc[:,['Sex','SibSp','Pclass']]
y = data.iloc[:,0].values
X_test = test.loc[:,['Sex','SibSp','Pclass']]

#Fit logistic regression to training dataset
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(random_state = 0)
log_reg.fit(X, y)

#predict the test set results
y_predict = log_reg.predict(X_test)
test['Survived'] = y_predict

#explorting data for kagale
export = pd.DataFrame(test['Survived'], index = test.index)
export.to_csv('Data/export.csv')
print(data['Name'])
print(log_reg.coef_)
print(log_reg.intercept_)
pred = log_reg.predict(X)
print(log_reg.score(X,y))
