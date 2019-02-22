# Databricks notebook source
dbutils.fs.ls("/mnt/data/titanic")

# COMMAND ----------

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #for plotting graphs
import seaborn as sns
sns.set(color_codes=True)
import pandas_profiling

# COMMAND ----------

# load data and convert to pandas dataframes
#test_rdd = sqlContext.read.csv('/mnt/data/titanic/test.csv', header=True, inferSchema=True)
#test = test_rdd.toPandas()
# test data is unlabeled as it is taken from the kaggle challenge
train_rdd = sqlContext.read.csv('/mnt/data/titanic/train.csv', header=True, inferSchema=True)
train = train_rdd.toPandas()


# COMMAND ----------

# MAGIC %sql
# MAGIC -- mode "FAILFAST" will abort file parsing with a RuntimeException if any malformed lines are encountered
# MAGIC CREATE TEMPORARY TABLE temp_titanic
# MAGIC   USING csv
# MAGIC   OPTIONS (path "/mnt/data/titanic/train.csv", header "true", mode "FAILFAST")

# COMMAND ----------

# MAGIC %sql SELECT * FROM temp_titanic

# COMMAND ----------

report = pandas_profiling.ProfileReport(train)
displayHTML(report.to_html())

# COMMAND ----------

train.dtypes

# COMMAND ----------

pd.options.display.line_width = 200
train.head()

# COMMAND ----------

train.shape

# COMMAND ----------

# MAGIC %md ###Drop some of the columns, we cannot use. In particular, let's get rid of 
# MAGIC 1. Cabin as most 687 values of it are missing and the remainder does not seem significant
# MAGIC 2. PassengerId as it does not contribute to survival
# MAGIC 3. Name does not contribute to survival, except if it is Chuck Norris. However, we will use the naming column in order to guess some of the missing values in the Age column. So, we will drop Name, once we filled the missing values in the Age column.
# MAGIC 4. Ticket, because it does not contribute to survival

# COMMAND ----------

print("Hallo Marcel")

# COMMAND ----------

pd.isnull(train['Cabin']).sum()
# since cabin has lot of NaN Values and theres no way of filling those NaN so we gonna drop it

# COMMAND ----------

#passenger id does not contribute in model building
train.drop(['PassengerId','Cabin','Ticket'],axis=1,inplace=True) 

# COMMAND ----------

train.isnull().sum()

# COMMAND ----------

# MAGIC %md ##Handling Missing Values in Age column
# MAGIC Age can be a important factor in prediction since children and senior citizens are first to be saved!
# MAGIC The name column contains titles like Miss, Master, Mr. and Mrs. We use them to guess age values.
# MAGIC 
# MAGIC We guess Master and Miss corresponds to age 0-10, 
# MAGIC and 'Mr.' and 'Mrs.'  corresponds to age 20-60

# COMMAND ----------

import re
train_ini=[] # list for storing initials
for i in train['Name']:
    train_ini.append(re.findall(r"\w+\.",i))

# COMMAND ----------

train_null_age_index=list(np.where(train['Age'].isnull()))

# COMMAND ----------

# MAGIC %md ###define new age

# COMMAND ----------

import random
from random import randint

train_newages=[]
for i in range(len(train_null_age_index)):
    if 'Mr.' or 'Mrs.' in train_ini[train_null_age_index[i]]:
        train_newages.append(random.randint(20,35))
                             
    elif 'Master.' or 'Miss.' in train_ini[train_null_age_index[i]]:
        train_newages.append(random.randint(10,18))

# COMMAND ----------

#assigning values to missing indices
for i in range(len(train_newages)):
    train['Age'][train_null_age_index.pop()]=train_newages[i]

# COMMAND ----------

# MAGIC %md ## Ok, time to get rid of Name column

# COMMAND ----------

train.drop(['Name'],axis=1,inplace=True)

# COMMAND ----------

# MAGIC %md ## Ok, lets have another look at our remaining DataFrame

# COMMAND ----------

report = pandas_profiling.ProfileReport(train)
displayHTML(report.to_html())

# COMMAND ----------

# MAGIC %md ### let's see whether there is an impact on survival

# COMMAND ----------

train[['Parch','Survived']].groupby(['Parch']).mean().sort_values(by='Survived', ascending=False)

# COMMAND ----------

train[['SibSp','Survived']].groupby(['SibSp']).mean().sort_values(by='Survived', ascending=False)

# COMMAND ----------

train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)

# COMMAND ----------

g = sns.FacetGrid(train, col='Survived')
obj = g.map(plt.hist, 'Age', bins=20)
display(obj.fig)

# COMMAND ----------

g = sns.FacetGrid(train, col='Survived')
obj = g.map(plt.hist, 'Fare', bins=50)
display(obj.fig)

# COMMAND ----------

# MAGIC %md ## Now we can be sure that the above features are important

# COMMAND ----------

train.isnull().any()

# COMMAND ----------

# MAGIC %md ### ... we still have missing values in 'Embarked'
# MAGIC ### Since, there are only few missing, we can either drop them or replace them with "our guess". We do the latter.

# COMMAND ----------

train['Embarked'].mode()

# COMMAND ----------

train['Embarked'].fillna('S',inplace=True)

# COMMAND ----------

# MAGIC %md ### Fare cost does not seem to have an impact on survival, so we drop it.

# COMMAND ----------

train.drop(['Fare'],axis=1,inplace=True)

# COMMAND ----------

train.isnull().any()

# COMMAND ----------

# MAGIC %md ### Ok, the data is clean. Let's set up a classification model.
# MAGIC 
# MAGIC First of all, we have some string columns and we need to convert them into numerical (resp. categorical) values
# MAGIC 1. Sex is a string and it is categorical
# MAGIC 2. Embarked is a string and it is categorical
# MAGIC 3. Pclass is actually categorical, but also ordered. Thus, we can leave it being an integer value. 

# COMMAND ----------

X = train.drop('Survived', axis=1)
y = train['Survived']
X.head()

# COMMAND ----------

X.shape

# COMMAND ----------

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X = LabelEncoder()
#transform columns 1 and 5
X.iloc[:,1] = labelencoder_X.fit_transform(X.iloc[:,1])
X.iloc[:,5] = labelencoder_X.fit_transform(X.iloc[:,5])
X.head()

# COMMAND ----------

#onehot encode the embarked categories
#note sex category encoding is identical since its a binary category
onehotencoder = OneHotEncoder(categorical_features=[5], sparse=False)
X = onehotencoder.fit_transform(X)
X[:5]

# COMMAND ----------

# MAGIC %md ## split data into training and test data

# COMMAND ----------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# COMMAND ----------

# import typical classifiers
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

# COMMAND ----------

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
random_forest.score(X_test, y_test)

# COMMAND ----------

sgd = SGDClassifier()
sgd.fit(X_train, y_train)
sgd.score(X_test, y_test)

# COMMAND ----------

dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
dtc.score(X_test, y_test)

# COMMAND ----------

# MAGIC %md ### We could try to improve results via hyper parameter tuning now.
# MAGIC ### maybe next time...

# COMMAND ----------

