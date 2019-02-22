# Databricks notebook source
import numpy as np
import pandas as pd
import zipfile
pd.options.display.width = 200

# COMMAND ----------

rdd = sqlContext.read.csv('/mnt/data/titanic/train.csv', header=True, inferSchema=True)


# COMMAND ----------

rdd.show(5)

# COMMAND ----------

df = rdd.select(['Survived','Pclass','Age','SibSp','Parch','Fare'])

# COMMAND ----------

from pyspark.sql.types import *

# Write a custom function to convert the data type of DataFrame columns
def convertColumn(df, names, newType):
  for name in names: 
     df = df.withColumn(name, df[name].cast(newType))
  return df 

# COMMAND ----------

# MAGIC %md ### Ok, our features and labels are there. It is time to set up our machine learning pipeline!
# MAGIC ### The spark ML frameworks asks you to provide a label column and a feature column. Note, that the feature column can consist of vectors/multiple values. 

# COMMAND ----------

df

# COMMAND ----------

from pyspark.ml.linalg import DenseVector
# Define the `input_data` 
input_data = df.rdd.map(lambda x: (x[0], DenseVector(x[1:])))

df_input = spark.createDataFrame(input_data, ["label", "features"])
df_input

# COMMAND ----------

# MAGIC %md ### As always or at least very often, scale your values!

# COMMAND ----------

from pyspark.ml.feature import MinMaxScaler
# Initialize the `standardScaler`
scaler = MinMaxScaler(inputCol="features", outputCol="features_scaled")
# Fit the DataFrame to the scaler
scaler = scaler.fit(df_input)

# COMMAND ----------

# Transform the data in `df` with the scaler
scaled_df = scaler.transform(df_input)
scaled_df.first()

# COMMAND ----------

# MAGIC %md ### Setup your ML algorithm! We use Linear Regression and a Random Forest Regressor here.

# COMMAND ----------

train_data, test_data = scaled_df.randomSplit([.8,.2], seed=7)
# Import `LinearRegression`
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression

# Initialize `lr`
lr = LogisticRegression(labelCol="label",featuresCol='features_scaled', maxIter=10, regParam=0.3, elasticNetParam=0.8)


# Fit the data to the model
linearModel = lr.fit(train_data)

# COMMAND ----------

rfr = RandomForestClassifier(labelCol='label', featuresCol='features_scaled')
randomForestModel = rfr.fit(train_data)

# COMMAND ----------

# MAGIC %md ### Predict on test data.

# COMMAND ----------

pred_log = linearModel.transform(test_data)
pred_rand = randomForestModel.transform(test_data)

# COMMAND ----------

pred_log.show(2)

# COMMAND ----------

pred_rand.show(2)

# COMMAND ----------

# MAGIC %md ### Ok, let's compare the predictions in terms of "mean absolute error".

# COMMAND ----------

import sklearn.metrics as metrics
# This is not a good idea for big data
acc_log = metrics.accuracy_score(pred_log.select('label').collect(), pred_log.select("prediction").collect())
acc_rand = metrics.accuracy_score(pred_rand.select('label').collect(), pred_rand.select("prediction").collect())

# COMMAND ----------

acc_rand

# COMMAND ----------

acc_log

# COMMAND ----------

