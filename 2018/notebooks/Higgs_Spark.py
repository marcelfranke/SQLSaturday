# Databricks notebook source
dbutils.fs.ls("/mnt/data/higgs")

# COMMAND ----------

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #for plotting graphs
import seaborn as sns
sns.set(color_codes=True)
import pandas_profiling

# COMMAND ----------

# load data and convert to pandas dataframes
# test data is unlabeled as it is taken from https://archive.ics.uci.edu/ml/datasets/HIGGS
df = sqlContext.read.csv('/mnt/data/higgs/HIGGS.csv', inferSchema=True)

# COMMAND ----------

# rename the columns
column_names = ['label', 'lepton pT', 'lepton eta', 'lepton phi', 'missing energy magnitude', 'missing energy phi', 'jet 1 pt', 'jet 1 eta', 'jet 1 phi', 'jet 1 b-tag', 'jet 2 pt', 'jet 2 eta', 'jet 2 phi', 'jet 2 b-tag', 'jet 3 pt', 'jet 3 eta', 'jet 3 phi', 'jet 3 b-tag', 'jet 4 pt', 'jet 4 eta', 'jet 4 phi', 'jet 4 b-tag', 'm_jj', 'm_jjj', 'm_lv', 'm_jlv', 'm_bb', 'm_wbb', 'm_wwbb']

for i in range(0, len(df.columns)):
  df = df.withColumnRenamed(df.columns[i], column_names[i])

# COMMAND ----------

df.printSchema()

# COMMAND ----------

display(df.take(5))

# COMMAND ----------

# MAGIC %md ###Time to screen and clean data
# MAGIC Luckily, there is not much, we have to do here. Maybe, we should make the first column an integer label. 

# COMMAND ----------

from pyspark.sql.types import *
df = df.withColumn('label', df['label'].cast(IntegerType()))

# COMMAND ----------

display(df.describe())

# COMMAND ----------

# note the first column becomes the label, the remainder is the features

from pyspark.ml.linalg import DenseVector
# Define the `input_data` 
input_data = df.rdd.map(lambda x: (x[0], DenseVector(x[1:])))

df_input = spark.createDataFrame(input_data, ["label", "features"])
df_input

# COMMAND ----------

# MAGIC %md ## split data into training and test data

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

train_data, test_data = scaled_df.randomSplit([.8,.2], seed=7)

from pyspark.ml.classification import RandomForestClassifier, LogisticRegression

# Initialize `lr`
lr = LogisticRegression(labelCol="label",featuresCol='features_scaled', maxIter=10, regParam=0.3, elasticNetParam=0.8)
rfr = RandomForestClassifier(labelCol='label', featuresCol='features_scaled')

# Fit the data to the model
linearModel = lr.fit(train_data)
randomForestModel = rfr.fit(train_data)

# COMMAND ----------

#predict on test data
pred_log = linearModel.transform(test_data)
pred_rand = randomForestModel.transform(test_data)

# COMMAND ----------

#evaluate
import sklearn.metrics as metrics
# This is not a good idea for big data
acc_log = metrics.accuracy_score(pred_log.select('label').collect(), pred_log.select("prediction").collect())
acc_rand = metrics.accuracy_score(pred_rand.select('label').collect(), pred_rand.select("prediction").collect())

# COMMAND ----------

print(acc_log)

# COMMAND ----------

acc_rand

# COMMAND ----------

# MAGIC %md ### We could try to improve results via hyper parameter tuning now.
# MAGIC ### maybe next time...

# COMMAND ----------

