# Databricks notebook source
import numpy as np
import pandas as pd
import zipfile
import sklearn.datasets
pd.options.display.width = 200

# COMMAND ----------

# MAGIC %md ### download housing data to a "local" temporary mount location
# MAGIC Local means it is on your hard drive! Btw. it is still packed.

# COMMAND ----------

data = sklearn.datasets.california_housing
data = data.fetch_california_housing('/mnt/house')

# COMMAND ----------

# MAGIC %md ### Combine features and labels into one dataframe and write it to disk

# COMMAND ----------

# It still has to be moved to the databricks file system (dbfs)
data_X_and_y = np.hstack((data.data, data.target.reshape(-1,1)))
features = data.feature_names
features.append('Price')
df = pd.DataFrame(data_X_and_y, columns=features)
df.to_csv('/mnt/house/house.csv', index=False)
df.head()

# COMMAND ----------

# MAGIC %md ### copy/move temporary files to our shared mount location

# COMMAND ----------

dbutils.fs.cp("file:///mnt/house/house.csv", "/mnt/house/house.csv")
dbutils.fs.cp("file:///mnt/house/house_names.txt", "/mnt/house/house_names.txt")

# COMMAND ----------

dbutils.fs.ls('/mnt/house/')

# COMMAND ----------

# MAGIC %md ### we can see: the files are there now!
# MAGIC 
# MAGIC So, read it into a pyspark dataframe

# COMMAND ----------

rdd = sqlContext.read.csv('/mnt/house/house.csv', header=True, inferSchema=True )

# COMMAND ----------

rdd.show(5)

# COMMAND ----------

df = rdd.select("Price", "AveBedrms", "AveOccup", "AveRooms", "HouseAge", "Population")
df.show(5)

# COMMAND ----------

# MAGIC %md ### although, this is obsolete here, it is time to make sure everything is a float value 
# MAGIC (sometimes a value may look like a float but is a string instead)

# COMMAND ----------

from pyspark.sql.types import *

# Write a custom function to convert the data type of DataFrame columns
def convertColumn(df, names, newType):
  for name in names: 
     df = df.withColumn(name, df[name].cast(newType))
  return df 

# COMMAND ----------

df_converted = convertColumn(df, df.columns, FloatType())

# COMMAND ----------

# MAGIC %md Prices are given as multiples of 10.000$, we need to multiply here first

# COMMAND ----------

from pyspark.sql.functions import *
df_converted = df_converted.withColumn('Price', col('Price') * 10000)

# COMMAND ----------

df_converted.describe().show()

# COMMAND ----------

# MAGIC %md ### Ok, our features and labels are there. It is time to set up our machine learning pipeline!
# MAGIC ### The spark ML frameworks asks you to provide a label column and a feature column. Note, that the feature column can consist of vectors/multiple values. 

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
from pyspark.ml.regression import RandomForestRegressor, LinearRegression

# Initialize `lr`
lr = LinearRegression(labelCol="label",featuresCol='features_scaled', maxIter=10, regParam=0.3, elasticNetParam=0.8)


# Fit the data to the model
linearModel = lr.fit(train_data)

# COMMAND ----------

rfr = RandomForestRegressor(labelCol='label', featuresCol='features_scaled')
randomForestModel = rfr.fit(train_data)

# COMMAND ----------

# MAGIC %md ### Predict on test data.

# COMMAND ----------

pred_lin = linearModel.transform(test_data)
pred_rand = randomForestModel.transform(test_data)

# COMMAND ----------

pred_lin.show(2)

# COMMAND ----------

pred_rand.show(2)

# COMMAND ----------

# MAGIC %md ### Ok, let's compare the predictions in terms of "mean absolute error".

# COMMAND ----------

import sklearn.metrics as metrics
# This is not a good idea for big data
error_lin = metrics.mean_absolute_error(pred_lin.select('label').collect(), pred_lin.select("prediction").collect())
error_rand = metrics.mean_absolute_error(pred_rand.select('label').collect(), pred_rand.select("prediction").collect())

# COMMAND ----------

pred_lin = pred_lin.withColumn('error', abs(col('label') - col('prediction')))
pred_rand = pred_rand.withColumn('error', abs(col('label') - col('prediction')))

# COMMAND ----------

print('Linear Regression')
pred_lin.select(mean(pred_lin.error).alias('mae')).show()
print('Random Forest Regression')
pred_rand.select(mean(pred_rand.error).alias('mae')).show()

# COMMAND ----------

# MAGIC %md ### We have a winner!

# COMMAND ----------

