# Databricks notebook source
display(dbutils.fs.ls('/mnt/data/higgs/'))

# COMMAND ----------

rdd = sqlContext.read.csv('/mnt/data/higgs/HIGGS.csv', header=True, inferSchema=True)

# COMMAND ----------

df = rdd.toPandas()

# COMMAND ----------

df.head()

# COMMAND ----------

