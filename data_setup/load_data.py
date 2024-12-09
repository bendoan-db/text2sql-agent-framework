# Databricks notebook source
import yaml
with open("../agent_config.yaml", "r") as file:
    agent_config = yaml.safe_load(file)

catalog = agent_config.get("databricks_resources")["catalog"]
schema = agent_config.get("databricks_resources")["schema"]

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")

# COMMAND ----------

import os
balance_sheet = (
    spark.read.format("csv")
    .option("header", "true")
    .option("inferSchema", "true")
    .load(
        "file://"+os.path.join(os.getcwd(), "data/balance_sheet/*.csv")
    )
)
cashflow = (
    spark.read.format("csv")
    .option("header", "true")
    .option("inferSchema", "true")
    .load(
        "file://"+os.path.join(os.getcwd(), "data/cashflow/*.csv")
    )
)

# COMMAND ----------

balance_sheet.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.balance_sheet")
cashflow.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.cashflow")
