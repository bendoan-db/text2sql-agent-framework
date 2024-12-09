# Databricks notebook source
# MAGIC %pip install -q -r requirements.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import yaml
with open("agent_config.yaml", "r") as file:
    agent_config = yaml.safe_load(file)

llm_endpoint_name = agent_config.get("llm_config")["llm_endpoint_name"]
warehouse_id = agent_config.get("databricks_resources")["sql_warehouse_id"]

# COMMAND ----------

import os
from dbruntime.databricks_repl_context import get_context

TOKEN = get_context().apiToken

os.environ['DATABRICKS_TOKEN'] = get_context().apiToken
os.environ['DATABRICKS_HOST'] = "https://adb-984752964297111.11.azuredatabricks.net"

# COMMAND ----------

# MAGIC %run ./01_sql_react_agent

# COMMAND ----------

#test on single question
input_example = {
    "messages": [
        {
            "role": "user",
            "content": "In what year did Apple report its highest total assets?"
        }
    ]
}

response = react_chain.invoke(input_example)

# COMMAND ----------

response

# COMMAND ----------

import os
from mlflow.models.resources import (
    DatabricksServingEndpoint,
    DatabricksSQLWarehouse,
)

#point code to dev notebook
code_path = os.path.join(os.getcwd(), "01_sql_react_agent")

#point config to yaml file created in 01_sql_react_agent
config_path = os.path.join(os.getcwd(), "agent_config.yaml")

#point to requirements file
requirements_path = os.path.join(os.getcwd(), "requirements.txt")

#log example message
input_example = {
    "messages": [
        {
            "role": "user",
            "content": "In what year did Apple report its highest total assets? Include the year and the total assets value",
        }
    ]
}

# Log chain to mlflow using the langchain flavor
with mlflow.start_run(run_name="log_text2sql_react_agent"):
  logged_chain_info = mlflow.langchain.log_model(
    lc_model=code_path,
    model_config=config_path, # If you specify this parameter, this is the configuration that is used for training the model. The development_config is overwritten.
    artifact_path="chain", # This string is used as the path inside the MLflow model where artifacts are stored
    input_example=input_example, # Must be a valid input to your chain
    example_no_conversion=True, # Required
    resources = [
            DatabricksServingEndpoint(endpoint_name=llm_endpoint_name),
            DatabricksSQLWarehouse(warehouse_id=warehouse_id)
        ],
    pip_requirements=[f"-r {requirements_path}"]
  )

print(f"\n\nMLflow Run: {logged_chain_info.run_id}")
print(f"Model URI: {logged_chain_info.model_uri}\n\n")

# COMMAND ----------

# MAGIC %md
# MAGIC # Run Evaluation

# COMMAND ----------

data = [
    {
        "request_id":"1",
        "request": "In what year did Apple report its free cash flow?",
        "expected_response": "2022"
    },
    {
        "request_id":"2",
        "request": "What was American Express's solvency ratio in 2020?",
        "expected_response": "2.8%"
    },
    {
        "request_id":"3",
        "request": "Was AAPL's cash and cash equivalents increasing between 2019 and 2022?",
        "expected_response": "No it decreased every year from 2019 to 2022."
    },
    {
        "request_id":"4",
        "request": "Of the company's available, list the top 3 companies that had the highest cash and cash equivalents in 2020",
        "expected_response": "BAC, AAPL, AXP"
    }
]

eval_data = spark.createDataFrame(data).toPandas()

# COMMAND ----------

with mlflow.start_run(run_name="evaluate_text2sql_react_agent"):
    # Evaluate the logged model
    eval_results = mlflow.evaluate(
        data=eval_data,
        model=f'runs:/{logged_chain_info.run_id}/chain',
        model_type="databricks-agent",
    )

# COMMAND ----------

# MAGIC %md
# MAGIC # Register and Deploy

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

model_name = "react_text2sql_agent"
UC_MODEL_NAME = f"{catalog}.{db}.{model_name}"

# register the model to UC
uc_registered_model_info = mlflow.register_model(model_uri=logged_chain_info.model_uri, name=UC_MODEL_NAME)

# COMMAND ----------

from databricks import agents

agent_token = dbutils.secrets.get(scope="doan", key="agent-token")

# Deploy the model to the review app and a model serving endpoint
agents.deploy(UC_MODEL_NAME, uc_registered_model_info.version, environment_vars={"DATABRICKS_HOST": os.environ["DATABRICKS_HOST"], "DATABRICKS_TOKEN": agent_token})

# COMMAND ----------



# COMMAND ----------


