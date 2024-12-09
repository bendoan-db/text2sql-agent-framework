# Databricks notebook source
import yaml
import mlflow
from langchain_databricks import ChatDatabricks
from langchain.prompts import PromptTemplate, ChatPromptTemplate

agent_config = mlflow.models.ModelConfig(development_config="agent_config.yaml")

#initialize everything to local variables for cleaner code
catalog = agent_config.get("databricks_resources")["catalog"]
db = agent_config.get("databricks_resources")["schema"]
sql_warehouse_id = agent_config.get("databricks_resources")["sql_warehouse_id"]
databricks_host = agent_config.get("databricks_resources")["databricks_host"]

llm_endpoint_name = agent_config.get("llm_config")["llm_endpoint_name"]
llm_parameters = agent_config.get("llm_config")["llm_parameters"]

# Initialize the ChatDatabricks model
model = ChatDatabricks(
    endpoint=llm_endpoint_name,
    extra_params=llm_parameters,
)

system_message =agent_config.get("prompt")

# COMMAND ----------

def extract_question(input: dict):
    """
    Extracts the user question from the input messages, which should follow the openAI messages spec.

    Args:
    =========
      question (str): Dictionary with a list of user questions

    Returns:
    =========
      str: The user question, assumed to be the last message in the list
    """
    return input[-1]["content"]

# COMMAND ----------

import os
os.environ["OPEN_AI_API_KEY"] = os.environ.get("DATABRICKS_TOKEN")

# COMMAND ----------

import os
from operator import itemgetter

from langchain_core.runnables import RunnablePassthrough, RunnableBranch, RunnableLambda
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url=f"https://{databricks_host}/serving-endpoints",
    model=llm_endpoint_name,
    api_key=os.environ.get("DATABRICKS_TOKEN"),
    streaming=False,    
    temperature=0.1,
)

agent_executor_kwargs = {
    "handle_parsing_errors": True,  # Enables the agent to handle parsing errors and retry
}

#create SQL database instance for agent to access delta tables
agent_db = SQLDatabase.from_databricks(
    catalog=catalog,
    schema=db,
    host=databricks_host,
    warehouse_id=sql_warehouse_id,
)

#instantiate toolkit for agent to access database
toolkit = SQLDatabaseToolkit(db=agent_db, llm=model)  # initialize toolkit

react_agent = create_react_agent(
    llm, tools=toolkit.get_tools(), state_modifier=system_message
)

# COMMAND ----------

def get_final_message(response):
  try:
    final_response = response["messages"][-1].content
    return final_response
  except Exception as e:
    return f"Error parsing output: {e} \n\n Full output:" + str(response)

# COMMAND ----------

from mlflow.langchain.output_parsers import ChatCompletionsOutputParser

react_chain = (
  react_agent | RunnableLambda(get_final_message) | ChatCompletionsOutputParser()
)

# COMMAND ----------

#set the model to log in MLflow (see 02_register_agent)
mlflow.models.set_model(model=react_chain)

# COMMAND ----------


