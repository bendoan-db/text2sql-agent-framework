#Overview 

This archive is a self-contained quickstart for text2sql ReAct agents on Databricks. It provides Text2SQL question-answering functionality on financial datasets found in the `data_setup` directory

# Files
* `agent_config.yaml`: Contains key parameters for the ReAct Agent, including: 
  * Model inference params: model endpoint name, temperature, max_tokens, etc.
  * SQL warehouse params: warehouse id
  * Databricks resource params: hostname, catalog, schema
* `01_sql_react_agent` contains code for the text2sql ReAct agent, implemented using LangChain
* `02_evaluate`: Tests/runs the code in `01_sql_react_agent`. It contains code that:
  * [Logs the chain to MLflow](https://docs.databricks.com/en/generative-ai/agent-framework/log-agent.html)
  * Implements  [mlflow.evaluate()](https://docs.databricks.com/en/generative-ai/agent-evaluation/index.html) to evaluate agent against a benchmark dataset
  * Registers the chain to Unity Catalog
  * Deploys the chain to a [serving endpoint](https://docs.databricks.com/en/generative-ai/agent-framework/deploy-agent.html) and starts a [review UI](https://docs.databricks.com/en/generative-ai/agent-evaluation/human-evaluation.html)
* `data_setup`: Contains data (csv) files for tables and notebook to create Delta tables for testing

# Setup
1. Update `agent_config.yaml` with the Databricks Resources (warehouse, hostname, catalog, schema), model resources (model endpoints, temperature, max tokens, etc.), and base prompt
2. Run `data_setup` to create data tables for testing
3. Review and customize `01_sql_react_agent` as needed
4. Test the agent code using `02_evaluate`. Once the code is stabilized, register the model, run evaluation, register and deploy.

# Requirements
* Permissions to create schemas and tables in Databricks
* Permissions to deploy model serving endpoints
* Enablement of [AI-assisted features](https://docs.databricks.com/en/notebooks/use-databricks-assistant.html#enable-or-disable-admin-features) on your Databricks workspace