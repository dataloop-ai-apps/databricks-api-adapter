# Databricks API Model Adapters

This repository contains the code for Dataloop model adapters that invoke models served in Databricks via their API. To use this model adapter, it is needed to have a Databricks workspace and a model serving endpoint in the workspace.

More information on how to create a Databricks workspace can be found [here](https://docs.databricks.com/en/admin/workspace/index.html).

More information on how to create a Model Serving Endpoint in the Databricks workspace can be found [here](https://docs.databricks.com/en/machine-learning/model-serving/index.html).

The models currently available are:
* **[DBRX-instruct](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm)**: A state-of-the-art LLM using MoE architecture to better provide precise answers across multiple fields of knowledge. The ```instruct``` version means it is fine-tuned and prepared to be served as a conversational model.