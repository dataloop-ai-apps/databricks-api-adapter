# Databricks API Model Adapters

This repository contains the code for Dataloop model adapters that invoke models served in Databricks via their API. To use this model adapter, it is needed to have a Databricks workspace and a model serving endpoint in the workspace.

More information on how to create a Databricks workspace can be found [here](https://docs.databricks.com/en/admin/workspace/index.html).

More information on how to create a Model Serving Endpoint in the Databricks workspace can be found [here](https://docs.databricks.com/en/machine-learning/model-serving/index.html).

The models currently available are:
* Chat Completion:
    * **[DBRX-instruct](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm)**: A state-of-the-art LLM using MoE architecture to better provide precise answers across multiple fields of knowledge. The ```instruct``` version means it is fine-tuned and prepared to be served as a conversational model.
* Text Embeddings:
    * **[BGE-Small-EN](https://marketplace.databricks.com/details/c0521a85-2bac-4983-a4ff-0ce6d4aa1944/Databricks_BGE-v15-Models)**: A lightweight English text embedding model by BAAI, designed for high-quality embeddings in resource-limited environments.

## Configuration:

Once you install this model adapter in a project in the Dataloop platform, visit its model page to finish its configuration. There you will need to change the two fields marked with the red rectangles here:

![](assets/dbrx-instruct-config.png)

these fields should be filled with information taken from your Databricks serving endpoint:

* ```model_name```: this should be the name of the model you chose during the endpoint creation.
* ```base_url```: The base url as appears in your workspace. It should follow a format like: "https://adb-xxxxxxxxx.y.azuredatabricks.net/serving-endpoints" where the ```x```s represent your 9-digit workspace number and ```y``` is a randomized number. This url uses Azure as a cloud provider, which could be changed for GCP or AWS.

Lastly, it will be required to add your Databricks access token as a secret in the service. Visit [this page](https://docs.databricks.com/en/dev-tools/auth/pat.html) to obtain more information on how to generate your Databricks Access Token. And [here](https://docs.dataloop.ai/docs/secrets-overview?highlight=secrets) you can find information on how to securely add your token as a secret in the Dataloop platform.

Once the model is [deployed](https://docs.dataloop.ai/docs/model-deployment?highlight=deploy) in your project, go to its service page, and select to edit its settings:
![](assets/edit-service-settings.png)

First, add the secret you created to the service by selecting the following option:

![](assets/edit-secret.png)

and then selecting the secret from the dropdown menu. After that, edit its ```init_inputs``` by selecting the following option:

![](assets/edit-init-inputs.png)

The ```databricks_api_key_name``` should be the string you used as a key for the secret containing your Databricks Token:

![](assets/edit-init-inputs-values.png)

you can keep the ```model_entity``` value. Save the configurations and the service should be ready to be used for predictions in the Dataloop platform.
