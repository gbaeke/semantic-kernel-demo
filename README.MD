# Azure OpenAI on your data with Semantic Kernel

Supply an .env file with the following variables:

```
AZURE_OPENAI_DEPLOYMENT_NAME="your_deployment_name"
AZURE_OPENAI_ENDPOINT="https://your_endpoint.openai.azure.com/"
AZURE_OPENAI_API_KEY="Your OpenAI API Key"
AZURE_OPENAI_API_VERSION="2023-12-01-preview" 

AZURE_AISEARCH_API_KEY="Your Azure Search API Key"
AZURE_AISEARCH_URL="https://your_endpoint.search.windows.net"
AZURE_AISEARCH_INDEX_NAME="your index name"
MODE="search" # any other value than search will be functions mode
BING_SEARCH_API_KEY="APIKEY"
```

Create a virtual environment with `python -m venv venv`. Activate the virtual environment with `source venv/bin/activate`.

Install the requirements with `pip install -r requirements.txt`.

Run the app with `streamlit run data.py`.

**IMPORTANT**: you need an Azure AI Search index with fields you can map to the fields the code expects. The code expects the following fields:
- titleField
- urlField
- contentFields: list
- vectorFields: list

In the code, you map your fields to these fields like so:

```python
azure_ai_search_settings["fieldsMapping"] = {
    "titleField": "Title",
    "urlField": "Url",
    "contentFields": ["Content"],
    "vectorFields": ["contentVector"], 
}
```

The vectorField contentVector contains a 1536-dimensional vector that represents the content of the document. You can use the OpenAI API to generate this vector. Use model `text-embedding-ada-002` to generate the vector.