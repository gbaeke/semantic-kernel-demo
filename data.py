import streamlit as st
import semantic_kernel as sk
import asyncio
import os
import semantic_kernel.connectors.ai.open_ai as sk_oai
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.utils.settings import azure_openai_settings_from_dot_env
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureAISearchDataSources,
    AzureChatPromptExecutionSettings,
    AzureDataSources,
    ExtraBody,
)
from semantic_kernel.core_plugins.time_plugin import TimePlugin
from semantic_kernel.connectors.ai.open_ai.utils import (
    get_tool_call_object,
)
from semantic_kernel.prompt_template.input_variable import InputVariable
from semantic_kernel.prompt_template.prompt_template_config import PromptTemplateConfig
import logging
from semantic_kernel.utils.logging import setup_logging
import json
import re
from semantic_kernel.connectors.search_engine import BingConnector
from semantic_kernel.core_plugins import WebSearchEnginePlugin


async def extract_urls(response):
    """
    Extracts URLs from the given response.

    Args:
        response (dict): The full JSON response from a semantic function containing the URLs.

    Returns:
        list: A list of unique URLs extracted from the response.
    """
    # extract URLs
    tool_content_str = None
    for message in response["value"][0]["inner_content"]["choices"][0]["message"]["context"]["messages"]:
        if message["role"] == "tool":
            tool_content_str = message["content"]
            break

    # Parse the JSON string inside the 'content' to get the citations
    if tool_content_str:
        tool_content = json.loads(tool_content_str)
        citations = tool_content.get("citations", [])
        
        # Extract URLs from the citations
        urls = [citation['url'] for citation in citations if 'url' in citation]
        # Convert the list of URLs to a dictionary with keys as "doc{i+1}" and values as the URLs
        result = {f"doc{i+1}": url for i, url in enumerate(urls)}

        # make urls unique
        return result

async def get_messages(chat_history):
    """
    Extracts the role and content fields from each message in the chat history.

    Args:
        chat_history (ChatHistory): The chat history object.

    Returns:
        list: A list of dictionaries containing the role and content fields of each message.
    """
    data = json.loads(chat_history.model_dump_json())

    # Extract the "role" and "content" fields from each message
    # Skip system messages
    messages = [{"role": message["role"], "content": message["content"]} for message in data["messages"] if message["role"] != "system"]

    return messages

def replace_with_links_brackets(text, url_dict):
    # Find all placeholders in the text
    placeholders = set(re.findall(r'\[\w+\]', text))
    
    # Replace each placeholder with its markdown link, keeping [] around the link
    for placeholder in placeholders:
        doc_key = placeholder.strip('[]')
        if doc_key in url_dict:
            # Adjusting the format to keep [] around the markdown link
            markdown_link = f"<a href='{url_dict[doc_key]}' target='_blank'>[{doc_key}]</a>"
            text = text.replace(placeholder, markdown_link)
    
    return text

async def main():
    setup_logging()
    # set DEBUG level
    logging.getLogger("semantic_kernel").setLevel(logging.DEBUG)
    
    # set system message
    system_message = """
    You are a chat bot. You help the user by answering questions.
    You always think step-by-step and provide clear explanations.
    Your name is Chatron.
    """
    
    # read the environment variables
    from dotenv import load_dotenv
    load_dotenv()

    # restart streamlit app if mode is changed    
    mode = os.getenv("MODE", "search")

    
    # search_data = True --> Use Azure OpenAI on your data
    # search_data = False --> Enable function calling and add a time plugin for demo purposes
    search_data = True if mode == "search" else False
    
    # configure sidebar
    with st.sidebar:
        st.title('🤖 SK Chat with Data')
        if st.button("Reset"):
            st.session_state.clear()
            st.rerun()
        if search_data:
            st.write("🔍 Using Azure OpenAI on your data")
        else:
            st.write("🕰️ Using function calling and time plugin")
    
    # initialize semantic kernel and store in session state
    if "kernel" not in st.session_state:
        st.session_state.kernel = sk.Kernel()

        # Azure AI Search integration
        azure_ai_search_settings = sk.azure_aisearch_settings_from_dot_env_as_dict()
        azure_ai_search_settings["fieldsMapping"] = {
            "titleField": "Title",
            "urlField": "Url",
            "contentFields": ["Content"],
            "vectorFields": ["contentVector"], 
        }
        azure_ai_search_settings["embeddingDependency"] = {
            "type": "DeploymentName",
            "deploymentName": "embedding"  # you need an embedding model with this deployment name is same region as AOAI
        }
        az_source = AzureAISearchDataSources(**azure_ai_search_settings, queryType="vectorSimpleHybrid", system_message=system_message) # set to simple for text only and vector for vector
        az_data = AzureDataSources(type="AzureCognitiveSearch", parameters=az_source)
        extra = ExtraBody(dataSources=[az_data]) if search_data else None
        
        
        # Azure OpenAI services
        service_id = "gpt"
        deployment, api_key, endpoint = azure_openai_settings_from_dot_env(include_api_version=False)
        chat_service = sk_oai.AzureChatCompletion(
            service_id=service_id,
            deployment_name=deployment,
            api_key=api_key,
            endpoint=endpoint,
            api_version="2023-12-01-preview" if search_data else "2024-02-01",  # azure openai on your data in SK only supports 2023-12-01-preview
            use_extensions=True if search_data else False # extensions are required for data search
        )
        st.session_state.kernel.add_service(chat_service)
        req_settings = AzureChatPromptExecutionSettings(
            service_id=service_id,
            extra_body=extra,
            tool_choice="none" if search_data else "auto", # no tool calling for data search
            temperature=0,
            max_tokens=1000)
        
        # add plugins
        if not search_data:
            # time plugin
            st.session_state.kernel.import_plugin_from_object(TimePlugin(), plugin_name="time")
            
            # bing plugin (uses Azure Bing Search.v7 resource kind; not Bing.CustomSearch)
            connector = BingConnector(api_key=os.getenv("BING_SEARCH_API_KEY"))
            st.session_state.kernel.import_plugin_from_object(WebSearchEnginePlugin(connector), "WebSearch")     
        
        # prompt template
        prompt_template_config = PromptTemplateConfig(
            template="{{$chat_history}}{{$user_input}}",
            name="chat",
            template_format="semantic-kernel",
            input_variables=[
                InputVariable(name="chat_history", description="The history of the conversation", is_required=True),
                InputVariable(name="user_input", description="The user input", is_required=True),
            ],
        )

        # create the chat function
        if "chat_function" not in st.session_state:
            st.session_state.chat_function = st.session_state.kernel.create_function_from_prompt(
                plugin_name="chat",
                function_name="chat",
                prompt_template_config=prompt_template_config,
            )
            
        # modify request settings to exclude chat for tool calling
        if not search_data:
            filter = {"exclude_plugin": ["chat"]}
            req_settings.tools = get_tool_call_object(st.session_state.kernel, filter)
            req_settings.auto_invoke_kernel_functions = True
        
        if "arguments" not in st.session_state:
            st.session_state.arguments = KernelArguments(settings=req_settings)
        
    

    # initialize chat history
    if "history" not in st.session_state:
        st.session_state.history = ChatHistory(system_message=system_message)        
        

    # In Streamlit, always show the chat history first
    for message in await get_messages(st.session_state.history):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    # Capture user input and invoke chat function      
    if prompt := st.chat_input("Type something..."):
        # Add user message to UI
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Add user message to chat history
        st.session_state.history.add_user_message(prompt)
        
        # Invoke chat function and get response and urls
        with st.spinner("Thinking..."):
            arguments = st.session_state.arguments
            arguments["chat_history"] = st.session_state.history
            arguments["user_input"] = prompt
            response = await st.session_state.kernel.invoke(st.session_state.chat_function, arguments=arguments)
            data = json.loads(response.model_dump_json())
            try:
                urls = await extract_urls(data)
            except:
                urls = []
                
                    
        # Add bot response to UI but remove the [docx] references
        with st.chat_message("assistant"):
            response = replace_with_links_brackets(str(response), urls)
            st.markdown(response, unsafe_allow_html=True)
             
        # print urls to sidebar
        if urls:
            with st.sidebar:
                with st.expander("🔗 Sources", False):    
                    for key, value in urls.items():
                        st.markdown(f'<a href="{value}" target="_blank">{key}</a>', unsafe_allow_html=True)
                st.markdown("**Note:** some sources may be duplicates")        
                
        # Add bot message to chat history (sources are not included)
        st.session_state.history.add_assistant_message(str(response))
        
if __name__ == '__main__':
    asyncio.run(main())