import streamlit as st
import time
import semantic_kernel as sk
import asyncio
import semantic_kernel.connectors.ai.open_ai as sk_oai
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.utils.settings import azure_openai_settings_from_dot_env_as_dict
import json

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
    messages = [{"role": message["role"], "content": message["content"]} for message in data["messages"] if message["role"] != "system"]

    return messages

async def main():
    
    
    # initialize semantic kernel
    if "kernel" not in st.session_state:
        st.session_state.kernel = sk.Kernel()
        # Azure OpenAI services
        service_id = "gpt4turbo"
        chat_service = sk_oai.AzureChatCompletion(
            service_id=service_id, **azure_openai_settings_from_dot_env_as_dict(include_api_version=True)
        )
        st.session_state.kernel.add_service(chat_service)
        # configure request settings
        req_settings = st.session_state.kernel.get_prompt_execution_settings_from_service_id(service_id)
        req_settings.max_tokens = 1000
        req_settings.temperature = 0.7

        # set system message
        system_message = """
        You are a chat bot. You help the user by answering questions.
        You always think step-by-step and provide clear explanations.
        """
        
        # create the chat function
        if "chat_function" not in st.session_state:
            st.session_state.chat_function = st.session_state.kernel.create_function_from_prompt(
                plugin_name="chat",
                function_name="chat",
                prompt="{{$chat_history}}{{$user_input}}",
                prompt_execution_settings=req_settings
            )
    

    # initialize chat history
    if "history" not in st.session_state:
        st.session_state.history = ChatHistory(system_message=system_message)

    st.title('Semantic Kernel Bot')

        
    for message in await get_messages(st.session_state.history):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            
    if prompt := st.chat_input("Type something..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.history.add_user_message(prompt)
        
        # Fake response
        with st.spinner("Thinking..."):
            chat_history = st.session_state.history
            response = await st.session_state.kernel.invoke(st.session_state.chat_function, KernelArguments(user_input=prompt, chat_history=chat_history))

            
        with st.chat_message("assistant"):
            st.markdown(response)
            
        # Add bot message to chat history
        st.session_state.history.add_assistant_message(str(response))
        
if __name__ == '__main__':
    asyncio.run(main())