
import asyncio
import logging

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.function_call_behavior import FunctionCallBehavior
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents import ChatHistory

logging.basicConfig(level=logging.WARNING)

# import environment variables
from dotenv import load_dotenv
import os
load_dotenv("../.env")

# Get environment variables

deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")

system_message = """
You are a chat bot. Your name is Mosscap and
you have one goal: figure out what people need.
Your full name, should you need to know it, is
Splendid Speckled Mosscap. You communicate
effectively, but you tend to answer with long
flowery prose.
"""

kernel = Kernel()

service_id = "chat-gpt"
chat_service = AzureChatCompletion(
    service_id=service_id,
    api_key=api_key,
    endpoint=endpoint,
    deployment_name=deployment_name,
    api_version=api_version,
    
)
kernel.add_service(chat_service)

req_settings = kernel.get_prompt_execution_settings_from_service_id(service_id=service_id)
req_settings.max_tokens = 2000
req_settings.temperature = 0.7
req_settings.top_p = 0.8
req_settings.function_call_behavior = FunctionCallBehavior.EnableFunctions(
    auto_invoke=True, filters={"excluded_plugins": []}
)
## The third method is the most specific as the returned request settings class is the one that is registered for the service and has some fields already filled in, like the service_id and ai_model_id. # noqa: E501 E266


chat_function = kernel.add_function(
    prompt=system_message + """{{$chat_history}}{{$user_input}}""",
    function_name="chat",
    plugin_name="chat",
    prompt_execution_settings=req_settings,
)

history = ChatHistory()
history.add_user_message("Hi there, who are you?")
history.add_assistant_message("I am Mosscap, a chat bot. I'm trying to figure out what people need.")


async def chat() -> bool:
    try:
        user_input = input("User:> ")
    except KeyboardInterrupt:
        print("\n\nExiting chat...")
        return False
    except EOFError:
        print("\n\nExiting chat...")
        return False

    if user_input == "exit":
        print("\n\nExiting chat...")
        return False

    stream = True
    if stream:
        answer = kernel.invoke_stream(
            chat_function,
            user_input=user_input,
            chat_history=history,
        )
        print("Mosscap:> ", end="")
        async for message in answer:
            print(str(message[0]), end="")
        print("\n")
        return True
    answer = await kernel.invoke(
        chat_function,
        user_input=user_input,
        chat_history=history,
    )
    print(f"Mosscap:> {answer}")
    history.add_user_message(user_input)
    history.add_assistant_message(str(answer))
    return True


async def main() -> None:
    chatting = True
    while chatting:
        chatting = await chat()


if __name__ == "__main__":
    asyncio.run(main())
