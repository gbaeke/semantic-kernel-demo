# Copyright (c) Microsoft. All rights reserved.

import asyncio
import os
from functools import reduce
from typing import TYPE_CHECKING

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.function_call_behavior import FunctionCallBehavior
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureChatPromptExecutionSettings
from semantic_kernel.contents import ChatHistory
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.function_call_content import FunctionCallContent
from semantic_kernel.contents.streaming_chat_message_content import StreamingChatMessageContent
from semantic_kernel.core_plugins import MathPlugin, TimePlugin
from semantic_kernel.functions import KernelArguments
from plugins import demo

if TYPE_CHECKING:
    from semantic_kernel.functions import KernelFunction
    
# add basic logging
import logging
logging.basicConfig(level=logging.INFO)
    
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
You are a chat bot.
"""

kernel = Kernel()

# Note: the underlying gpt-35/gpt-4 model version needs to be at least version 0613 to support tools.
kernel.add_service(AzureChatCompletion(
    service_id="chat",
    api_key=api_key,
    endpoint=endpoint,
    deployment_name=deployment_name,
    api_version=api_version,))

# kernel.add_plugin(MathPlugin(), plugin_name="math")
# kernel.add_plugin(TimePlugin(), plugin_name="time")
kernel.add_plugin(demo.Scrape(), plugin_name="scrape")

chat_function = kernel.add_function(
    prompt="{{$chat_history}}{{$user_input}}",
    plugin_name="ChatBot",
    function_name="Chat",
)
# enabling or disabling function calling is done by setting the function_call parameter for the completion.
# when the function_call parameter is set to "auto" the model will decide which function to use, if any.
# if you only want to use a specific function, set the name of that function in this parameter,
# the format for that is 'PluginName-FunctionName', (i.e. 'math-Add').
# if the model or api version does not support this you will get an error.

# Note: the number of responses for auto invoking tool calls is limited to 1.
# If configured to be greater than one, this value will be overridden to 1.
execution_settings = AzureChatPromptExecutionSettings(
    service_id="chat",
    max_tokens=2000,
    temperature=0.7,
    top_p=0.8,
    function_call_behavior=FunctionCallBehavior.EnableFunctions(
        auto_invoke=True, filters={"included_plugins": ["math", "time", "scrape"]}
    )
)

history = ChatHistory()

history.add_system_message(system_message)
history.add_user_message("Hi there, who are you?")
history.add_assistant_message("I am Mosscap, a chat bot. I'm trying to figure out what people need.")

arguments = KernelArguments(settings=execution_settings)


def print_tool_calls(message: ChatMessageContent) -> None:
    # A helper method to pretty print the tool calls from the message.
    # This is only triggered if auto invoke tool calls is disabled.
    items = message.items
    formatted_tool_calls = []
    for i, item in enumerate(items, start=1):
        if isinstance(item, FunctionCallContent):
            tool_call_id = item.id
            function_name = item.name
            function_arguments = item.arguments
            formatted_str = (
                f"tool_call {i} id: {tool_call_id}\n"
                f"tool_call {i} function name: {function_name}\n"
                f"tool_call {i} arguments: {function_arguments}"
            )
            formatted_tool_calls.append(formatted_str)
    print("Tool calls:\n" + "\n\n".join(formatted_tool_calls))


async def handle_streaming(
    kernel: Kernel,
    chat_function: "KernelFunction",
    arguments: KernelArguments,
) -> None:
    response = kernel.invoke_stream(
        chat_function,
        return_function_results=False,
        arguments=arguments,
    )

    print("Mosscap:> ", end="")
    streamed_chunks: list[StreamingChatMessageContent] = []
    async for message in response:
        if not execution_settings.function_call_behavior.auto_invoke_kernel_functions and isinstance(
            message[0], StreamingChatMessageContent
        ):
            streamed_chunks.append(message[0])
        else:
            print(str(message[0]), end="")

    if streamed_chunks:
        streaming_chat_message = reduce(lambda first, second: first + second, streamed_chunks)
        print("Auto tool calls is disabled, printing returned tool calls...")
        print_tool_calls(streaming_chat_message)

    print("\n")


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
    arguments["user_input"] = user_input
    arguments["chat_history"] = history

    stream = True
    if stream:
        await handle_streaming(kernel, chat_function, arguments=arguments)
    else:
        result = await kernel.invoke(chat_function, arguments=arguments)

        # If tools are used, and auto invoke tool calls is False, the response will be of type
        # ChatMessageContent with information about the tool calls, which need to be sent
        # back to the model to get the final response.
        function_calls = [item for item in result.value[-1].items if isinstance(item, FunctionCallContent)]
        if not execution_settings.function_call_behavior.auto_invoke_kernel_functions and len(function_calls) > 0:
            print_tool_calls(result.value[0])
            return True

        print(f"Mosscap:> {result}")
    return True


async def main() -> None:
    chatting = True
    print(
        "Welcome to the chat bot!\
        \n  Type 'exit' to exit.\
        \n  Try a math question to see the function calling in action (i.e. what is 3+3?)."
    )
    while chatting:
        chatting = await chat()


if __name__ == "__main__":
    asyncio.run(main())