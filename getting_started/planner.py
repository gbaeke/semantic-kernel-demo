# Copyright (c) Microsoft. All rights reserved.

import asyncio
import os

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.core_plugins import MathPlugin, TimePlugin
from semantic_kernel.planners import FunctionCallingStepwisePlanner, FunctionCallingStepwisePlannerOptions
from plugins import email
from promptflow.tracing import start_trace

start_trace(collection="func")

# logging
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

async def main():
    kernel = Kernel()

    service_id = "planner"
    kernel.add_service(
        AzureChatCompletion(
            service_id=service_id,
            api_key=api_key,
            endpoint=endpoint,
            deployment_name=deployment_name,
            api_version=api_version,
        ),
    )

    kernel.add_plugin(MathPlugin(), "MathPlugin")
    kernel.add_plugin(TimePlugin(), "TimePlugin")
    kernel.add_plugin(email.EmailPlugin(), "EmailPlugin")

    questions = [
        "What is the current hour number, plus 5?",
    ]

    options = FunctionCallingStepwisePlannerOptions(
        max_iterations=10,
        max_tokens=4000,
    )

    planner = FunctionCallingStepwisePlanner(service_id=service_id, options=options)

    for question in questions:
        result = await planner.invoke(kernel, question)
        print(f"Q: {question}\nA: {result.final_answer}\n")

        # Uncomment the following line to view the planner's process for completing the request
        print(f"Chat history: {result.chat_history}\n")


if __name__ == "__main__":
    asyncio.run(main())