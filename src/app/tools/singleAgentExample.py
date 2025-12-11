import os
from openai import AzureOpenAI
from dotenv import load_dotenv
import time

load_dotenv()

# Retrieve credentials from .env file or environment
endpoint = os.getenv("gpt_endpoint")
deployment = os.getenv("gpt_deployment")
api_key = os.getenv("gpt_api_key")
api_version = os.getenv("gpt_api_version")

# Initialize Azure OpenAI client using Responses API (2025-04-01-preview)
client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=api_key,
    api_version=api_version,
)

def generate_response(text_input):
    start_time = time.time()
    """
    Input:
        text_input (str): The user's chat input.

    Output:
        response (str): A Markdown-formatted response from the agent.
    """

    # Prepare input for Responses API with system and user messages


    instruction_Parts = ["You are a helpful assistant working for Zava, a company that specializes in offering ",
        "products to assist homeowners with do-it-yourself projects. Respond to customer inquiries ",
        "with relevant product recommendations and DIY tips. If a customer asks for paint, suggest ",
        "one of the following three colors: blue, green, and white. If a customer asks for something ",
        "not related to a DIY project, politely inform them that you can only assist with DIY-related ",
        "inquiries. Zava has a variety of store locations across the country. If a customer asks about ",
        "store availability, direct the customer to the Miami store."]

    instruction_Input = " ".join(instruction_Parts)

    # Call Azure OpenAI Responses API (preview)
    completion = client.responses.create(
        model=deployment,
        instructions=instruction_Input,
        input=text_input,
        max_output_tokens=10000,
        top_p=1,
    )
    end_sum = time.time()
    print(f"generate_response Execution Time: {end_sum - start_time} seconds")
    # Return response text (Responses API)
    return completion.output_text
