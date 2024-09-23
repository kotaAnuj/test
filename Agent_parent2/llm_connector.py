import openai
import json
import logging

# Ensure this is done before you create an instance of AIAgentCreator
api_key = "2d2d8893395640a89cf366271e1a84fe"


class LLMConnector:
    def __init__(self, api_key: str, model_name: str, base_url: str):
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url

        # Initialize the client with the provided API key and base URL
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)

    async def connect(self, messages: list) -> str:
        try:
            logging.info(f"Sending request to LLM with model: {self.model_name}")

            # Send the request to the LLM API
            chat_completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.7,
                max_tokens=128
            )
            
            # Access the content directly as an attribute, not a dictionary key
            response = chat_completion.choices[0].message.content
            return response

        except Exception as e:
            logging.error(f"An unexpected error occurred: {str(e)}")
            return "An unexpected error occurred while processing your request."

