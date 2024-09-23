# llm_connector.py
import openai
import time
from openai import RateLimitError
import asyncio

class LLMConnector:
    def __init__(self, api_key, model_name="codellama/CodeLlama-70b-Python-hf", base_url="https://api.aimlapi.com"):
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url
    
    async def connect(self, system_content, user_content):
        system_content = str(system_content)
        user_content = str(user_content)
        if not isinstance(system_content, str) or not isinstance(user_content, str):
            raise ValueError("System content and user content must be strings.")
        
        retries = 3
        for i in range(retries):
            try:
                client = openai.OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                )

                # Ensure that both system_content and user_content are strings and not empty
                if not system_content or not user_content:
                    raise ValueError("System content or User content cannot be empty.")

                messages = [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content},
                ]
        
                chat_completion = await asyncio.to_thread(
                    client.chat.completions.create,
                    model=self.model_name,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=128,
                )

                response = chat_completion.choices[0].message.content
                return response
            except openai.APIResponseValidationError as e:
                wait_time = 2 ** i  # Exponential backoff
                print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
            except openai.RateLimitError as e:
                print(f"An error occurred: {e}")
                raise  # Re-raise the exception if necessary
        raise Exception("Max retries exceeded.")

    async def generate(self, prompt: str) -> str:
        # This method is added for compatibility with the previous implementation
        # It uses the connect method internally
        return await self.connect("You are a helpful AI assistant.", prompt)