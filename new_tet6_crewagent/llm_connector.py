# llm_connector.py
import random
import openai
from abc import ABC, abstractmethod
import asyncio

class LLMProvider(ABC):
    @abstractmethod
    async def generate(self, system_content, user_content, **kwargs):
        pass

class OpenAIProvider(LLMProvider):
    def __init__(self, api_key, base_url="https://api.openai.com", model="gpt-3.5-turbo"):
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model

    async def generate(self, system_content, user_content, **kwargs):
        try:
            chat_completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content},
                ],
                temperature=kwargs.get('temperature', 0.7),
                max_tokens=kwargs.get('max_tokens', 128),
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            print(f"Error in OpenAI generation: {e}")
            return None


class MistralProvider(LLMProvider):
    def __init__(self, api_key, base_url="https://api.aimlapi.com", model="mistralai/Mistral-7B-Instruct-v0.2"):
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model

    async def generate(self, system_content, user_content, **kwargs):
        max_retries = 5
        for attempt in range(max_retries):
            try:
                chat_completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=kwargs.get('temperature', 0.7),
                    max_tokens=kwargs.get('max_tokens', 128),
                )
                return chat_completion.choices[0].message.content
            except openai.RateLimitError as e:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + random.random()  # Exponential backoff with jitter
                    print(f"Rate limit exceeded. Retrying in {wait_time:.2f} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"Max retries reached. Error: {e}")
                    return "I apologize, but I'm currently unavailable due to high demand. Please try again later."
            except Exception as e:
                print(f"Error in Mistral generation: {e}")
                return "I encountered an error while processing your request. Please try again."

class LLMConnector:
    def __init__(self, provider: LLMProvider):
        self.provider = provider

    async def generate(self, system_content, user_content, **kwargs):
        return await self.provider.generate(system_content, user_content, **kwargs)

# Factory function to create LLM providers
def create_llm_provider(provider_name, api_key, **kwargs):
    if provider_name.lower() == 'openai':
        return OpenAIProvider(api_key, **kwargs)
    elif provider_name.lower() == 'mistral':
        return MistralProvider(api_key, **kwargs)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider_name}")