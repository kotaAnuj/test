import openai
api_key = "1723871e17b1432c9644a5b0d0e1574c"

class QueryTranslationAgent:
    def __init__(self, api_key):
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.aimlapi.com",
        )

    def translate_query(self, query):
        system_content = "You are an AI assistant."
        response = self.client.chat.completions.create(
            model="mistralai/Mistral-7B-Instruct-v0.2",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": query}
            ],
            temperature=0.7,
            max_tokens=128
        )
        return response.choices[0].message.content
