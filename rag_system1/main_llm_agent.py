import openai
api_key = "1723871e17b1432c9644a5b0d0e1574c"
class MainLLMAgent:
    def __init__(self, api_key):
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.aimlapi.com",
        )

    def refine_response(self, document):
        system_content = "You are a main LLM. Refine the response document and ensure it meets user requirements. Ask clarifying questions if the user query is not detailed."
        response = self.client.chat.completions.create(
            model="mistralai/Mistral-7B-Instruct-v0.2",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": document}
            ],
            temperature=0.7,
            max_tokens=256
        )
        return response.choices[0].message.content
