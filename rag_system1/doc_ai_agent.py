import openai
api_key = "1723871e17b1432c9644a5b0d0e1574c"
class DocAIAgent:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.aimlapi.com"
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def compile_document(self, data):
        if not data or not isinstance(data, list) or len(data) == 0:
            return "No data provided to compile."

        # Extract text content from each dictionary
        text_data = []
        for item in data:
            if isinstance(item, dict):
                text_data.append(item.get("content", ""))
            elif isinstance(item, str):
                text_data.append(item)

        system_content = "You are a document compiler. Combine the following data into a coherent document."
        user_content = " ".join(text_data)
        
        chat_completion = self.client.chat.completions.create(
            model="mistralai/Mistral-7B-Instruct-v0.2",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            temperature=0.7,
            max_tokens=512,  # Adjusted for more comprehensive responses
        )
        
        response = chat_completion.choices[0].message.content
        return response

    def refine_response(self, document):
        if not document or document == "No data provided to compile.":
            return "No document provided to refine."

        system_content = "You are a response refiner. Refine the following document to make it clear and concise."
        user_content = document
        
        chat_completion = self.client.chat.completions.create(
            model="mistralai/Mistral-7B-Instruct-v0.2",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            temperature=0.7,
            max_tokens=512,  # Adjusted for more comprehensive responses
        )
        
        response = chat_completion.choices[0].message.content
        return response


