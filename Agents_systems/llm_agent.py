import openai
#from memory_management import retrieve_embeddings, store_embedding, get_embedding
from tools import weather_api, web_scraper, serp_api, code_executor

class LLM_Agent:
    def __init__(self, name, api_key):
        self.name = name
        self.api_key = api_key
        openai.api_key = api_key
        self.back_story = ""
        self.character = ""
        self.tools = {
            "weather_api": weather_api,
            "web_scraper": web_scraper,
            "serp_api": serp_api,
            "code_executor": code_executor
        }
        self.communication_agents = {}

    def set_attributes(self, back_story, character, tools, communication_agents):
        self.back_story = back_story
        self.character = character
        self.tools.update(tools)
        self.communication_agents = communication_agents

    def execute_task(self, query, user_id):
        #embedding = get_embedding(query)
        # Store the query embedding
        #store_embedding(conn, user_id, embedding)
        # Determine which tool to use based on query
        if "weather" in query:
            response = self.tools["weather_api"](query.split("in")[-1].strip())
        elif "scrape" in query:
            response = self.tools["web_scraper"](query.split("at")[-1].strip())
        elif "search" in query:
            response = self.tools["serp_api"](query.split("for")[-1].strip())
        elif "run" in query:
            response = self.tools["code_executor"](query.split("code:")[-1].strip())
        else:
            response = self.llm_query(query)
        return response

    def llm_query(self, query):
        client = openai.OpenAI(api_key=self.api_key, base_url="https://api.aimlapi.com")
        chat_completion = client.chat.completions.create(
                model="mistralai/Mistral-7B-Instruct-v0.2",
                messages=[
                    {"role": "system", "content": "you are a helpfull automation agent"},
                    {"role": "user", "content": query},
                ],
                temperature=0.7,
                max_tokens=128,
            )
        response = chat_completion.choices[0].message.content
        return response.choices[0].text.strip()
