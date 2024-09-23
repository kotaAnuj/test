import requests
from serpapi import GoogleSearch

class Agent:
    def __init__(self, role, task, backstory, tools):
        self.role = role
        self.task = task
        self.backstory = backstory
        self.tools = tools

    def describe(self):
        return f"Role: {self.role}\nTask: {self.task}\nBackstory: {self.backstory}\nTools: {', '.join(self.tools)}"

class ResearchAgent(Agent):
    def __init__(self, role, task, backstory, tools, serpapi_key, aiml_api_key, aiml_base_url):
        super().__init__(role, task, backstory, tools)
        self.serpapi_key = serpapi_key
        self.aiml_api_key = aiml_api_key
        self.aiml_base_url = aiml_base_url

    def perform_search(self, query):
        params = {
            "engine": "google",
            "q": query,
            "api_key": self.serpapi_key
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        organic_results = results.get("organic_results", [])
        return organic_results

    def summarize_results(self, results):
        content_to_summarize = "\n\n".join([result["snippet"] for result in results[:5]])
        payload = {
            "model": "mistralai/Mistral-7B-Instruct-v0.2",
            "messages": [
                {"role": "system", "content": "You are a travel agent. Be descriptive and helpful."},
                {"role": "user", "content": content_to_summarize}
            ],
            "temperature": 0.7,
            "max_tokens": 150
        }
        headers = {
            "Authorization": f"Bearer {self.aiml_api_key}",
            "Content-Type": "application/json"
        }
        response = requests.post(self.aiml_base_url, json=payload, headers=headers)
        response_json = response.json()

        # Debugging: Print response_json to understand its structure
        print(response_json)

        # Adjust based on the actual structure of response_json
        if "message" in response_json and "content" in response_json["message"]:
            summary = response_json["message"]["content"]
        else:
            summary = "Summary not available"

        return summary

    def get_ai_review(self, content):
        payload = {
            "model": "mistralai/Mistral-7B-Instruct-v0.2",
            "messages": [
                {"role": "system", "content": "You are a travel agent. Be descriptive and helpful."},
                {"role": "user", "content": content}
            ],
            "temperature": 0.7,
            "max_tokens": 150
        }
        headers = {
            "Authorization": f"Bearer {self.aiml_api_key}",
            "Content-Type": "application/json"
        }
        response = requests.post(self.aiml_base_url, json=payload, headers=headers)
        response_json = response.json()

        # Debugging: Print response_json to understand its structure
        print(response_json)

        # Adjust based on the actual structure of response_json
        if "message" in response_json and "content" in response_json["message"]:
            review = response_json["message"]["content"]
        else:
            review = "Review not available"

        return review

class ImagesAgent(ResearchAgent):
    def perform_image_search(self, query, location=""):
        params = {
            "engine": "google_images",
            "q": query,
            "api_key": self.serpapi_key
        }
        if location:
            params["location"] = location
        
        search = GoogleSearch(params)
        results = search.get_dict()
        image_results = results.get("images_results", [])
        return image_results

class VideosAgent(ResearchAgent):
    def perform_video_search(self, query):
        params = {
            "engine": "youtube",
            "search_query": query,
            "api_key": self.serpapi_key
        }
        
        search = GoogleSearch(params)
        results = search.get_dict()
        video_results = results.get("video_results", [])
        return video_results
