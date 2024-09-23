import openai
from serpapi import GoogleSearch
import wikipediaapi
api_key = "1723871e17b1432c9644a5b0d0e1574c"
serpapi_key = "cff1141144fc78b9790b84e240e0cdf7f42b6021b9075698c179dda0abd464ce"
import requests

class RoutingAgent:
    def __init__(self, api_key, serpapi_key):
        self.api_key = api_key
        self.serpapi_key = serpapi_key

    def route_query(self, query):
        # Implement your routing logic here
        return query

    def search_with_serpapi(self, query):
        params = {
            "q": query,
            "api_key": self.serpapi_key,
            "engine": "google"
        }
        response = requests.get("https://serpapi.com/search", params=params)
        if response.status_code == 200:
            return response.json().get("organic_results", [])
        else:
            return []

    def retrieve_data(self, query):
        google_results = self.search_with_serpapi(query)
        wikipedia_data = self.search_wikipedia(query)
        return google_results, wikipedia_data

    def search_wikipedia(self, query):
        # Example function to search Wikipedia
        response = requests.get(f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={query}&format=json")
        if response.status_code == 200:
            search_results = response.json().get("query", {}).get("search", [])
            if search_results:
                page_id = search_results[0].get("pageid", "")
                page_response = requests.get(f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&pageids={page_id}&format=json")
                if page_response.status_code == 200:
                    page_data = page_response.json().get("query", {}).get("pages", {}).get(str(page_id), {}).get("extract", "")
                    return page_data
        return ""   


  