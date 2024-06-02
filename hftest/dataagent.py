# dataagent.py

import wikipedia
from serpapi import GoogleSearch

class DataAgent:
    def __init__(self, serpapi_api_key):
        self.serpapi_api_key = serpapi_api_key

    def get_summary_from_wikipedia(self, term):
        try:
            summary = wikipedia.summary(term, sentences=2)
            return summary
        except wikipedia.exceptions.DisambiguationError as e:
            return wikipedia.summary(e.options[0], sentences=2)
        except wikipedia.exceptions.PageError:
            return self.get_summary_from_serpapi(term)

    def get_summary_from_serpapi(self, term):
        params = {
            "q": term,
            "num": 1,
            "api_key": self.serpapi_api_key
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        if "organic_results" in results and len(results["organic_results"]) > 0:
            return results["organic_results"][0]["snippet"]
        else:
            return "No summarized information available."
