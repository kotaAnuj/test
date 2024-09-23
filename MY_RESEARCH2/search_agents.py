# search_agents.py
from serpapi import GoogleSearch
def get_web_data(query):
    params = {
        "engine": "google",
        "q": query,
        "api_key": "cff1141144fc78b9790b84e240e0cdf7f42b6021b9075698c179dda0abd464ce"
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    return results.get("organic_results", [])

def get_video_data(query):
    params = {
        "engine": "youtube",
        "search_query": query,
        "api_key": "cff1141144fc78b9790b84e240e0cdf7f42b6021b9075698c179dda0abd464ce"
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    return results.get("video_results", [])

def get_image_data(query):
    params = {
        "engine": "google_images",
        "q": query,
        "location": "Austin, TX, Texas, United States",
        "api_key": "cff1141144fc78b9790b84e240e0cdf7f42b6021b9075698c179dda0abd464ce"
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    return results.get("images_results", [])
