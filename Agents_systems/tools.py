import requests
from bs4 import BeautifulSoup
from serpapi import GoogleSearch

def weather_api(location):
    # Example function to call a weather API
    return f"Weather for {location}"

def web_scraper(url):
    # Example web scraper function
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    return soup.title.string

def serp_api(query):
    params = {
        "engine": "google",
        "q": query,
        "api_key": "cff1141144fc78b9790b84e240e0cdf7f42b6021b9075698c179dda0abd464ce"
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    organic_results = results.get("organic_results", [])
    return [result["title"] for result in organic_results]

def code_executor(code):
    import subprocess
    result = subprocess.run(code, shell=True, capture_output=True, text=True)
    return result.stdout
