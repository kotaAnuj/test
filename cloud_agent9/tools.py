# tools.py
import webbrowser
import os
import subprocess
import requests
from bs4 import BeautifulSoup
from serpapi import GoogleSearch
import asyncio
import json
from typing import Callable, Any, Dict
from pydantic import BaseModel, Field
from typing import Callable, Any, Dict

class Tool(BaseModel):
    name: str
    function: Callable
    description: str = ""

    class Config:
        arbitrary_types_allowed = True

    async def execute(self, *args, **kwargs) -> Dict[str, Any]:
        try:
            result = await self.function(*args, **kwargs) if asyncio.iscoroutinefunction(self.function) else self.function(*args, **kwargs)
            return {"status": "success", "data": result}
        except Exception as e:
            return {"status": "error", "message": str(e)}
        
        
async def weather_api(location):
    """Get the current weather for a given location."""
    return f"The current weather in {location} is sunny."

def open_youtube(query):
    # Define the parameters for the search
    params = {
        "engine": "youtube",
        "search_query": query,
        "api_key": "cff1141144fc78b9790b84e240e0cdf7f42b6021b9075698c179dda0abd464ce"
    }

    # Perform the search
    search = GoogleSearch(params)
    results = search.get_dict()

    # Initialize an empty list to hold video results
    video_results = []

    # Check if video results are present in the response
    if "video_results" in results:
        for video in results["video_results"]:
            video_info = {
                "title": video["title"],  # Video title
                "link": video["link"],    # Video link
                "thumbnail": video["thumbnail"]  # Video thumbnail URL
            }
            video_results.append(video_info)  # Add the video info to the list

    # Return the results as a JSON-formatted string
    return json.dumps(video_results, indent=2)

# Example usage



    #"""Open YouTube with the specified search query."""
    #search_url = f"https://www.youtube.com/results?search_query={query}"
    #webbrowser.open(search_url)
    #return {"status": "success", "message": f"Opened YouTube with query: {query}"}



def serp_api(query):
    """Perform a search using SerpAPI and return top search results."""
    try:
        params = {
            "engine": "google",
            "q": query,
            "api_key": "cff1141144fc78b9790b84e240e0cdf7f42b6021b9075698c179dda0abd464ce"
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        return results.get("organic_results", [])
    except Exception as e:
        return {"status": "error", "message": str(e)}

def document_loader(doc_path):
    """Load the content of a document."""
    try:
        if os.path.exists(doc_path):
            with open(doc_path, 'r') as file:
                return {"status": "success", "data": file.read()}
        else:
            return {"status": "error", "message": "Document not found."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def create_file(file_path, content):
    """Create a new file with the given content."""
    try:
        with open(file_path, 'w') as file:
            file.write(content)
        return {"status": "success", "message": f"File {file_path} created."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def create_folder(folder_path):
    """Create a new folder."""
    try:
        os.makedirs(folder_path, exist_ok=True)
        return {"status": "success", "message": f"Folder {folder_path} created."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

async def code_executor(code):
    """Execute Python code asynchronously and return the result."""
    try:
        result = await asyncio.create_subprocess_exec(
            'python', '-c', code,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, stderr = await result.communicate()
        if result.returncode == 0:
            return {"status": "success", "data": stdout.decode()}
        else:
            return {"status": "error", "message": stderr.decode()}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def web_crawler(url):
    """Crawl a webpage and extract all links."""
    try:
        response =  requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        links = [link.get('href') for link in soup.find_all('a') if link.get('href')]
        return {"status": "success", "data": links}
    except Exception as e:
        return {"status": "error", "message": str(e)}

async def web_scraper(url):
    """Scrape a webpage and return its text content."""
    try:
        response = await requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text(separator=' ', strip=True)
        return {"status": "success", "data": text}
    except Exception as e:
        return {"status": "error", "message": str(e)}

async def make_api(workflow_id, data):
    """Trigger a Make.com API workflow."""
    url = f"https://api.make.com/v2/workflows/{workflow_id}/executions"
    headers = {
        "Authorization": "Bearer 117535c6-4bd6-4070-ad1c-d9a29e27b0ec",
        "Content-Type": "application/json"
    }
    try:
        response = await requests.post(url, json=data, headers=headers)
        if response.status_code == 200:
            return {"status": "success", "data": response.json()}
        else:
            return {"status": "error", "message": f"API request failed with status code: {response.status_code}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

async def zapier_webhook(webhook_url, data):
    """Send data to a Zapier webhook."""
    try:
        response = await requests.post(webhook_url, json=data)
        return {"status": "success", "data": response.json()}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def open_youtube(query):
    """Open YouTube with the specified search query."""
    search_url = f"https://www.youtube.com/results?search_query={query}"
    webbrowser.open(search_url)
    return {"status": "success", "message": f"Opened YouTube with query: {query}"}



# Tool registry
tools = {
    "WeatherAPI": Tool(name="WeatherAPI", function=weather_api, description="Fetches current weather information. Input: location. Output: weather description."),
    "SerpAPI": Tool(name="SerpAPI", function=serp_api, description="Fetches top search results using SerpAPI. Input: search query. Output: list of search results."),
    "DocumentLoader": Tool(name="DocumentLoader", function=document_loader, description="Loads content from a document. Input: document path. Output: document content."),
    "FileCreator": Tool(name="FileCreator", function=create_file, description="Creates a file with the given content. Input: file path and content. Output: confirmation message."),
    "FolderCreator": Tool(name="FolderCreator", function=create_folder, description="Creates a folder at the specified path. Input: folder path. Output: confirmation message."),
    "CodeExecutor": Tool(name="CodeExecutor", function=code_executor, description="Executes a given Python code snippet. Input: code. Output: execution result."),
    "WebCrawler": Tool(name="WebCrawler", function=web_crawler, description="Crawls a webpage and extracts all links. Input: URL. Output: list of links."),
    "WebScraper": Tool(name="WebScraper", function=web_scraper, description="Scrapes a webpage for text content. Input: URL. Output: page text."),
    "MakeAPI": Tool(name="MakeAPI", function=make_api, description="Makes a request to a Make.com API. Input: workflow ID and data. Output: API response."),
    "ZapierWebhook": Tool(name="ZapierWebhook", function=zapier_webhook, description="Sends data to a Zapier webhook. Input: webhook URL and data. Output: webhook response."),
    "YouTubeSearch": Tool(name="YouTubeSearch", function=open_youtube, description="Fetch top search with a given query. Input: search query. Output: confirmation message."),
}