# main.py

from agent import InstructionAgent
from searchagent import SearchAgent
from browseragent import BrowserAgent
from dataagent import DataAgent
import logging
import asyncio

API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
API_KEY = "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"
SERPAPI_API_KEY = "cff1141144fc78b9790b84e240e0cdf7f42b6021b9075698c179dda0abd464ce"  # Replace with your actual SerpApi API key

def main():
    logging.basicConfig(level=logging.INFO)
    instruction_agent = InstructionAgent(API_URL, API_KEY)
    search_agent = SearchAgent(instruction_agent)
    browser_agent = BrowserAgent()
    data_agent = DataAgent(SERPAPI_API_KEY)

    while True:
        command = input("Enter command (search/browser/exit): ").lower()

        if command == "search":
            search_query = input("Enter your search query: ")
            result = search_agent.perform_search(search_query)
            if "Sorry" in result:
                print("Fetching additional information...")
                summary = data_agent.get_summary_from_wikipedia(search_query)
                print("Summary:", summary)
            else:
                print(result)
        
        elif command == "browser":
            search_query = input("Enter your search query: ")
            asyncio.run(browser_task(browser_agent, search_query))
            
        
        elif command == "exit":
            break
        
        else:
            print("Invalid command. Please enter 'search', 'browser', or 'exit'.")

async def browser_task(browser_agent, search_query=None):
    await browser_agent.start_browser()
    
    if search_query:
        # Example: Open YouTube and search for the query
        url = "https://www.youtube.com"
        await browser_agent.navigate_to(url)
        await asyncio.sleep(15)  # Wait for page load
        await browser_agent.fill_form('input[name="search_query"]', search_query)
        await browser_agent.click_button('button[id="search-icon-legacy"]')
        await asyncio.sleep(20)

    else:
        # Example: Open Google
        url = "https://www.google.com"
        await browser_agent.navigate_to(url)
        await asyncio.sleep(2)  # Wait for page load
        # Additional tasks can be performed here
    
    await asyncio.sleep(5)  # Allow some time for user interaction
    await browser_agent.close_browser()

if __name__ == "__main__":
    main()
