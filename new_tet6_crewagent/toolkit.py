import asyncio
from serpapi import GoogleSearch  # Import SerpAPI library

class ToolKit:
    def __init__(self):
        self.tools = {
            "web_search": self.web_search,
            "calculator": self.calculator,
            "geo_info": self.geo_info,
        }

    async def use_tool(self, tool_name, **kwargs):
        if tool_name in self.tools:
            print(f"[DEBUG] Using tool: {tool_name} with args: {kwargs}")  # Debug log
            result = await self.tools[tool_name](**kwargs)
            print(f"[DEBUG] Result from tool {tool_name}: {result}")  # Debug log
            return result
        else:
            return f"Tool {tool_name} not found"

    async def web_search(self, query, **kwargs):
        # Implement web search logic using SerpAPI
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self.serp_api, query)
        return result

    def serp_api(self, query):
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

    async def calculator(self, expression, **kwargs):
        # Implement calculation logic here
        try:
            return eval(expression)
        except Exception as e:
            return f"Error in calculation: {str(e)}"

    async def geo_info(self, location, **kwargs):
        # Implement geographical information retrieval here
        return f"Geo info for {location}"
