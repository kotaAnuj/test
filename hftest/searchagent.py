# searchagent.py

from agent import InstructionAgent

class SearchAgent:
    def __init__(self, agent):
        self.agent = agent
    
    def perform_search(self, search_query):
        payload = {"inputs": search_query}
        result = self.agent.query(payload)
        if result:
            return result
        return "No results found."

    def detailed_search(self, base_query, details):
        full_query = f"{base_query} {details}"
        return self.perform_search(full_query)
