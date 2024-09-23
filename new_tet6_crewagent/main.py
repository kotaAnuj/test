# main.py
import asyncio
from hybrid_crew_chain import HybridCrewChain

async def main():
    llm_config = {
        'provider': 'mistral',
        'api_key': '04dc65dc54044d4aaab35f3643bc7727',
        'base_url': 'https://api.aimlapi.com',
        'model': 'mistralai/Mistral-7B-Instruct-v0.2'
    }
    
    # Create HybridCrewChain with rate limiting parameters
    crew_chain = HybridCrewChain(llm_config, calls_per_minute=5, burst_limit=10)

    # Add agents
    crew_chain.add_agent("Alice", "Researcher", ["search", "analyze"], ["Find accurate information"])
    crew_chain.add_agent("Bob", "Calculator", ["calculate", "compute"], ["Provide accurate calculations"])
    crew_chain.add_agent("Charlie", "Geographer", ["locate", "describe"], ["Provide geographical information"])

    # Run a task
    task = "What are the top 3 attractions in New York City?"
    result = await crew_chain.run(task)
    print(f"Result: {result}")

if __name__ == "__main__":
    asyncio.run(main())