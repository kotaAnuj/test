from memory_database import MemoryDatabase
import streamlit as st
from agent_communication import AgentCommunication
from search_ai_agent import SearchAIAgent
from tools import weather_tool, serp_tool, web_scraper_tool, youtube_tool, web_crawler_tool
import asyncio

# Initialize Memory Database
memory_db = MemoryDatabase()

def create_agents():
    # Initialize agents with different tasks and tools
    agent1 = SearchAIAgent("Agent 1", api_key="9028f044d5ca413ca4dc918dd13aa5ad", memory_db=memory_db)
    agent1.set_attributes(
        character="Informative and concise.",
        back_story="I am an AI agent specialized in retrieving and summarizing search results.",
        task="Retrieve the top 5 search results and generate a helpful response based on the user's query, only give the websites related to the query.",
        tools={"WebAutomation": web_crawler_tool, "SERP": serp_tool}
    )

    agent2 = SearchAIAgent("Agent 2", api_key="9028f044d5ca413ca4dc918dd13aa5ad", memory_db=memory_db)
    agent2.set_attributes(
        character="Researcher.",
        back_story="I am an AI agent specialized in retrieving videos related to the user.",
        task="Retrieve the top 3 search results and generate a helpful response based on the user's query, only give the video related to the query.",
        tools={"YouTube": youtube_tool, "SERP": serp_tool}
    )

    agent3 = SearchAIAgent("Agent 3", api_key="9028f044d5ca413ca4dc918dd13aa5ad", memory_db=memory_db)
    agent3.set_attributes(
        character="Weather Specialist",
        back_story="Provides weather information.",
        task="Give weather updates based on the user's location.",
        tools={"WebScraper": web_scraper_tool, "WeatherAPI": weather_tool}
    )

    return {"Agent 1": agent1, "Agent 2": agent2, "Agent 3": agent3}

# Streamlit UI
st.title("Multi-Agent Search System")

query = st.text_input("Enter your search query:")

if st.button("Search"):
    agents = create_agents()
    agent_communication = AgentCommunication(agents, memory_db)

    # Monitor agents        
    agent_communication.monitor_agents()

    if query:
        try:
            response = asyncio.run(agent_communication.collaborate(query))
            st.write(response)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Close the memory database connection on exit
memory_db.close()
