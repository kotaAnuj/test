from memory_database import MemoryDatabase
import streamlit as st
from agent_communication import AgentCommunication
from search_ai_agent import SearchAIAgent
from tools import weather_tool, serp_tool, web_scraper_tool, youtube_tool, web_crawler_tool
import asyncio

# Initialize Memory Database
memory_db = MemoryDatabase()

def create_agents():
    # Sub-Agent 2: Video Search Agent
    agent2 = SearchAIAgent("Agent 2", api_key="04dc65dc54044d4aaab35f3643bc7727", memory_db=memory_db)
    agent2.set_attributes(
        character="Video Researcher.",
        back_story="I am an AI agent specialized in retrieving videos related to the user.",
        task="Retrieve the top 3 video results based on the user's query.",
        tools={"SERP": serp_tool}
    )

    # Sub-Agent 3: Weather Information Agent
    agent3 = SearchAIAgent("Agent 3", api_key="04dc65dc54044d4aaab35f3643bc7727", memory_db=memory_db)
    agent3.set_attributes(
        character="Weather Specialist",
        back_story="Provides weather information.",
        task="Provide weather updates based on the user's location.",
        tools={"WebScraper": web_scraper_tool, "WeatherAPI": weather_tool}
    )

    # Agent 1: Coordinator Agent
    agent1 = SearchAIAgent("Agent 1", api_key="04dc65dc54044d4aaab35f3643bc7727", memory_db=memory_db)
    agent1.set_attributes(
        character="Coordinator.",
        back_story="I am an AI agent responsible for planning and distributing tasks to sub-agents.",
        task="Take the user's query, distribute tasks to the appropriate sub-agents, and compile their responses.",
        tools={},  # Agent 1 doesn't need any specific tools, it coordinates the others
        sub_agents={"Agent 2": agent2, "Agent 3": agent3}  # Attach sub-agents
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
        st.write("**Agent 1** is planning the tasks...")
        try:
            response = asyncio.run(agent_communication.collaborate(query))
            st.write("**Final Response:**")
            st.write(response)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Close the memory database connection on exit
memory_db.close()
