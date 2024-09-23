# main.py
import asyncio
from agent import Agent
from task import Task
from tools import tools
from llm_connector import LLMConnector

async def main():
    llm_connector = LLMConnector(api_key="9028f044d5ca413ca4dc918dd13aa5ad")

    research_assistant = Agent(
        name="Research Assistant",
        role="Information Gatherer",
        goal="Find information relevant to the task",
        backstory="You are an AI research assistant with a keen eye for detail and a passion for learning.",
        tools={k: v for k, v in tools.items() if k in ["WeatherAPI", "SerpAPI"]},
        llm_connector=llm_connector
    )

    task = Task(description="What is the capital of Germany?")
    result = await research_assistant.execute_task(task)
    print(result)


    code_helper = Agent(
    name="Code Helper",
    role="Programming Assistant",
    goal="Assist with coding tasks and provide code explanations",
    backstory="You are an AI coding expert with extensive knowledge of multiple programming languages.",
    tools={k: v for k, v in tools.items() if k in ["CodeExecutor"]},
    llm_connector=llm_connector
    )

    coding_task = Task(description="Create a Python function to calculate the fibonacci sequence up to n terms.")
    coding_result = await code_helper.execute_task(coding_task)
    print(coding_result)

if __name__ == "__main__":
    asyncio.run(main())