import logging
import json
import asyncio
import os
import numpy as np
from typing import Dict, Any, List

# Importing the LLMConnector
from llm_connector import LLMConnector

# Practical tools for agents
def search_function(query: str) -> str:
    return f"Search results for: {query}"

def calculator_function(expression: str) -> str:
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {str(e)}"

def get_current_datetime() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_weather(location: str) -> str:
    return f"Weather in {location}: Sunny, 22°C"

# Tool registry
TOOLS = {
    "search": search_function,
    "calculator": calculator_function,
    "get_current_datetime": get_current_datetime,
    "get_weather": get_weather,
}

class TestingModule:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def run_tests(self, agent_code: str) -> tuple:
        self.logger.info("Running tests on the generated agent code")
        accuracy = np.random.uniform(0.8, 1.0)
        performance = np.random.uniform(0.8, 1.0)
        return accuracy, performance

    def is_performance_satisfactory(self, accuracy: float, performance: float) -> bool:
        return accuracy > 0.9 and performance > 0.8

class AIAgentCreator:
    def __init__(self, api_key: str, model_name: str = "codellama/CodeLlama-34b-Instruct-hf", base_url: str = "https://api.aimlapi.com/v1"):
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        self.llm_connector = LLMConnector(api_key, model_name, base_url)
        self.testing_module = TestingModule()
        self.agent_classes = {}
        

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("ai_agent_creator.log"),
                logging.StreamHandler()
            ]
        )

    import re
    import ast

    async def create_agent(self, task_description: str, agent_type: str):
        try:
            messages = [{"role": "user", "content": f"Generate Python code for an AI agent that {task_description} for a {agent_type} agent."}]
            agent_code = await self.llm_connector.connect(messages)

            logging.info("Raw agent code response from LLM:")
            logging.info(agent_code)

            cleaned_code = self.clean_generated_code(agent_code)
            namespace = {}

            try:
                compiled_code = compile(cleaned_code, "<string>", "exec")
                exec(compiled_code, namespace)
            except IndentationError as e:
                logging.error(f"Indentation error in generated code: {e}")
                return None
            except SyntaxError as e:
                logging.error(f"Syntax error in generated code: {e}")
                return None
            except Exception as e:
                logging.error(f"An error occurred while executing the code: {e}")
                return None

            return namespace.get('agent', None)

        except Exception as e:
            logging.error(f"An error occurred: {e}")
            return None

    def clean_generated_code(self, agent_code):
        import re
        # Remove unnecessary text before and after the code block
        cleaned_code = re.sub(r'^.*?```(?:python)?\s*', '', agent_code, flags=re.DOTALL)
        cleaned_code = re.sub(r'```.*$', '', cleaned_code, flags=re.DOTALL)
        
        # Optional: Further clean the code for any common issues
        cleaned_code = re.sub(r'\r\n', '\n', cleaned_code)  # Convert Windows line endings to Unix
        cleaned_code = re.sub(r'^\s+', '', cleaned_code)  # Remove leading whitespace
        
        return cleaned_code

    async def generate_design(self, task_description: str, agent_type: str) -> Dict[str, Any]:
        prompt = f"Design an AI agent architecture for the following task: {task_description}. Agent type: {agent_type}"
        response = await self.llm_connector.connect([{"role": "user", "content": prompt}])
        
        # Debugging print statement to inspect response
        print(f"API Response: {response}")

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            self.logger.error(f"Invalid JSON response: {response}")
            return {"raw_text": response} 


    async def generate_code(self, agent_design: Dict[str, Any], agent_type: str) -> str:
        prompt = f"Generate Python code for an AI agent with the following design: {json.dumps(agent_design)}. Agent type: {agent_type}. Include practical tools like search, calculator, get_current_datetime, and get_weather."
        return await self.llm_connector.connect([{"role": "user", "content": prompt}])

class Agent:
    def __init__(self, name: str, role: str, input_spec: tuple, output_spec: int, llm_connector: LLMConnector):
        self.name = name
        self.role = role
        self.input_spec = input_spec
        self.output_spec = output_spec
        self.llm_connector = llm_connector
        self.memory = []

    async def perform_task(self, task: str) -> str:
        prompt = self.construct_prompt(task)
        response = await self.llm_connector.connect([{"role": "user", "content": prompt}])
        self.memory.append({"task": task, "response": response})
        return self.process_response(response)

    def construct_prompt(self, task: str) -> str:
        context = "\n".join([f"Previous task: {m['task']}, Response: {m['response']}" for m in self.memory[-3:]])
        tools_description = "\n".join([f"- {name}: {func.__doc__ or 'No description'}" for name, func in TOOLS.items()])
        return f"""
        You are {self.name}, an AI agent with the role of {self.role}.
        Your task: {task}
        Input spec: {self.input_spec}, Output spec: {self.output_spec}
        
        Available tools:
        {tools_description}
        
        Recent context:
        {context}
        
        Provide your response, including any tool usage if necessary.
        Use the format: [TOOL_NAME](argument) to use a tool.
        """

    def process_response(self, response: str) -> str:
        import re
        try:
            response = json.loads(response)
            # Process the valid JSON response
        except json.JSONDecodeError:
            print("Invalid JSON response:", response)
        def tool_caller(match):
            tool_name = match.group(1)
            argument = match.group(2)
            if tool_name in TOOLS:
                return str(TOOLS[tool_name](argument))
            return f"Error: Tool '{tool_name}' not found"

        return re.sub(r'\[(\w+)\]\(([^)]+)\)', tool_caller, response)

class CrewAgent:
    def __init__(self, llm_connector: LLMConnector):
        architecture = {'team_size': 3, 'roles': ['analyst', 'strategist', 'executor']}
        input_spec = (10,)
        output_spec = 1
        self.team = [Agent(f"SubAgent_{i}", role, input_spec, output_spec, llm_connector) for i, role in enumerate(architecture['roles'])]

    async def coordinate(self, task: str) -> str:
        results = await asyncio.gather(*[agent.perform_task(task) for agent in self.team])
        return "\n".join(results)

class SoloAgent(Agent):
    def __init__(self, llm_connector: LLMConnector):
        super().__init__("SoloAgent", "general task execution", (10,), 1, llm_connector)

class CollaborativeAgent:
    def __init__(self, llm_connector: LLMConnector):
        architecture = {'team_size': 5, 'roles': ['researcher', 'analyst', 'strategist', 'executor', 'reviewer']}
        input_spec = (20,)
        output_spec = 2
        self.team = [Agent(f"CollabAgent_{i}", role, input_spec, output_spec, llm_connector) for i, role in enumerate(architecture['roles'])]

    async def collaborate(self, task: str) -> str:
        results = []
        for agent in self.team:
            result = await agent.perform_task(task)
            results.append(result)
            task = f"Based on this input: {result}\n{task}"
        return "\n".join(results)

async def handle_user_input(user_input: str, creator: AIAgentCreator):
    if user_input.lower().startswith("crew:"):
        agent = await creator.create_agent(user_input[5:].strip(), "crew")
        if agent:
            result = await agent(creator.llm_connector).coordinate(user_input[5:].strip())
        else:
            result = "Failed to create Crew Agent"
    elif user_input.lower().startswith("solo:"):
        agent = await creator.create_agent(user_input[5:].strip(), "solo")
        if agent:
            result = await agent(creator.llm_connector).perform_task(user_input[5:].strip())
        else:
            result = "Failed to create Solo Agent"
    elif user_input.lower().startswith("collab:"):
        agent = await creator.create_agent(user_input[7:].strip(), "collaborative")
        if agent:
            result = await agent(creator.llm_connector).collaborate(user_input[7:].strip())
        else:
            result = "Failed to create Collaborative Agent"
    else:
        result = "Please start your command with 'crew:', 'solo:', or 'collab:' to specify the agent type."
    
    print(result)

import json

# Function to clean generated code by removing unwanted phrases
def clean_generated_code(agent_code):
    unwanted_phrases = ["Here is", "In the following", "Example:", "Let's generate"]
    for phrase in unwanted_phrases:
        agent_code = agent_code.replace(phrase, "")
    return agent_code.strip()

async def create_agent(self, task_description, agent_type):
    try:
        # Send request to LLM and receive the agent code (this is a placeholder, adjust accordingly)
        agent_code = await self.send_request_to_llm(task_description, agent_type)
        
        # Log or print the raw response from LLM for debugging
        print("Raw agent code response from LLM:")
        print(agent_code)

        # Clean the code response before executing it
        cleaned_code = clean_generated_code(agent_code)

        # Set up a namespace for exec
        namespace = {}

        # Execute the cleaned code
        try:
            exec(cleaned_code, namespace)
        except IndentationError as e:
            print(f"Indentation error in generated code: {e}")
            return None  # Handle the error appropriately
        except Exception as e:
            print(f"An error occurred while executing the code: {e}")
            return None  # Handle other errors

        # Return the instantiated agent from the namespace
        return namespace.get('agent', None)

    except json.JSONDecodeError:
        print("Received invalid JSON from the LLM API. Here's the raw response:")
        print(agent_code)
        return None  # Handle JSON error appropriately
    except Exception as e:
        print(f"An error occurred: {e}")
        return None  # Handle other generic errors

async def main():
    api_key = "2d2d8893395640a89cf366271e1a84fe"  # Make sure to use your actual API key
    creator = AIAgentCreator(api_key)

    print("Welcome to the Practical AI Agent System!")
    print("You can interact with different types of agents:")
    print("- Crew Agent: Start your command with 'crew:'")
    print("- Solo Agent: Start your command with 'solo:'")
    print("- Collaborative Agent: Start your command with 'collab:'")
    print("Type 'exit' to quit.")

    while True:
        user_input = input("\nEnter your command: ")
        if user_input.lower() == 'exit':
            break
        await handle_user_input(user_input, creator)

if __name__ == "__main__":
    asyncio.run(main())