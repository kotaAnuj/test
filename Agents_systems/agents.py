import openai
import re
from concurrent.futures import ThreadPoolExecutor
import sqlite3

def setup_database():
    conn = sqlite3.connect('agent_memory.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS memory (
            id INTEGER PRIMARY KEY,
            user_query TEXT,
            agent_response TEXT
        )
    ''')
    conn.commit()
    conn.close()

setup_database()

def store_memory(user_query, agent_response):
    conn = sqlite3.connect('agent_memory.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO memory (user_query, agent_response)
        VALUES (?, ?)
    ''', (user_query, agent_response))
    conn.commit()
    conn.close()

def retrieve_memory():
    conn = sqlite3.connect('agent_memory.db')
    cursor = conn.cursor()
    cursor.execute('SELECT user_query, agent_response FROM memory')
    rows = cursor.fetchall()
    conn.close()
    return rows

import pinecone

pinecone.init(api_key='your_pinecone_api_key', environment='us-west1-gcp')
index_name = "agent-memory"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=512)  # Dimension should match the embedding size
index = pinecone.Index(index_name)

def get_embedding(text):
    response = openai.Embedding.create(
    model="text-embedding-ada-002",
    input=text
    )
    return response["data"][0]["embedding"]

def store_embedding(user_query, agent_response, embedding):
    # Store the query and response as metadata
    index.upsert([(user_query, embedding, {"query": user_query, "response": agent_response})])

def retrieve_embeddings(query, top_k=5):
    embedding = get_embedding(query)
    results = index.query([embedding], top_k=top_k, include_metadata=True)
    return [(res["metadata"]["query"], res["metadata"]["response"]) for res in results["matches"]]

class LLM_Agent:
    def __init__(self, name, api_key):
        self.name = name
        self.api_key = api_key
        self.back_story = ""
        self.character = ""
        self.tools = {}
        self.communication_agents = {}
        setup_database()  # Initialize the database

    def set_attributes(self, back_story, character, tools, communication_agents):
        self.back_story = back_story
        self.character = character
        self.tools = tools
        self.communication_agents = communication_agents

    def execute_task(self, query):
        # Retrieve relevant memory
        past_interactions = retrieve_embeddings(query)
        memory_context = "\n".join([f"User: {q} \nAgent: {r}" for q, r in past_interactions])

        # Create the prompt using the template function
        prompt = create_prompt(
            agent_name=self.name,
            character=self.character,
            back_story=self.back_story,
            memory_context=memory_context,
            user_query=query
        )

        # Call the AI API to get an initial response
        client = openai.OpenAI(api_key=self.api_key, base_url="https://api.aimlapi.com")
        chat_completion = client.chat.completions.create(
            model="mistralai/Mistral-7B-Instruct-v0.2",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query},
            ],
            temperature=0.7,
            max_tokens=128,
        )
        response = chat_completion.choices[0].message.content

        # Process the response to determine if tools or other agents are needed
        response = self.process_response(response)

        # Generate embedding and store interaction in memory
        embedding = get_embedding(query)
        store_embedding(query, response, embedding)

        return response

    def process_response(self, response):
        tasks = []

        # Check for tools in the response
        for tool_name, tool_func in self.tools.items():
            tool_pattern = re.compile(rf"\{{({tool_name})\((.*?)\)\}}")
            match = tool_pattern.search(response)
            if match:
                param = match.group(2)
                tasks.append((tool_func, param, match.group(0), tool_name))

        # Check for other agents in the response
        for agent_name, agent in self.communication_agents.items():
            agent_pattern = re.compile(rf"\{{(agent_{agent_name})\((.*?)\)\}}")
            match = agent_pattern.search(response)
            if match:
                subtask = match.group(2)
                tasks.append((agent.execute_task, subtask, match.group(0), agent_name))

        # Execute tasks in parallel
        with ThreadPoolExecutor() as executor:
            future_to_task = {executor.submit(func, param): (placeholder, name) for func, param, placeholder, name in tasks}
            for future in future_to_task:
                placeholder, name = future_to_task[future]
                result = future.result()
                response = response.replace(placeholder, f"{name.capitalize()} Info: {result}")

        return response



def create_prompt(agent_name, character, back_story, memory_context, user_query, additional_instructions=""):
    """
    Generate a structured prompt for the AI model.
    
    Args:
        agent_name (str): Name of the agent.
        character (str): Character or personality of the agent.
        back_story (str): Background story of the agent.
        memory_context (str): Context from past interactions.
        user_query (str): The user's query.
        additional_instructions (str, optional): Any additional instructions for the agent.
        
    Returns:
        str: The formatted prompt.
    """
    prompt = f"""
    You are {agent_name}, a {character}.
    {back_story}
    
    Here is some context from previous interactions:
    {memory_context}
    
    User Query: {user_query}
    
    {additional_instructions}
    """
    return prompt.strip()



#test
import openai
import re
from concurrent.futures import ThreadPoolExecutor
 
# Define placeholder functions for API calls
def weather_api(location):
    return f"Weather information for {location}"

def serp_api(query):
    return f"Search results for {query}"

# Define embedding functions
def get_embedding(text):
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response["data"][0]["embedding"]

def store_embedding(user_query, agent_response, embedding):
    # Placeholder for storing embedding in a vector database
    pass

def retrieve_embeddings(query, top_k=5):
    # Placeholder for retrieving embeddings from a vector database
    return []

# Define prompt template function
def create_prompt(agent_name, character, back_story, memory_context, user_query, additional_instructions=""):
    prompt = f"""
    You are {agent_name}, a {character}.
    {back_story}
    
    Here is some context from previous interactions:
    {memory_context}
    
    User Query: {user_query}
    
    {additional_instructions}
    """
    return prompt.strip()

# Define the LLM_Agent class
class LLM_Agent:
    def __init__(self, name, api_key):
        self.name = name
        self.api_key = api_key
        self.back_story = ""
        self.character = ""
        self.tools = {}
        self.communication_agents = {}

    def set_attributes(self, back_story, character, tools, communication_agents):
        self.back_story = back_story
        self.character = character
        self.tools = tools
        self.communication_agents = communication_agents

    def execute_task(self, query):
        # Retrieve relevant memory
        past_interactions = retrieve_embeddings(query)
        memory_context = "\n".join([f"User: {q} \nAgent: {r}" for q, r in past_interactions])

        # Create the prompt using the template function
        prompt = create_prompt(
            agent_name=self.name,
            character=self.character,
            back_story=self.back_story,
            memory_context=memory_context,
            user_query=query
        )

        # Call the AI API to get an initial response
        client = openai.OpenAI(api_key=self.api_key, base_url="https://api.aimlapi.com")
        chat_completion = client.chat.completions.create(
            model="mistralai/Mistral-7B-Instruct-v0.2",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query},
            ],
            temperature=0.7,
            max_tokens=128,
        )
        response = chat_completion.choices[0].message.content

        # Process the response to determine if tools or other agents are needed
        response = self.process_response(response)

        # Generate embedding and store interaction in memory
        embedding = get_embedding(query)
        store_embedding(query, response, embedding)

        return response

    def process_response(self, response):
        tasks = []

        # Check for tools in the response
        for tool_name, tool_func in self.tools.items():
            tool_pattern = re.compile(rf"\{{({tool_name})\((.*?)\)\}}")
            match = tool_pattern.search(response)
            if match:
                param = match.group(2)
                tasks.append((tool_func, param, match.group(0), tool_name))

        # Check for other agents in the response
        for agent_name, agent in self.communication_agents.items():
            agent_pattern = re.compile(rf"\{{(agent_{agent_name})\((.*?)\)\}}")
            match = agent_pattern.search(response)
            if match:
                subtask = match.group(2)
                tasks.append((agent.execute_task, subtask, match.group(0), agent_name))

        # Execute tasks in parallel
        with ThreadPoolExecutor() as executor:
            future_to_task = {executor.submit(func, param): (placeholder, name) for func, param, placeholder, name in tasks}
            for future in future_to_task:
                placeholder, name = future_to_task[future]
                result = future.result()
                response = response.replace(placeholder, f"{name.capitalize()} Info: {result}")

        return response

# Create instances of LLM_Agent
weather_agent = LLM_Agent(name="WeatherAgent", api_key="your_openai_api_key_here")
info_agent = LLM_Agent(name="InfoAgent", api_key="your_openai_api_key_here")

# Set attributes for the agents
weather_agent.set_attributes(
    back_story="You provide weather updates.",
    character="a helpful weather assistant",
    tools={"weather_api": weather_api},
    communication_agents={}
)

info_agent.set_attributes(
    back_story="You provide information on various topics.",
    character="an informative assistant",
    tools={"serp_api": serp_api, "weather_api": weather_api},
    communication_agents={"WeatherAgent": weather_agent}
)

# User queries
user_query_1 = "Get the current weather for New York."
user_query_2 = "Find information about San Francisco and check the weather there."

# Execute tasks directly using user queries
print("Weather Agent Response:\n", weather_agent.execute_task(user_query_1))
print("\nInfo Agent Response:\n", info_agent.execute_task(user_query_2))
