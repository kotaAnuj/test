import streamlit as st
from llm_agent import LLM_Agent
import random

# Initialize the Streamlit app
st.title("AI Agent Interface")
st.write("Interact with the AI Agent for web automation and information retrieval.")

# Generate or retrieve a random numeric user ID
def get_user_id():
    if 'user_id' not in st.session_state:
        st.session_state['user_id'] = str(random.randint(100000, 999999))  # Generate a random 6-digit number
    return st.session_state['user_id']

user_id = get_user_id()
st.write(f"User ID: {user_id}")  # Display the user ID for debugging purposes

# Input fields for user queries
query = st.text_input("Enter your query:")

# Define the agent
agent = LLM_Agent(name="WebInfoAgent", api_key="your_openai_api_key")

# Set the agent's attributes
agent.set_attributes(
    back_story="I am an AI agent specialized in web automation and information retrieval.",
    character="Helpful and efficient.",
    tools={},  # Additional tools can be added here if needed
    communication_agents={}
)

# Function to execute the task and get the response
def get_response(query):
    response = agent.execute_task(query, user_id)
    return response

# Button to execute the task
if st.button("Execute Task"):
    if query:
        response = get_response(query)
        st.write("Response from the AI Agent:")
        st.write(response)
    else:
        st.write("Please enter a query.")
