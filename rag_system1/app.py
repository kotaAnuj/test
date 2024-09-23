import streamlit as st
from query_translation_agent import QueryTranslationAgent
from routing_agent import RoutingAgent
from vector_db_agent import VectorDBAgent
from graph_db_agent import GraphDBAgent
from relational_db_agent import RelationalDBAgent
from doc_ai_agent import DocAIAgent
from main_llm_agent import MainLLMAgent

# Initialize agents
api_key = "1723871e17b1432c9644a5b0d0e1574c"
serpapi_key = "cff1141144fc78b9790b84e240e0cdf7f42b6021b9075698c179dda0abd464ce"
vector_db_api_key = "b50b0162-43cc-4758-a182-fa7fb1f72eab"
index_name = "quickstart-index"
graph_db_uri = "neo4j+s://demo.neo4jlabs.com"
graph_db_user = "your_username_here"
graph_db_password = "lKmw2qzYzsuj8BbKGNVumG5cxN8BNF1fAsD6ePfX9FM"
relational_db_path = "database.db"

query_translation_agent = QueryTranslationAgent(api_key)
routing_agent = RoutingAgent(api_key, serpapi_key)
vector_db_agent = VectorDBAgent("b50b0162-43cc-4758-a182-fa7fb1f72eab","quickstart-index")
graph_db_agent = GraphDBAgent(graph_db_uri, graph_db_user, graph_db_password)
relational_db_agent = RelationalDBAgent(relational_db_path)
doc_ai_agent = DocAIAgent(api_key)
main_llm_agent = MainLLMAgent(api_key)

st.title("AI Assistant")

# File upload section
uploaded_file = st.file_uploader("Upload a file", type=["pdf", "docx", "txt"])
if uploaded_file:
    st.write("File uploaded successfully!")
    # Process the uploaded file and store embeddings
    file_content = uploaded_file.read().decode("utf-8")
    vector_db_agent.store_embeddings([file_content])
    graph_db_agent.store_data(file_content)
    relational_db_agent.store_data(file_content)

# User query input
user_query = st.text_input("Enter your query:")

if user_query:
    translated_query = query_translation_agent.translate_query(user_query)
    routing_response = routing_agent.route_query(translated_query)
    
    google_results, wikipedia_data = routing_agent.retrieve_data(translated_query)
    combined_data = google_results + [wikipedia_data]
    
    combined_data = []
    combined_data.extend(google_results)
    combined_data.append(wikipedia_data)

 # Validate combined_data before passing to doc_ai_agent
    if combined_data and any(combined_data):
        document = doc_ai_agent.compile_document(combined_data)
        final_response = main_llm_agent.refine_response(document)
        st.write("Response:", final_response)
    else:
        st.write("No relevant data found to compile the document.")


# Old data management
import os
import time

DATA_DIR = "data_files/"
WEEK_IN_SECONDS = 7 * 24 * 60 * 60

def clean_old_data():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    current_time = time.time()
    for filename in os.listdir(DATA_DIR):
        file_path = os.path.join(DATA_DIR, filename)
        if os.path.isfile(file_path):
            file_age = current_time - os.path.getctime(file_path)
            if file_age > WEEK_IN_SECONDS:
                os.remove(file_path)

clean_old_data()
