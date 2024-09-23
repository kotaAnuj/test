# main.py
import streamlit as st
from search_agents import get_web_data, get_video_data, get_image_data
from automation_agents import generate_automation_code, execute_task
from data_review_agent import review_data
import openai
# Initialize AIML API client

client = openai.OpenAI(
    api_key="d33ba10832e14e51a4dc57068d16e206",
    base_url="https://api.aimlapi.com",
)



def get_ai_response(system_content, user_content):
    chat_completion = client.chat.completions.create(
        model="mistralai/Mistral-7B-Instruct-v0.2",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
        temperature=0.7,
        max_tokens=128,
    )
    return chat_completion.choices[0].message.content

def main():
    st.title("Multi-Agent AI System")
    
    user_query = st.text_input("Enter your query:")
    if st.button("Submit"):
        if user_query:
            st.write("Processing...")

            # AI Response
            ai_response = get_ai_response("You are an assistant. Be descriptive and helpful.", user_query)
            st.write("AI Response:", ai_response)
            
            # Search Agents
            web_data = get_web_data(user_query)
            video_data = get_video_data(user_query)
            image_data = get_image_data(user_query)
            
            

            
            st.write("Web Data:", web_data)
            st.write("Video Data:", video_data)
            st.write("Image Data:", image_data)
            
            # Data Review
            reviewed_data = review_data(web_data, video_data, image_data)
            st.write("Reviewed Data:", reviewed_data)
            
            # Automation
            automation_code = generate_automation_code(user_query)
            execute_task(automation_code)
            
            
            st.write("Task Executed Successfully!")










if __name__ == "__main__":
    main()
