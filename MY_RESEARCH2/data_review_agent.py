# data_review_agent.py
import openai

client = openai.OpenAI(
    api_key="d33ba10832e14e51a4dc57068d16e206",
    base_url="https://api.aimlapi.com",
)


def review_data(web_data, video_data, image_data):
    data_summary = "Summarize the following data:\n\n"
    
    for idx, item in enumerate(video_data):
        title = item.get('title', 'No title')
        description = item.get('description', 'No description')
        data_summary += f"Video {idx}: {title} - {description}\n"

    for idx, item in enumerate(web_data, 1):
        data_summary += f"Web {idx}: {item['snippet']}\n"
    
    for idx, item in enumerate(video_data, 1):
        data_summary += f"Video {idx}: {item['title']} - {item['description']}\n"
    
    for idx, item in enumerate(image_data, 1):
        data_summary += f"Image {idx}: {item['title']} - {item['link']}\n"
    
    chat_completion = client.chat.completions.create(
        model="mistralai/Mistral-7B-Instruct-v0.2",
        messages=[
            {"role": "system", "content": "Summarize the following information."},
            {"role": "user", "content": data_summary},
        ],
        temperature=0.7,
        max_tokens=500,
    )
    return chat_completion.choices[0].message.content