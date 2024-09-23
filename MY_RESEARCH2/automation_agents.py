# automation_agents.py
import openai

client = openai.OpenAI(
    api_key="d33ba10832e14e51a4dc57068d16e206",
    base_url="https://api.aimlapi.com",
)


def generate_automation_code(query):
    chat_completion = client.chat.completions.create(
        model="mistralai/Mistral-7B-Instruct-v0.2",
        messages=[
            {"role": "system", "content": "Generate an automation script."},
            {"role": "user", "content": query},
        ],
        temperature=0.7,
        max_tokens=128,
    )
    return chat_completion.choices[0].message.content

def execute_task(code):
    exec(code)
