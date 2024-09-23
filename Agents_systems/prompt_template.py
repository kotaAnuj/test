# prompt_template.py

def create_prompt(agent_name, character, back_story, memory_context, user_query, additional_instructions, user_preferences):
    prompt = f"""
    You are {agent_name}, a {character}. {back_story}
    Here is some context from your memory:
    {memory_context}

    User query: {user_query}

    {additional_instructions}

    User preferences: {user_preferences}
    """
    return prompt
