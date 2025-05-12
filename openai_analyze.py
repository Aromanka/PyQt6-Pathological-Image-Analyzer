from openai import OpenAI

def construct_prompt(system_prompt, user_prompt):

    message = [ {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                ]

    return message

def ask_AI(system_prompt, user_prompt, API_KEY, base_url="https://api.deepseek.com/v1", model="deepseek-chat"):
    client = OpenAI(api_key=API_KEY, base_url=base_url)
    messages = construct_prompt(system_prompt, user_prompt)
    stream_response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        stream=True
    )
    return stream_response
