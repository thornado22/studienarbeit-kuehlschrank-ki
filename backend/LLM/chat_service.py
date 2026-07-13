import os
import json
from groq import Groq
from pathlib import Path
import project_config

PROMPTS_PATH = Path(__file__).resolve().parents[1] / project_config.PROMPT

# Chat history object
chat = {
    "messages": []
}

# Load API key from environment variable or fallback to demo config
def get_groq_api_key() -> str:

    api_key = project_config.GROQ_API_KEY
    if api_key:
        return api_key

    raise RuntimeError(
        "Groq API key is missing. Set GROQ_API_KEY"
    )
GROQ_API_KEY = get_groq_api_key()

def load_prompts() -> dict:
    with open(PROMPTS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)

def build_followup_with_images_prompt(question: str, additional_contents: list[str]) -> str:
    prompts = load_prompts()
    template = prompts["followup_with_images"]["template"]

    return template.format(
        question=question,
        additional_contents=", ".join(additional_contents)
    )

def get_groq_response(prompt: str, use_history: bool = True) -> str:
    client = Groq(api_key=os.environ.get("GROQ_API_KEY", GROQ_API_KEY))
    
    if use_history:
        messages = chat["messages"].copy()
        messages.append({"role": "user", "content": prompt})
    else:
        messages = [{"role": "user", "content": prompt}]
    
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages
    )
    
    response = completion.choices[0].message.content
    
    # Add to chat history if enabled
    if use_history:
        chat["messages"].append({"role": "user", "content": prompt})
        chat["messages"].append({"role": "assistant", "content": response})
    
    return response


def get_initial_answer(fridge_contents: list) -> str:
    prompts = load_prompts()
    template = prompts["initial_recipe"]["template"]

    content = template.format(
        fridge_contents=", ".join(fridge_contents)
    )

    chat["messages"].clear()
    answer = get_groq_response(content, use_history=True)
    return answer


def get_follow_up_answer(user_input: str) -> str:
    answer = get_groq_response(user_input, use_history=True)
    return answer
