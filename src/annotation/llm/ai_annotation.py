import os
from pathlib import Path
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()  # Loads the environment variables from .env

# Initialize the client (ensure GEMINI_API_KEY is set in your environment)
client = genai.Client(api_key=os.environ.get("GEMINI_ANNOTATION_API_KEY"))

# Read example prompts from md files

prompt_path = Path("prompts/annotator/annotation_prompt.md")
system_path = Path("prompts/annotator/system_prompt.md")

with open(prompt_path, "r", encoding="utf-8") as prompt:
    user_content = prompt.read()

with open(system_path, "r", encoding="utf-8") as system:
    system_prompt = system.read()

# Generate content with 
response = client.models.generate_content(
    model='gemini-2.5-flash', 
    contents=user_content,
    config=types.GenerateContentConfig(
        system_instruction=system_prompt
    )
)

print(response.text)