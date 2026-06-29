import os
from google import genai
from dotenv import load_dotenv

load_dotenv()  # Loads the environment variables from .env

# Initialize the client (ensure GEMINI_API_KEY is set in your environment)
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# Generate content
response = client.models.generate_content(
    model='gemini-2.5-flash', # Or use gemini-3.5-flash for the latest model
    contents='Explain how AI works in a few words'
)

print(response.text)