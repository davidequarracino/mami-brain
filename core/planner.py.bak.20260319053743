import os
from dotenv import load_dotenv
from openai import OpenAI
from api.schema import MamiPlan

# Load environment variables (API keys, configurations) from .env file
load_dotenv()

# Initialize the OpenAI client
# The API key is securely retrieved from the environment
client = OpenAI(api_key=os.getenv("LLM_API_KEY"))

def generate_mami_plan(user_input: str, world_state: str, history: list = []) -> MamiPlan:
    """
    Translates a natural language instruction into a structured 
    robotic execution plan based on the Mami schema.
    """
    
    # Call the reasoning model (GPT-4o) with structured output enforcement
    response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {
                "role": "system", 
                "content": (
                    "You are 'Mami', a wise robotic agent. "
                    "Analyze the ENVIRONMENT and the USER REQUEST "
                    "to define a strategy and justify every action."
                )
            },
            *history,  # <--- Questo inserisce i messaggi passati
            {
                "role": "user", 
                "content": (
                    f"ENVIRONMENT: {world_state}\n"
                    f"USER REQUEST: {user_input}"
                )
            },
        ],
        # Enforce the schema defined in api/schema.py
        response_format=MamiPlan,
    )
    
    # Returns the validated object, ready for the robot's hardware abstraction layer
    return response.choices[0].message.parsed