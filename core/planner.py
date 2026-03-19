import os
import logging
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError, AuthenticationError, RateLimitError
from api.schema import MamiPlan

# Load environment variables (API keys, configurations) from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize the OpenAI client
# The API key is securely retrieved from the environment
client = OpenAI(api_key=os.getenv("LLM_API_KEY"))

def generate_mami_plan(user_input: str, world_state: str, history: list = []) -> MamiPlan:
    """
    Translates a natural language instruction into a structured 
    robotic execution plan based on the Mami schema.
    """
    logger.info("Generating Mami plan for input: %s", user_input)
    
    try:
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
        
        plan = response.choices[0].message.parsed
        logger.info("Successfully generated Mami plan")
        return plan

    except AuthenticationError:
        logger.error("OpenAI Authentication Error. Please check your API key.")
        raise
    except RateLimitError:
        logger.error("OpenAI Rate Limit Error. Too many requests.")
        raise
    except OpenAIError as e:
        logger.error(f"OpenAI API Error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in generate_mami_plan: {e}")
        raise