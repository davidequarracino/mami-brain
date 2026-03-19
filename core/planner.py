import os
import logging
from dotenv import load_dotenv
from openai import OpenAI
from api.schema import MamiPlan

# Load environment variables (API keys, configurations) from .env file
load_dotenv()

# Initialize the logging module
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Initialize the OpenAI client
# The API key is securely retrieved from the environment
client = OpenAI(api_key=os.getenv("LLM_API_KEY"))


def generate_mami_plan(
    user_input: str, world_state: str, history: list = []
) -> MamiPlan:
    """
    Translates a natural language instruction into a structured
    robotic execution plan based on the Mami schema.
    """

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
                    ),
                },
                *history,  # <--- Questo inserisce i messaggi passati
                {
                    "role": "user",
                    "content": (
                        f"ENVIRONMENT: {world_state}\n" f"USER REQUEST: {user_input}"
                    ),
                },
            ],
            # Enforce the schema defined in api/schema.py
            response_format=MamiPlan,
        )

        # Log the successful response
        logging.info("Received successful response from OpenAI API")

        # Returns the validated object, ready for the robot's hardware abstraction layer
        return response.choices[0].message.parsed

    except openai.AuthenticationError as e:
        # Log the authentication error
        logging.error(f"Authentication error: {e}")
        raise

    except openai.RateLimitError as e:
        # Log the rate limit error
        logging.error(f"Rate limit error: {e}")
        raise

    except Exception as e:
        # Log any other unexpected errors
        logging.error(f"Unexpected error: {e}")
        raise
