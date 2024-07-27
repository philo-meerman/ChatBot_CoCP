import logging
from llama_index.core.llms import ChatMessage
from llama_index.llms.openai import OpenAI
from config import Config
from utils.api import set_openai_api_key

# Set up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def summarize_history(history):
    """
    Summarize the conversation history to retain relevant context.

    Parameters:
    - history (list): The conversation history.

    Returns:
    - summary (str): The summarized conversation history.
    """
    # Set the OpenAI API key
    set_openai_api_key()

    # Initialize the OpenAI chat model for summarization
    llm = OpenAI(temperature=0, model=Config.CHAT_MODEL, max_tokens=Config.SUMMARY_MAX_TOKENS)

    # Prepare the summarization prompt
    prompt = "Vat de volgende gespreksgeschiedenis samen om de relevante context te behouden:\n\n"
    for msg in history:
        prompt += f"{msg.role}: {msg.content}\n"

    # Log the summarization prompt
    logger.info("Summarization prompt:\n%s", prompt)

    # Get the summary from the chat model
    response = llm.chat([ChatMessage(role="system", content=prompt)])
    summary = response.message.content

    # Log the generated summary
    logger.info("Generated summary:\n%s", summary)

    return summary
