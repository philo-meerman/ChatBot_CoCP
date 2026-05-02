"""
Module for Summarizing Conversation History

This module provides a function to summarize a conversation history using an OpenAI chat model.
It constructs a summarization prompt from the conversation history, logs the prompt and generated
summary, and returns the summary along with token usage data from the API call.

Dependencies:
    - llama_index.core.llms: Provides the ChatMessage class.
    - llama_index.llms.openai: Provides the OpenAI class for interacting with the OpenAI API.
    - config: Contains configuration parameters such as CHAT_MODEL and SUMMARY_MAX_TOKENS.
    - utils.api: Provides the set_openai_api_key function to configure the OpenAI API key.
    - logging: Used for logging information about the summarization process.

Usage:
    >>> from utils.summarizer import summarize_history
    >>> summary, usage = summarize_history(conversation_history)
"""

import logging
from llama_index.core.llms import ChatMessage
from config import Config
from utils.api import make_llm, set_openai_api_key

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
    - usage (CompletionUsage): The token usage data from the LLM API call.
    """
    # Set the OpenAI API key
    set_openai_api_key()

    # Initialize the chat model (NIM or OpenAI)
    llm = make_llm(
        temperature=0,
        model=Config.CHAT_MODEL,
        max_tokens=Config.SUMMARY_MAX_TOKENS,
    )

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

    # Extract and return the token usage data
    usage = response.raw.usage if hasattr(response.raw, "usage") else None

    return summary, usage
