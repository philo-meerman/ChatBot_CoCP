"""
API Utility Module

This module provides a utility function to configure the OpenAI API key for the ChatBot_DPC project.
It retrieves the API key from the project's configuration (Config.OPENAI_API_KEY) and sets it for
the OpenAI library, ensuring that subsequent API calls are properly authenticated.

Functions:
    set_openai_api_key():
        Sets the OpenAI API key for the OpenAI library using the key defined in Config.

Usage:
    >>> from utils.api import set_openai_api_key
    >>> set_openai_api_key()
"""

import openai
from config import Config


def set_openai_api_key():
    """
    Set the OpenAI API key for authentication.

    This function sets the OpenAI API key using the value stored in Config.OPENAI_API_KEY.
    The API key is required for making authenticated requests to OpenAI's services.

    Note:
        Config.OPENAI_API_KEY should be properly configured before calling this function.

    Returns:
        None
    """
    openai.api_key = Config.OPENAI_API_KEY
