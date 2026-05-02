"""
API Utility Module

This module provides utilities for configuring and constructing LLM clients for the
ChatBot_DPC project. It supports both the standard OpenAI provider (via LlamaIndex) and
NVIDIA NIM endpoints (via the OpenAI SDK with a custom base URL).

Functions:
    set_openai_api_key(): Sets the global OpenAI API key and optional base URL.
    make_llm(temperature, model, max_tokens): Returns an LLM instance for the configured provider.

Usage:
    >>> from utils.api import set_openai_api_key, make_llm
    >>> set_openai_api_key()
    >>> llm = make_llm(temperature=0, model="gpt-4o-mini", max_tokens=1024)
"""

import openai
from openai import OpenAI as OpenAIClient
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
    if Config.OPENAI_API_BASE:
        openai.base_url = Config.OPENAI_API_BASE


# ---------------------------------------------------------------------------
# NIM-compatible chat wrappers
# LlamaIndex's OpenAI class validates model names against OpenAI's known list,
# which rejects NIM model names. These thin wrappers use the OpenAI SDK directly
# and expose the same .chat(messages) interface that the rest of the code expects.
# ---------------------------------------------------------------------------


class _NIMMessage:  # pylint: disable=too-few-public-methods
    """Mirrors LlamaIndex's ChatMessage.content interface."""

    def __init__(self, content):
        self.content = content


class _NIMChatResponse:  # pylint: disable=too-few-public-methods
    """Mirrors LlamaIndex's ChatResponse interface (.message.content and .raw.usage)."""

    def __init__(self, raw):
        self.raw = raw
        self.message = _NIMMessage(raw.choices[0].message.content)


class _NIMChatClient:  # pylint: disable=too-few-public-methods
    """
    Thin chat wrapper for NVIDIA NIM (and any OpenAI-compatible endpoint).
    Exposes the same .chat(messages) interface as LlamaIndex's OpenAI class.
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(self, api_key, base_url, model, temperature, max_tokens):
        self._client = OpenAIClient(api_key=api_key, base_url=base_url)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    def chat(self, messages):
        """Send a chat request and return a response with .message.content and .raw.usage."""
        messages_dict = [
            {
                "role": msg.role.value if hasattr(msg.role, "value") else str(msg.role),
                "content": msg.content,
            }
            for msg in messages
        ]
        raw = self._client.chat.completions.create(
            model=self._model,
            messages=messages_dict,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        return _NIMChatResponse(raw)


def make_llm(temperature, model, max_tokens):
    """
    Construct an LLM client for the configured provider.

    When OPENAI_API_BASE is set (i.e. NIM or another custom endpoint), returns a
    _NIMChatClient that bypasses LlamaIndex's model-name validation. Otherwise
    returns a standard LlamaIndex OpenAI instance.

    Parameters:
    - temperature (float): Sampling temperature.
    - model (str): Model name.
    - max_tokens (int): Maximum tokens to generate.

    Returns:
    - An LLM client with a .chat(messages) method.
    """
    if Config.OPENAI_API_BASE:
        return _NIMChatClient(
            api_key=Config.OPENAI_API_KEY,
            base_url=Config.OPENAI_API_BASE,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    # Lazy import: LlamaIndex is only needed for the standard OpenAI path.
    # pylint: disable=import-outside-toplevel
    from llama_index.llms.openai import OpenAI

    return OpenAI(
        temperature=temperature,
        model=model,
        max_tokens=max_tokens,
        api_key=Config.OPENAI_API_KEY,
    )
