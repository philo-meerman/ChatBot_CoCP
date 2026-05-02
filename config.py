"""
This module contains configuration settings for the ChatBot_DPC application.
"""

import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()


class Config:
    """
    Configuration class to store all the constant values and environment variables
    needed throughout the application.
    """

    # pylint: disable=too-few-public-methods

    # LLM provider selection: "openai" (default) or "nvidia_nim"
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")

    if LLM_PROVIDER == "nvidia_nim":
        OPENAI_API_KEY = os.getenv("NVIDIA_API_KEY") or os.getenv("OPENAI_API_KEY")
        OPENAI_API_BASE = os.getenv(
            "OPENAI_API_BASE", "https://integrate.api.nvidia.com/v1"
        )
        CHAT_MODEL = os.getenv("CHAT_MODEL", "meta/llama-3.1-8b-instruct")
        EMBED_MODEL = os.getenv("EMBED_MODEL", "nvidia/nv-embed-v1")
        # NIM embedding models require input_type to distinguish passages from queries
        EMBED_MODEL_KWARGS = {"input_type": "passage"}
        QUERY_EMBED_MODEL_KWARGS = {"input_type": "query"}
        # Smaller chunks improve retrieval recall on NIM models
        CHUNK_SIZE = 256
        CHUNK_OVERLAP = 25
        # NIM cosine scores are lower than OpenAI's; tuned via benchmark
        MIN_SIMILARITY = 0.30
    else:
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
        CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
        EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
        EMBED_MODEL_KWARGS = {}
        QUERY_EMBED_MODEL_KWARGS = {}
        CHUNK_SIZE = 1024
        CHUNK_OVERLAP = 50
        MIN_SIMILARITY = 0.5

    # Maximum number of tokens for generating responses
    MAX_TOKENS = 1024

    # Maximum number of tokens for generating summaries
    SUMMARY_MAX_TOKENS = 150

    # Path to the PDF data file
    DATAPATH = "data/vm1kkye15yy2.pdf"

    # Number of chunks to return from RAG Model
    TOP_K = 5

    # Default system role message for the assistant
    SYSTEM_ROLE = """
        Je bent een behulpzame assistant voor een politieambtenaar.
        Je hebt het Nieuwe Wetboek van Strafvordering tot je beschikking, waarmee je de gebruiker van waardevolle informatie kunt voorzien.
        Antwoord alleen op basis van de gegeven context.
        Als het antwoord niet kan worden gevonden in de gegeven context, zeg dan dat je het antwoord niet weet.
        Antwoord in het Nederlands.
        """
