import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

class Config:
    """
    Configuration class to store all the constant values and environment variables
    needed throughout the application.
    """

    # API key for OpenAI services, loaded from environment variables
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # Model configuration for OpenAI's chat model
    CHAT_MODEL = "gpt-4o-mini"

    # Model configuration for OpenAI's Embeddings model
    EMBED_MODEL = "text-embedding-3-small"

    # Maximum number of tokens for generating responses
    MAX_TOKENS = 1024

    # Maximum number of tokens for generating summaries
    SUMMARY_MAX_TOKENS = 150

    # Path to the PDF data file
    DATAPATH = "data/vm1kkye15yy2.pdf"

    # RAG Model Chunk size
    CHUNK_SIZE = 1024

    # RAG Model Chunk overlap
    CHUNK_OVERLAP = 50

    # Number of chunks to return from RAG Model
    TOP_K = 5

    # Minimum cosine similarity to return from RAG Model
    MIN_SIMILARITY = 0.5

    # Default system role message for the assistant
    SYSTEM_ROLE = """
        Je bent een behulpzame assistant voor een politieambtenaar. 
        Je hebt het Nieuwe Wetboek van Strafvordering tot je beschikking, waarmee je de gebruiker van waardevolle informatie kunt voorzien. 
        Antwoord alleen op basis van de gegeven context. 
        Als het antwoord niet kan worden gevonden in de gegeven context, zeg dan dat je het antwoord niet weet.
        Antwoord in het Nederlands.
        """
