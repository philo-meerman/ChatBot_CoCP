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
    CHAT_MODEL = "gpt-3.5-turbo"
    
    # Maximum number of tokens for generating responses
    MAX_TOKENS = 1024
    
    # Maximum number of tokens for generating summaries
    SUMMARY_MAX_TOKENS = 150
    
    # Path to the PDF data file
    DATAPATH = "data/vm1kkye15yy2.pdf"
    
    # Flag to indicate whether embeddings need to be regenerated (0 for no, 1 for yes)
    EMBED_REGEN = 0
    
    # Default system role message for the assistant
    SYSTEM_ROLE = """
        Je bent een behulpzame assistant voor een politieambtenaar. 
        Je hebt het Nieuwe Wetboek van Strafvordering tot je beschikking, 
        waarmee je de gebruiker van waardevolle informatie kunt voorzien. 
        Antwoord in het Nederlands.
        """
