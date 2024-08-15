import openai
from config import Config

def set_openai_api_key():
    openai.api_key = Config.OPENAI_API_KEY
