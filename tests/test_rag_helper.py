import unittest
from unittest.mock import patch, MagicMock
from utils.answer_generator import generate_answer
from models.rag_model import RAGModel
from dotenv import load_dotenv
import os

class RagHelperTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load environment variables from the .env file
        load_dotenv()
        cls.api_key = os.getenv("OPENAI_API_KEY")
        if not cls.api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file")

    @patch("utils.rag_helper.OpenAI")
    def test_generate_answer(self, mock_openai):
        # Set up a mock RAGModel
        mock_rag_model = MagicMock(spec=RAGModel)
        mock_rag_model.get_relevant_chunks.return_value = [
            "Chunk 1: This is a relevant chunk.",
            "Chunk 2: Another relevant chunk."
        ]

        # Mock the response from the OpenAI API
        mock_response = MagicMock()
        mock_response.message.content = "This is a generated answer."
        mock_openai.return_value.chat.return_value = mock_response

        query = "What are the conditions for systematic observation under Dutch law?"
        response = generate_answer(query, mock_rag_model, self.api_key)

        # Assertions
        self.assertEqual(response, "This is a generated answer.")
        mock_rag_model.get_relevant_chunks.assert_called_once_with(query, k=5)
        mock_openai.assert_called_once()

if __name__ == "__main__":
    unittest.main()
