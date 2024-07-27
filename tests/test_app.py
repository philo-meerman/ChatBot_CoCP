import unittest
import json
from flask import Flask
from web.app import app, initialize_rag_model

class FlaskAppTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Initialize the RAG model once for all tests
        initialize_rag_model()
        cls.client = app.test_client()
        cls.client.testing = True

    def test_index(self):
        """Test the index page."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'<!DOCTYPE html>', response.data)  # Check if HTML content is returned

    def test_chat(self):
        """Test the chat endpoint."""
        test_message = "What are the conditions for systematic observation under Dutch law?"
        response = self.client.post("/chat", 
                                    data=json.dumps({"message": test_message}),
                                    content_type='application/json')
        self.assertEqual(response.status_code, 200)
        response_json = response.get_json()
        self.assertIn("response", response_json)
        self.assertIsInstance(response_json["response"], str)
        self.assertGreater(len(response_json["response"]), 0)

if __name__ == "__main__":
    unittest.main()
