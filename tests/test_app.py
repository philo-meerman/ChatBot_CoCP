"""
Test Suite for the ChatBot_DPC Flask Application

This module contains unit tests for the ChatBot_DPC Flask application using Python's unittest
framework. It verifies the functionality of key endpoints, such as the index page ("/") and the
chat endpoint ("/chat"), ensuring that the application returns the expected HTTP status codes
and content.

Key Tests:
    - test_index: Ensures that a GET request to the index page returns a 200 status code and
      valid HTML content.
    - test_chat: Sends a sample JSON POST request to the chat endpoint and verifies that the
      response contains a valid string response.

Before executing the tests, the module initializes the Retrieval-Augmented Generation (RAG)
model required for processing queries. This setup ensures that the Flask application's endpoints
have access to the necessary resources.

Usage:
    Run the test suite using:
        python -m unittest tests/test_app.py

Dependencies:
    - unittest: Python's built-in testing framework.
    - json: For encoding and decoding JSON data in tests.
    - web.app: Provides the Flask application and the initialize_rag_model function.
"""

import unittest
import json
from web.app import app, initialize_rag_model


class FlaskAppTests(unittest.TestCase):
    """A test suite for the Flask application.

    This class contains test cases for verifying the functionality of the Flask application,
    including the index page and chat endpoint. It uses unittest framework for testing and
    initializes the RAG (Retrieval-Augmented Generation) model once for all tests.

    Attributes:
        client: A Flask test client instance used for making requests to the application.

    Methods:
        setUpClass: Class method to set up test environment before running any tests.
        test_index: Tests the index page endpoint ('/') for correct HTTP response and HTML content.
        test_chat: Tests the chat endpoint ('/chat') for proper handling of chat messages and
        responses.
    """

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
        self.assertIn(
            b"<!DOCTYPE html>", response.data
        )  # Check if HTML content is returned

    def test_chat(self):
        """Test the chat endpoint."""
        test_message = (
            "What are the conditions for systematic observation under Dutch law?"
        )
        response = self.client.post(
            "/chat",
            data=json.dumps({"message": test_message}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        response_json = response.get_json()
        self.assertIn("response", response_json)
        self.assertIsInstance(response_json["response"], str)
        self.assertGreater(len(response_json["response"]), 0)


if __name__ == "__main__":
    unittest.main()
