# tests/test_rag_helper.py
import unittest
from utils.rag_helper import generate_answer


class TestRAGHelper(unittest.TestCase):
    def test_generate_answer(self):
        query = "Hello, how are you?"
        context = "The user is starting a conversation."
        response = generate_answer(query, context)
        print(f"Test response: {response}")
        self.assertTrue(isinstance(response, str))
        self.assertNotEqual(response, "")


if __name__ == "__main__":
    unittest.main()
