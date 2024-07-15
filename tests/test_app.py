# tests/test_app.py
import unittest
from web.app import app


class TestApp(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_index(self):
        result = self.app.get("/")
        self.assertEqual(result.status_code, 200)
        self.assertIn(b"Chatbot", result.data)

    def test_chat(self):
        result = self.app.post("/chat", json={"message": "Hello"})
        self.assertEqual(result.status_code, 200)
        self.assertIn("response", result.json)


if __name__ == "__main__":
    unittest.main()
