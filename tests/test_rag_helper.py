# tests/test_rag_helper.py
import unittest
from utils.rag_helper import generate_answer


class TestRAGHelper(unittest.TestCase):
    def test_generate_answer(self):
        query = "De hoofdstad van Nederland is <mask>."
        context = (
            "Nederland is een land in Europa. De hoofdstad van Nederland is Amsterdam."
        )
        response = generate_answer(query, context)
        print(f"Test response: {response}")
        self.assertIn("Amsterdam", response)

    def test_generate_answer_without_mask(self):
        query = "De hoofdstad van Nederland is"
        context = (
            "Nederland is een land in Europa. De hoofdstad van Nederland is Amsterdam."
        )
        response = generate_answer(query, context)
        print(f"Test response: {response}")
        self.assertIn("Amsterdam", response)


if __name__ == "__main__":
    unittest.main()
