# tests/test_pdf_processor.py
import unittest
from utils.pdf_processor import extract_text_from_pdf


class TestPDFProcessor(unittest.TestCase):
    def test_extract_text_from_pdf(self):
        text = extract_text_from_pdf("data/sample.pdf")
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)


if __name__ == "__main__":
    unittest.main()
