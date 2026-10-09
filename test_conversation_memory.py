"""Offline tests for conversation context, using only Python's standard library."""
import io
import json
import unittest
from unittest.mock import patch
from rag_qa import generate_answer, find_relevant_docs

class _FakeResponse:
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False
    def read(self):
        return json.dumps({"choices": [{"message": {"content": "Next question"}}]}).encode()

class ConversationMemoryTests(unittest.TestCase):
    def test_model_prompt_includes_previous_problem_and_answer(self):
        seen = {}
        def fake_urlopen(request):
            seen["body"] = json.loads(request.data)
            return _FakeResponse()
        history = [
            {"role": "user", "content": "My laptop will not turn on"},
            {"role": "assistant", "content": "Is it a Mac or Lenovo?"},
        ]
        with patch("rag_qa.urllib.request.urlopen", fake_urlopen):
            answer = generate_answer("Lenovo", "Power protocol", "https://example.invalid", "model", "dummy", history)
        prompt = seen["body"]["messages"][0]["content"]
        self.assertEqual(answer, "Next question")
        self.assertIn("My laptop will not turn on", prompt)
        self.assertIn("Is it a Mac or Lenovo?", prompt)
        self.assertIn("User's Question/Issue: Lenovo", prompt)

    def test_model_prompt_without_history(self):
        seen = {}
        def fake_urlopen(request):
            seen["body"] = json.loads(request.data)
            return _FakeResponse()
        with patch("rag_qa.urllib.request.urlopen", fake_urlopen):
            generate_answer("Laptop problem", "Power protocol", "https://example.invalid", "model", "dummy")
        self.assertIn("(No previous messages)", seen["body"]["messages"][0]["content"])

if __name__ == "__main__":
    unittest.main()
