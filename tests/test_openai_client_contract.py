"""
Contract Tests for the OpenAI Client Boundary

These tests close the gap that the shipyard review of the openai 1.81 -> 2.54.0 upgrade
could not close: nothing in the existing suite ever executes the response-handling path in
`utils.api`, because the only test that reaches it (`test_app.FlaskAppTests.test_chat`)
dies at the network before the success path runs.

Rather than call a live endpoint, these tests drive the *real* OpenAI SDK against an
in-process `httpx.MockTransport`. The SDK therefore performs its genuine request
serialisation and its genuine response deserialisation -- the two things a major version
bump is most likely to change -- while no network call and no billing occurs.

What is pinned here is the contract the rest of the codebase depends on:

    responses.message.content          -> str   (utils/answer_generator.py:148,
                                                 utils/summarizer.py:61,
                                                 evaluation/run_test_queries_and_compare.py:95)
    hasattr(responses.raw, "usage")    -> True  (utils/answer_generator.py:151,
                                                 utils/summarizer.py:67)
    responses.raw.usage.prompt_tokens      -> int
    responses.raw.usage.completion_tokens  -> int
    responses.raw.usage.total_tokens       -> int

Plus the outbound direction: LlamaIndex `ChatMessage` roles must serialise to plain
strings ("system", "user"), not to their enum repr, or the endpoint rejects the request.

Usage:
    python -m pytest tests/test_openai_client_contract.py -v
"""

import json
import unittest
from unittest.mock import patch

import httpx
from llama_index.core.llms import ChatMessage

from utils.api import _NIMChatClient, _NIMChatResponse, make_llm
from config import Config


# A response body in the shape an OpenAI-compatible endpoint (NVIDIA NIM, or OpenAI
# itself) actually returns. The SDK parses this into its own typed objects, so the
# assertions below test the SDK's real deserialisation rather than a hand-built stub.
CANNED_RESPONSE = {
    "id": "chatcmpl-test-0001",
    "object": "chat.completion",
    "created": 1757100000,
    "model": "meta/llama-3.1-8b-instruct",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Artikel 27 Sv omschrijft de verdachte.",
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 41, "completion_tokens": 12, "total_tokens": 53},
}


class OpenAIClientContractTests(unittest.TestCase):
    """Pins the request and response contract of the OpenAI SDK boundary."""

    def setUp(self):
        """Capture outbound requests and serve a canned response without a network call."""
        self.captured_requests = []

        def handler(request):
            self.captured_requests.append(request)
            return httpx.Response(200, json=CANNED_RESPONSE)

        self.transport = httpx.MockTransport(handler)

    def _build_client(
        self, model="meta/llama-3.1-8b-instruct", temperature=0, max_tokens=1024
    ):
        """Build a _NIMChatClient whose real SDK client speaks to the mock transport."""
        real_openai_client = __import__("openai").OpenAI

        def client_factory(api_key, base_url):
            return real_openai_client(
                api_key=api_key,
                base_url=base_url,
                http_client=httpx.Client(transport=self.transport),
            )

        with patch("utils.api.OpenAIClient", side_effect=client_factory):
            return _NIMChatClient(
                api_key="sk-test-not-a-real-key",
                base_url="https://integrate.api.nvidia.test/v1",
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )

    def _chat(self, messages=None):
        """Run one full round trip through the real SDK."""
        if messages is None:
            messages = [
                ChatMessage(role="system", content="Je bent een juridische assistent."),
                ChatMessage(role="user", content="Wat is een verdachte?"),
            ]
        return self._build_client().chat(messages)

    # -- Response direction -------------------------------------------------

    def test_message_content_is_readable(self):
        """responses.message.content must yield the assistant text as a plain string."""
        response = self._chat()
        self.assertIsInstance(response.message.content, str)
        self.assertEqual(
            response.message.content, "Artikel 27 Sv omschrijft de verdachte."
        )

    def test_raw_exposes_usage(self):
        """Callers guard on hasattr(response.raw, 'usage'); that guard must still pass."""
        response = self._chat()
        self.assertTrue(
            hasattr(response.raw, "usage"),
            "answer_generator and summarizer both skip cost logging when this is False",
        )

    def test_usage_fields_match_caller_expectations(self):
        """The three token fields read in answer_generator.py:153-155 must be present ints."""
        usage = self._chat().raw.usage
        self.assertEqual(usage.prompt_tokens, 41)
        self.assertEqual(usage.completion_tokens, 12)
        self.assertEqual(usage.total_tokens, 53)
        for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
            self.assertIsInstance(getattr(usage, field), int)

    def test_cost_estimation_arithmetic_survives(self):
        """The cost calculation in answer_generator.py must run on a real usage object."""
        usage = self._chat().raw.usage
        estimated_cost = (usage.prompt_tokens * (0.15 / 1000000)) + (
            usage.completion_tokens * (0.60 / 1000000)
        )
        self.assertGreater(estimated_cost, 0)

    def test_wrapper_accepts_a_genuine_sdk_response_object(self):
        """_NIMChatResponse must read the SDK's own parsed object, not just a stub."""
        # Reaching into the SDK client is the point: this asserts the wrapper reads a
        # genuine parsed response rather than a hand-built stub.
        # pylint: disable=protected-access
        raw = self._build_client()._client.chat.completions.create(
            model="meta/llama-3.1-8b-instruct",
            messages=[{"role": "user", "content": "hallo"}],
        )
        wrapped = _NIMChatResponse(raw)
        self.assertEqual(
            wrapped.message.content, "Artikel 27 Sv omschrijft de verdachte."
        )
        self.assertIs(wrapped.raw, raw)

    def test_missing_usage_is_none_and_hasattr_does_not_guard(self):
        """An endpoint may omit usage; the SDK then sets it to None rather than dropping it.

        This is why utils/answer_generator.py guards with `is not None` and not with
        `hasattr`: the attribute is always defined, so the hasattr form is always True
        and would raise AttributeError on the None value one line later.
        """

        def handler(request):  # pylint: disable=unused-argument
            body = dict(CANNED_RESPONSE)
            body.pop("usage")
            return httpx.Response(200, json=body)

        self.transport = httpx.MockTransport(handler)
        response = self._chat()

        self.assertTrue(
            hasattr(response.raw, "usage"), "attribute is defined even when absent"
        )
        self.assertIsNone(response.raw.usage)
        with self.assertRaises(AttributeError):
            _ = response.raw.usage.prompt_tokens

    # -- Request direction --------------------------------------------------

    def test_chatmessage_roles_serialise_to_plain_strings(self):
        """msg.role.value must produce 'system'/'user', never a MessageRole enum repr."""
        self._chat()
        sent = json.loads(self.captured_requests[0].content)
        self.assertEqual([m["role"] for m in sent["messages"]], ["system", "user"])
        for message in sent["messages"]:
            self.assertNotIn("MessageRole", message["role"])

    def test_request_carries_model_and_sampling_parameters(self):
        """Temperature, max_tokens and model must survive serialisation by the SDK."""
        self._build_client(model="gpt-4o-mini", temperature=0.3, max_tokens=512).chat(
            [ChatMessage(role="user", content="test")]
        )
        sent = json.loads(self.captured_requests[0].content)
        self.assertEqual(sent["model"], "gpt-4o-mini")
        self.assertEqual(sent["temperature"], 0.3)
        self.assertEqual(sent["max_tokens"], 512)

    def test_message_content_survives_serialisation(self):
        """The prompt text itself must reach the endpoint unmodified."""
        self._chat([ChatMessage(role="user", content="Wat is artikel 27 Sv?")])
        sent = json.loads(self.captured_requests[0].content)
        self.assertEqual(sent["messages"][0]["content"], "Wat is artikel 27 Sv?")

    # -- Provider routing ---------------------------------------------------

    def test_make_llm_routes_to_nim_client_when_base_url_is_set(self):
        """With OPENAI_API_BASE set, make_llm must bypass LlamaIndex model validation."""
        with (
            patch.object(
                Config, "OPENAI_API_BASE", "https://integrate.api.nvidia.test/v1"
            ),
            patch.object(Config, "OPENAI_API_KEY", "sk-test-not-a-real-key"),
        ):
            llm = make_llm(
                temperature=0, model="meta/llama-3.1-8b-instruct", max_tokens=1024
            )
        self.assertIsInstance(llm, _NIMChatClient)


if __name__ == "__main__":
    unittest.main()
