"""LLM provider adapters for OpenAI, Anthropic, Google Gemini, and HuggingFace."""

from __future__ import annotations

from typing import Protocol

from ouap.config import MODEL_PRICING


class LLMProvider(Protocol):
    """Protocol for LLM providers."""

    def complete(self, messages: list[dict]) -> str: ...
    def count_tokens(self, text: str) -> int: ...

    @property
    def model_name(self) -> str: ...

    @property
    def cost_per_input_token(self) -> float: ...

    @property
    def cost_per_output_token(self) -> float: ...


class OpenAIProvider:
    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None):
        import openai

        self.client = openai.OpenAI(api_key=api_key)
        self._model = model
        self._pricing = MODEL_PRICING.get(model, {"input": 0, "output": 0})

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def cost_per_input_token(self) -> float:
        return self._pricing["input"]

    @property
    def cost_per_output_token(self) -> float:
        return self._pricing["output"]

    def complete(self, messages: list[dict]) -> str:
        response = self.client.chat.completions.create(
            model=self._model, messages=messages, temperature=0.0
        )
        return response.choices[0].message.content or ""

    def count_tokens(self, text: str) -> int:
        import tiktoken

        try:
            enc = tiktoken.encoding_for_model(self._model)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))


class AnthropicProvider:
    def __init__(self, model: str = "claude-sonnet-4-20250514", api_key: str | None = None):
        import anthropic

        self.client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._pricing = MODEL_PRICING.get(model, {"input": 0, "output": 0})

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def cost_per_input_token(self) -> float:
        return self._pricing["input"]

    @property
    def cost_per_output_token(self) -> float:
        return self._pricing["output"]

    def complete(self, messages: list[dict]) -> str:
        system = ""
        user_msgs = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                user_msgs.append(m)
        response = self.client.messages.create(
            model=self._model,
            system=system,
            messages=user_msgs,
            max_tokens=4096,
            temperature=0.0,
        )
        if not response.content:
            return ""
        return response.content[0].text

    def count_tokens(self, text: str) -> int:
        # Approximate: ~4 chars per token for Claude models
        return len(text) // 4


class GoogleProvider:
    """Gemini via Generative Language API using a Google API key."""

    def __init__(self, model: str = "gemini-2.0-flash", api_key: str | None = None):
        import requests as req

        self._model = model
        self._api_key = api_key
        self._pricing = MODEL_PRICING.get(model, {"input": 0, "output": 0})

        if not self._api_key:
            raise ValueError("Google API key is required. Set GOOGLE_API_KEY or pass --api-key.")

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def cost_per_input_token(self) -> float:
        return self._pricing["input"]

    @property
    def cost_per_output_token(self) -> float:
        return self._pricing["output"]

    def complete(self, messages: list[dict]) -> str:
        import requests as req

        system_parts = []
        contents = []
        for m in messages:
            if m["role"] == "system":
                system_parts.append({"text": m["content"]})
            else:
                role = "user" if m["role"] == "user" else "model"
                contents.append({"role": role, "parts": [{"text": m["content"]}]})

        body: dict = {
            "contents": contents,
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 4096},
        }
        if system_parts:
            body["systemInstruction"] = {"parts": system_parts}

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self._model}:generateContent?key={self._api_key}"
        resp = req.post(url, json=body, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            return ""
        parts = candidates[0].get("content", {}).get("parts", [])
        return "".join(p.get("text", "") for p in parts)

    def count_tokens(self, text: str) -> int:
        # Approximate: ~4 chars per token for Gemini models
        return len(text) // 4


class HuggingFaceProvider:
    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct"):
        from transformers import AutoTokenizer, pipeline

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pipe = pipeline(
            "text-generation", model=model_name, tokenizer=self.tokenizer
        )
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def cost_per_input_token(self) -> float:
        return 0.0

    @property
    def cost_per_output_token(self) -> float:
        return 0.0

    def complete(self, messages: list[dict]) -> str:
        output = self.pipe(messages, max_new_tokens=4096, temperature=0.01)
        return output[0]["generated_text"][-1]["content"]

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))


def get_provider(model: str, api_key: str | None = None) -> LLMProvider:
    """Create the appropriate provider for a model name."""
    pricing = MODEL_PRICING.get(model, {})
    provider_type = pricing.get("provider", "")

    if provider_type == "google" or model.startswith("gemini-"):
        return GoogleProvider(model=model, api_key=api_key)
    elif provider_type == "openai" or model.startswith("gpt-"):
        return OpenAIProvider(model=model, api_key=api_key)
    elif provider_type == "anthropic" or model.startswith("claude-"):
        return AnthropicProvider(model=model, api_key=api_key)
    elif provider_type == "huggingface" or model == "local":
        return HuggingFaceProvider(model_name=model if model != "local" else "meta-llama/Llama-3.1-8B-Instruct")
    else:
        # Default: try OpenAI-compatible
        return OpenAIProvider(model=model, api_key=api_key)
