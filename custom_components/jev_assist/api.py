"""Minimal async client for the OpenRouter endpoints this integration uses."""

import base64
import json
from collections.abc import AsyncGenerator
from typing import Any

import aiohttp
from homeassistant.exceptions import HomeAssistantError

from .const import API_BASE, DECISIONS_URL

_HEADERS = {"X-Title": "Home Assistant Jev Assist"}


class OpenRouterError(HomeAssistantError):
    """OpenRouter request failed."""


class OpenRouterAuthError(OpenRouterError):
    """OpenRouter rejected the API key."""


class OpenRouterClient:
    """Talks to OpenRouter: decisions (Jev), chat, speech-to-text, text-to-speech."""

    def __init__(self, session: aiohttp.ClientSession, api_key: str) -> None:
        """Initialize the client."""
        self._session = session
        self._headers = {**_HEADERS, "Authorization": f"Bearer {api_key}"}

    async def _request(
        self, method: str, url: str, payload: Any = None, timeout: float = 60
    ) -> bytes:
        try:
            async with self._session.request(
                method,
                url,
                json=payload,
                headers=self._headers,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as response:
                await self._raise_for_status(response)
                return await response.read()
        except (aiohttp.ClientError, TimeoutError) as err:
            raise OpenRouterError(f"Cannot reach OpenRouter: {err}") from err

    @staticmethod
    async def _raise_for_status(response: aiohttp.ClientResponse) -> None:
        if response.status in (401, 403):
            raise OpenRouterAuthError("OpenRouter rejected the API key")
        if response.status >= 400:
            detail = (await response.read()).decode("utf-8", "replace")[:500]
            raise OpenRouterError(f"OpenRouter {response.status}: {detail}")

    async def _json(
        self, method: str, url: str, payload: Any = None, timeout: float = 60
    ) -> dict[str, Any]:
        body = await self._request(method, url, payload, timeout)
        try:
            data = json.loads(body)
        except ValueError as err:
            raise OpenRouterError("OpenRouter returned invalid JSON") from err
        if not isinstance(data, dict):
            raise OpenRouterError("OpenRouter returned an unexpected response")
        # OpenRouter can report a provider failure inside a 200 response.
        if error := data.get("error"):
            message = error.get("message") if isinstance(error, dict) else error
            raise OpenRouterError(f"OpenRouter error: {message}")
        return data

    async def validate_key(self) -> None:
        """Raise if the API key is not accepted."""
        await self._json("GET", f"{API_BASE}/key", timeout=15)

    async def list_models(self, output_modality: str | None = None) -> list[dict]:
        """Return the model catalog, optionally filtered by output modality."""
        url = f"{API_BASE}/models"
        if output_modality:
            url += f"?output_modalities={output_modality}"
        data = await self._json("GET", url, timeout=15)
        return [m for m in data.get("data", []) if isinstance(m, dict) and "id" in m]

    async def decide(
        self, model: str, state: Any, questions: dict[str, dict]
    ) -> dict[str, dict]:
        """Ask a decision model every question about `state` in one request."""
        data = await self._json(
            "POST",
            DECISIONS_URL,
            {"model": model, "state": state, "questions": questions},
            timeout=20,
        )
        answers = data.get("answers")
        if not isinstance(answers, dict):
            raise OpenRouterError("Decisions response has no answers")
        if missing := [name for name in questions if name not in answers]:
            raise OpenRouterError(f"Decision model did not answer: {missing}")
        return answers

    async def chat(
        self, model: str, messages: list[dict], tools: list[dict] | None = None
    ) -> dict[str, Any]:
        """Run one chat completion and return the assistant message."""
        payload: dict[str, Any] = {"model": model, "messages": messages}
        if tools:
            payload["tools"] = tools
        data = await self._json("POST", f"{API_BASE}/chat/completions", payload, 120)
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as err:
            raise OpenRouterError("OpenRouter returned an empty response") from err
        if not isinstance(message, dict):
            raise OpenRouterError("OpenRouter returned an empty response")
        return message

    async def transcribe(
        self, model: str, audio: bytes, audio_format: str, language: str | None
    ) -> str:
        """Transcribe audio to text."""
        payload: dict[str, Any] = {
            "model": model,
            "input_audio": {
                "data": base64.b64encode(audio).decode("ascii"),
                "format": audio_format,
            },
        }
        if language:
            payload["language"] = language
        data = await self._json("POST", f"{API_BASE}/audio/transcriptions", payload)
        text = data.get("text")
        if not isinstance(text, str):
            raise OpenRouterError("Transcription response has no text")
        return text

    async def speak(
        self, model: str, voice: str, audio_format: str, text: str
    ) -> AsyncGenerator[bytes]:
        """Synthesize `text` and yield the audio, MP3 or raw PCM, as it arrives."""
        payload = {
            "model": model,
            "input": text,
            "voice": voice,
            "response_format": audio_format,
        }
        received = False
        try:
            async with self._session.post(
                f"{API_BASE}/audio/speech",
                json=payload,
                headers=self._headers,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as response:
                await self._raise_for_status(response)
                async for chunk in response.content.iter_any():
                    received = True
                    yield chunk
        except (aiohttp.ClientError, TimeoutError) as err:
            raise OpenRouterError(f"Cannot reach OpenRouter: {err}") from err
        if not received:
            raise OpenRouterError("OpenRouter returned no audio")
