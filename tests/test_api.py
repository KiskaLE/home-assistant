"""Tests of the OpenRouter client against a local server."""

import asyncio
from unittest.mock import patch

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from custom_components.jev_assist.api import (
    OpenRouterAuthError,
    OpenRouterClient,
    OpenRouterError,
)


@pytest.fixture
async def speech(socket_enabled):
    """Serve /audio/speech with a handler the test sets, and return a client."""
    handlers = {}

    async def handle(request):
        return await handlers["speech"](request)

    app = web.Application()
    app.router.add_post("/audio/speech", handle)
    server = TestServer(app)
    await server.start_server()
    base = str(server.make_url("")).rstrip("/")
    async with aiohttp.ClientSession() as session:
        with patch("custom_components.jev_assist.api.API_BASE", base):
            yield handlers, OpenRouterClient(session, "sk-test")
    await server.close()


async def test_speak_yields_audio_while_it_is_still_being_sent(speech):
    handlers, client = speech
    rest = asyncio.Event()
    requests = []

    async def handler(request):
        requests.append((request.headers["Authorization"], await request.json()))
        response = web.StreamResponse()
        await response.prepare(request)
        await response.write(b"ID3")
        await rest.wait()
        await response.write(b"audio")
        return response

    handlers["speech"] = handler
    audio = client.speak("elevenlabs/eleven-flash-v2.5", "Ahoj", "sarah", "pcm")

    async with asyncio.timeout(5):
        assert await anext(audio) == b"ID3"
    rest.set()
    assert b"".join([chunk async for chunk in audio]) == b"audio"
    assert requests == [
        (
            "Bearer sk-test",
            {
                "model": "elevenlabs/eleven-flash-v2.5",
                "input": "Ahoj",
                "voice": "sarah",
                "response_format": "pcm",
            },
        )
    ]


@pytest.mark.parametrize(
    ("status", "body", "error", "message"),
    [
        (200, b"", OpenRouterError, "no audio"),
        (401, b"nope", OpenRouterAuthError, "rejected the API key"),
        (502, b"provider is down", OpenRouterError, "502: provider is down"),
    ],
)
async def test_speak_reports_a_failed_request(speech, status, body, error, message):
    handlers, client = speech

    async def handler(request):
        return web.Response(status=status, body=body)

    handlers["speech"] = handler

    with pytest.raises(error, match=message):
        [chunk async for chunk in client.speak("model", "Ahoj", "sarah")]
