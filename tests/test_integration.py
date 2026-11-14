"""End-to-end tests against a mocked OpenRouter client."""

from homeassistant.components import conversation, stt, tts
from homeassistant.config_entries import SOURCE_USER
from homeassistant.const import CONF_API_KEY
from homeassistant.core import Context
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.helpers import entity_registry as er
from homeassistant.setup import async_setup_component
from pytest_homeassistant_custom_component.common import async_mock_service

from custom_components.jev_assist.api import OpenRouterAuthError, OpenRouterError
from custom_components.jev_assist.config_flow import DEFAULT_OPTIONS
from custom_components.jev_assist.const import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_DECISION_MODEL,
    DEFAULT_STT_MODEL,
    DEFAULT_TTS_MODEL,
    DOMAIN,
)

from .test_decision import answers, choice, noul

AGENT = "conversation.jev_assist"


async def _converse(hass, text, language="en"):
    return await conversation.async_converse(
        hass, text, None, Context(), language=language, agent_id=AGENT
    )


async def test_config_flow(hass, client):
    assert await async_setup_component(hass, "homeassistant", {})
    client.validate_key.side_effect = OpenRouterAuthError("no")
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": SOURCE_USER}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_API_KEY: "bad"}
    )
    assert result["errors"] == {"base": "invalid_auth"}

    client.validate_key.side_effect = None
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_API_KEY: "sk-test"}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "models"
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {**DEFAULT_OPTIONS, "chat_model": "openai/custom", "llm_hass_api": []},
    )
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["data"] == {CONF_API_KEY: "sk-test"}
    assert result["options"]["decision_model"] == DEFAULT_DECISION_MODEL
    assert result["options"]["chat_model"] == "openai/custom"
    assert "llm_hass_api" not in result["options"]


async def test_options_flow(hass, entry):
    result = await hass.config_entries.options.async_init(entry.entry_id)
    assert result["type"] is FlowResultType.FORM
    user_input = {**entry.options, "chat_model": "openai/custom", "tts_voice": "george"}
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], user_input
    )
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert entry.options["chat_model"] == "openai/custom"


async def test_entities_are_created(hass, entry):
    registry = er.async_get(hass)
    domains = {
        e.domain for e in er.async_entries_for_config_entry(registry, entry.entry_id)
    }
    assert domains == {"conversation", "stt", "tts"}


async def test_confident_command_calls_service_without_llm(hass, entry, client):
    hass.states.async_set("light.kitchen", "off", {"friendly_name": "Kitchen light"})
    hass.states.async_set("light.bedroom", "off", {"friendly_name": "Bedroom light"})
    calls = async_mock_service(hass, "light", "turn_on")
    client.decide.return_value = answers()

    result = await _converse(hass, "turn on the kitchen light")

    assert len(calls) == 1
    assert calls[0].data == {"entity_id": ["light.kitchen"]}
    assert result.response.speech["plain"]["speech"] == "Turning on Kitchen light."
    client.chat.assert_not_called()
    model, state, questions = client.decide.call_args.args
    assert model == DEFAULT_DECISION_MODEL
    assert state["request"] == "turn on the kitchen light"
    assert "action_light" in questions


async def test_compound_request_is_split_and_each_part_executed(hass, entry, client):
    hass.states.async_set("light.kitchen", "off", {"friendly_name": "Kitchen light"})
    hass.states.async_set("light.bedroom", "on", {"friendly_name": "Bedroom light"})
    on = async_mock_service(hass, "light", "turn_on")
    off = async_mock_service(hass, "light", "turn_off")
    client.decide.side_effect = [
        answers(compound=noul(0.95)),
        answers(),
        answers(device=choice("light.bedroom"), action_light=choice("turn_off")),
    ]
    client.chat.return_value = {
        "content": '["turn on the kitchen light", "turn off the bedroom light"]'
    }

    result = await _converse(hass, "kitchen light on and bedroom light off")

    assert len(on) == 1 and len(off) == 1
    assert result.response.speech["plain"]["speech"] == (
        "Turning on Kitchen light. Turning off Bedroom light."
    )


async def test_conversation_falls_back_to_llm(hass, entry, client):
    hass.states.async_set("light.kitchen", "off", {"friendly_name": "Kitchen light"})
    hass.states.async_set("light.bedroom", "off", {"friendly_name": "Bedroom light"})
    calls = async_mock_service(hass, "light", "turn_on")
    client.decide.return_value = answers(category=choice("conversation"))
    client.chat.return_value = {"role": "assistant", "content": "Why did the bulb..."}

    result = await _converse(hass, "tell me a joke")

    assert not calls
    assert result.response.speech["plain"]["speech"] == "Why did the bulb..."
    model, messages, tools = client.chat.call_args.args
    assert model == DEFAULT_CHAT_MODEL
    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "tell me a joke"}
    assert tools and all(t["type"] == "function" for t in tools)


async def test_llm_tool_call_round_trip(hass, entry, client):
    hass.states.async_set("light.kitchen", "off", {"friendly_name": "Kitchen light"})
    hass.states.async_set("light.bedroom", "off", {"friendly_name": "Bedroom light"})
    client.decide.return_value = answers(names_device=noul(0.3))
    client.chat.side_effect = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "GetDateTime", "arguments": "{}"},
                }
            ],
        },
        {"role": "assistant", "content": "Done."},
    ]

    result = await _converse(hass, "turn it on")

    assert result.response.speech["plain"]["speech"] == "Done."
    # The mock keeps a reference to the list, which also gets the final reply.
    messages = client.chat.call_args_list[1].args[1]
    assert [m["role"] for m in messages[-3:]] == ["assistant", "tool", "assistant"]
    assert messages[-3]["tool_calls"][0]["function"]["name"] == "GetDateTime"
    assert messages[-2]["tool_call_id"] == "call_1"


async def test_decision_failure_falls_back_to_llm(hass, entry, client):
    hass.states.async_set("light.kitchen", "off", {"friendly_name": "Kitchen light"})
    calls = async_mock_service(hass, "light", "turn_on")
    client.decide.side_effect = OpenRouterError("down")
    client.chat.return_value = {"role": "assistant", "content": "Hello."}

    result = await _converse(hass, "turn on the kitchen light")

    assert not calls
    assert result.response.speech["plain"]["speech"] == "Hello."


async def test_stt_wraps_pcm_in_wav(hass, entry, client):
    client.transcribe.return_value = " rozsviť v kuchyni "
    entity = hass.data[stt.DATA_COMPONENT].get_entity("stt.jev_assist_speech_to_text")
    metadata = stt.SpeechMetadata(
        language="cs-CZ",
        format=stt.AudioFormats.WAV,
        codec=stt.AudioCodecs.PCM,
        bit_rate=stt.AudioBitRates.BITRATE_16,
        sample_rate=stt.AudioSampleRates.SAMPLERATE_16000,
        channel=stt.AudioChannels.CHANNEL_MONO,
    )

    async def stream():
        yield b"\x00\x01" * 100

    result = await entity.async_process_audio_stream(metadata, stream())

    assert result.result is stt.SpeechResultState.SUCCESS
    assert result.text == "rozsviť v kuchyni"
    model, audio, audio_format, language = client.transcribe.call_args.args
    assert (model, audio_format, language) == (DEFAULT_STT_MODEL, "wav", "cs")
    assert audio.startswith(b"RIFF") and audio.endswith(b"\x00\x01" * 100)


async def test_tts_returns_mp3_with_configured_voice(hass, entry, client):
    client.speak.return_value = b"ID3audio"
    entity = hass.data[tts.DATA_COMPONENT].get_entity("tts.jev_assist_text_to_speech")

    assert [v.voice_id for v in entity.async_get_supported_voices("cs-CZ")] == [
        "george",
        "sarah",
    ]
    assert await entity.async_get_tts_audio("Ahoj", "cs-CZ", {}) == ("mp3", b"ID3audio")
    assert client.speak.call_args.args == (
        DEFAULT_TTS_MODEL,
        "Ahoj",
        "sarah",
    )
