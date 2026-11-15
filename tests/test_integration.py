"""End-to-end tests against a mocked OpenRouter client."""

import asyncio
import io
import sys
import wave
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from homeassistant.components import conversation, stt, tts
from homeassistant.config_entries import SOURCE_USER
from homeassistant.const import CONF_API_KEY
from homeassistant.core import Context
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import chat_session
from homeassistant.helpers import entity_registry as er
from homeassistant.setup import async_setup_component
from pytest_homeassistant_custom_component.common import async_mock_service

from custom_components.jev_assist.api import OpenRouterAuthError, OpenRouterError
from custom_components.jev_assist.const import DOMAIN

from .test_decision import answers, choice, noul

AGENT = "conversation.jev_assist"
TTS = "tts.jev_assist_text_to_speech"


async def _converse(hass, text, language="en", conversation_id=None, device_id=None):
    return await conversation.async_converse(
        hass,
        text,
        conversation_id,
        Context(),
        language=language,
        agent_id=AGENT,
        device_id=device_id,
    )


def _speak(client, speak):
    """Make the mocked client synthesize speech with `speak`."""
    client.speak = MagicMock(side_effect=speak)


@pytest.fixture
def pipelines(hass):
    """Stand in for the voice pipelines, which need packages the tests lack."""
    found = [SimpleNamespace(conversation_engine=AGENT, tts_engine=TTS, tts_voice=None)]
    module = SimpleNamespace(async_get_pipelines=lambda hass: found)
    hass.config.components.add("assist_pipeline")
    with (
        patch.dict(sys.modules, {"homeassistant.components.assist_pipeline": module}),
        patch("homeassistant.components.assist_pipeline", module, create=True),
    ):
        yield found


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
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["options"]["decision_model"] == "~typesafe/jev-latest"


async def test_options_flow(hass, entry):
    result = await hass.config_entries.options.async_init(entry.entry_id)
    assert result["type"] is FlowResultType.FORM
    user_input = {
        **entry.options,
        "chat_model": "openai/custom",
        "tts_voice": "george",
        "tts_format": "pcm",
    }
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], user_input
    )
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert entry.options["chat_model"] == "openai/custom"
    assert entry.options["tts_format"] == "pcm"


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
    # The light going on is the confirmation; nothing is said.
    assert result.response.speech["plain"]["speech"] == ""
    assert not result.continue_conversation
    client.chat.assert_not_called()
    model, state, questions = client.decide.call_args.args
    assert model == "~typesafe/jev-latest"
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
    assert result.response.speech["plain"]["speech"] == ""


async def test_failed_part_of_a_request_is_spoken_with_the_rest(hass, entry, client):
    hass.states.async_set("light.kitchen", "off", {"friendly_name": "Kitchen light"})
    hass.states.async_set("light.bedroom", "on", {"friendly_name": "Bedroom light"})
    async_mock_service(hass, "light", "turn_on")

    async def broken(call):
        raise HomeAssistantError("Bedroom light is unreachable")

    hass.services.async_register("light", "turn_off", broken)
    client.decide.side_effect = [
        answers(compound=noul(0.95)),
        answers(),
        answers(device=choice("light.bedroom"), action_light=choice("turn_off")),
    ]
    client.chat.return_value = {
        "content": '["turn on the kitchen light", "turn off the bedroom light"]'
    }

    result = await _converse(hass, "kitchen light on and bedroom light off")

    assert result.response.speech["plain"]["speech"] == (
        "Turning on Kitchen light."
        " Sorry, that did not work: Bedroom light is unreachable"
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
    assert model == "~anthropic/claude-haiku-latest"
    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "tell me a joke"}
    # Home Assistant's tools plus the search OpenRouter runs on its side.
    assert tools[-1] == {"type": "openrouter:web_search"}
    assert len(tools) > 1 and all(t["type"] == "function" for t in tools[:-1])
    assert "search the web" in messages[0]["content"]


async def test_web_search_can_be_turned_off(hass, entry, client):
    hass.config_entries.async_update_entry(
        entry, options={**entry.options, "web_search": False}
    )
    await hass.async_block_till_done()
    client.chat.return_value = {"role": "assistant", "content": "Hello."}

    await _converse(hass, "hello")

    _, messages, tools = client.chat.call_args.args
    assert tools and all(t["type"] == "function" for t in tools)
    assert "search the web" not in messages[0]["content"]


async def test_web_search_works_without_home_assistant_tools(hass, entry, client):
    options = {k: v for k, v in entry.options.items() if k != "llm_hass_api"}
    hass.config_entries.async_update_entry(entry, options=options)
    await hass.async_block_till_done()
    # OpenRouter may echo the search it already ran; nothing here executes it.
    client.chat.return_value = {
        "role": "assistant",
        "content": "It is sunny in Prague.",
        "tool_calls": [
            {
                "id": "srv_1",
                "type": "function",
                "function": {"name": "openrouter:web_search", "arguments": "{}"},
            }
        ],
    }

    result = await _converse(hass, "what is the weather in Prague")

    assert result.response.speech["plain"]["speech"] == "It is sunny in Prague."
    client.chat.assert_called_once()
    assert client.chat.call_args.args[2] == [{"type": "openrouter:web_search"}]


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


async def test_text_sent_with_a_tool_call_is_not_part_of_the_reply(hass, entry, client):
    client.chat.side_effect = [
        {
            "role": "assistant",
            "content": "Let me check the time.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "GetDateTime", "arguments": "{}"},
                }
            ],
        },
        {"role": "assistant", "content": "It is noon."},
    ]
    spoken = []

    def listener(chat_log, delta):
        spoken.append(delta.get("content"))

    # A voice pipeline reads every piece of assistant text it is sent aloud.
    with (
        chat_session.async_get_chat_session(hass) as session,
        conversation.async_get_chat_log(
            hass, session, chat_log_delta_listener=listener
        ),
    ):
        result = await _converse(
            hass, "what time is it", conversation_id=session.conversation_id
        )

    assert result.response.speech["plain"]["speech"] == "It is noon."
    assert [text for text in spoken if text] == ["It is noon."]


@pytest.mark.parametrize("reply", [".", " . ", None, "", "?"])
async def test_llm_reply_without_words_is_not_spoken(hass, entry, client, reply):
    client.chat.return_value = {"role": "assistant", "content": reply}

    result = await _converse(hass, "set a nice mood in the kitchen")

    assert result.response.speech["plain"]["speech"] == ""
    assert not result.continue_conversation
    assert 'reply with only "."' in client.chat.call_args.args[1][0]["content"]

    # The history sent on the next turn never holds an empty message.
    client.chat.return_value = {"role": "assistant", "content": "It is on."}
    result = await _converse(
        hass, "is the light on", conversation_id=result.conversation_id
    )

    assert result.response.speech["plain"]["speech"] == "It is on."
    messages = client.chat.call_args.args[1]
    assert messages[2] == {"role": "assistant", "content": reply or "."}


async def test_decision_failure_falls_back_to_llm(hass, entry, client):
    hass.states.async_set("light.kitchen", "off", {"friendly_name": "Kitchen light"})
    calls = async_mock_service(hass, "light", "turn_on")
    client.decide.side_effect = OpenRouterError("down")
    client.chat.return_value = {"role": "assistant", "content": "Hello."}

    result = await _converse(hass, "turn on the kitchen light")

    assert not calls
    assert result.response.speech["plain"]["speech"] == "Hello."


async def test_answer_to_a_question_goes_to_the_llm_with_the_question(
    hass, entry, client
):
    hass.states.async_set("light.kitchen", "off", {"friendly_name": "Kitchen light"})
    hass.states.async_set("light.bedroom", "off", {"friendly_name": "Bedroom light"})
    calls = async_mock_service(hass, "light", "turn_on")
    client.decide.return_value = answers(names_device=noul(0.3))
    client.chat.return_value = {"role": "assistant", "content": "Which light?"}

    result = await _converse(hass, "turn on the light")

    # The question keeps the microphone open, so no wake word is needed.
    assert result.continue_conversation
    assert "question mark" in client.chat.call_args.args[1][0]["content"]

    # On its own Jev would read the answer as a clear command.
    client.decide.reset_mock()
    client.decide.return_value = answers()
    client.chat.return_value = {"role": "assistant", "content": "Done."}

    result = await _converse(
        hass, "the kitchen one", conversation_id=result.conversation_id
    )

    client.decide.assert_not_called()
    assert not calls
    assert not result.continue_conversation
    messages = client.chat.call_args.args[1]
    assert [(m["role"], m["content"]) for m in messages[1:4]] == [
        ("user", "turn on the light"),
        ("assistant", "Which light?"),
        ("user", "the kitchen one"),
    ]

    # Nothing is pending any more, so the next command is Jev's again.
    await _converse(
        hass, "turn on the kitchen light", conversation_id=result.conversation_id
    )

    client.decide.assert_called_once()
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("heard", "echo"),
    [
        # The whole question, as a microphone that was open too early hears it.
        ("Which light do you mean, the kitchen or the bedroom one?", True),
        ("which light do you mean the kitchen or the bedroom", True),
        ("light do you mean the kitchen or the bedroom one", True),
        # Answers may well repeat a part of the question.
        ("the kitchen one", False),
        ("the kitchen or the bedroom one, both", False),
        ("I mean the bedroom one", False),
    ],
)
async def test_own_question_played_back_is_not_answered(
    hass, entry, client, heard, echo
):
    question = "Which light do you mean, the kitchen or the bedroom one?"
    client.chat.return_value = {"role": "assistant", "content": question}
    result = await _converse(hass, "turn on the light")
    client.chat.reset_mock()
    client.chat.return_value = {"role": "assistant", "content": "And which one?"}

    result = await _converse(hass, heard, conversation_id=result.conversation_id)

    assert client.chat.called is not echo
    assert result.continue_conversation is not echo
    assert result.response.speech["plain"]["speech"] == (
        "" if echo else "And which one?"
    )


async def test_question_is_synthesized_before_a_satellite_hears_of_it(
    hass, entry, client, pipelines
):
    events = []

    async def speak(model, voice, audio_format, text):
        events.append(("synthesized", text))
        yield b"ID3audio"

    _speak(client, speak)
    client.chat.return_value = {"role": "assistant", "content": "Which light?"}

    def listener(chat_log, delta):
        if delta.get("content"):
            events.append(("announced", delta["content"]))

    with (
        chat_session.async_get_chat_session(hass) as session,
        conversation.async_get_chat_log(
            hass, session, chat_log_delta_listener=listener
        ),
    ):
        await _converse(
            hass,
            "turn on the light",
            conversation_id=session.conversation_id,
            device_id="voice-pe",
        )

    assert events == [("synthesized", "Which light?"), ("announced", "Which light?")]

    # The satellite then gets that audio at once, without a second request.
    async def message():
        yield "Which light?"

    stream = tts.async_create_stream(hass, TTS)
    stream.async_set_message_stream(message())
    assert [chunk async for chunk in stream.async_stream_result()] == [b"ID3audio"]
    assert client.speak.call_count == 1


@pytest.mark.parametrize(
    ("reply", "device_id", "pipeline"),
    [
        # Only a question opens the microphone after it has played.
        ("It is noon.", "voice-pe", {}),
        # A text chat is never spoken.
        ("Which light?", None, {}),
        # Another engine speaks for this satellite.
        ("Which light?", "voice-pe", {"tts_engine": "tts.piper"}),
        ("Which light?", "voice-pe", {"conversation_engine": "conversation.other"}),
    ],
)
async def test_speech_is_only_prepared_when_it_will_be_used(
    hass, entry, client, pipelines, reply, device_id, pipeline
):
    vars(pipelines[0]).update(pipeline)
    _speak(client, None)
    client.chat.return_value = {"role": "assistant", "content": reply}

    await _converse(hass, "hello", device_id=device_id)

    client.speak.assert_not_called()


async def test_question_is_asked_even_when_preparing_the_speech_fails(
    hass, entry, client, pipelines
):
    async def speak(model, voice, audio_format, text):
        raise OpenRouterError("down")
        yield

    _speak(client, speak)
    client.chat.return_value = {"role": "assistant", "content": "Which light?"}

    result = await _converse(hass, "turn on the light", device_id="voice-pe")

    assert result.response.speech["plain"]["speech"] == "Which light?"
    assert result.continue_conversation


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("Which light do you mean?", True),
        ("Did you mean the **kitchen light?**", True),
        ("Kterou myslíte, „kuchyň“ nebo „ložnici“? 🙂", True),
        ("Which one? Tell me the room.", False),
        ("Done.", False),
    ],
)
async def test_only_a_closing_question_keeps_listening(
    hass, entry, client, reply, expected
):
    client.chat.return_value = {"role": "assistant", "content": reply}

    result = await _converse(hass, "hello")

    assert result.continue_conversation is expected


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
    assert (model, audio_format, language) == ("elevenlabs/scribe-v2", "wav", "cs")
    assert audio.startswith(b"RIFF") and audio.endswith(b"\x00\x01" * 100)


async def test_tts_returns_mp3_with_configured_voice(hass, entry, client):
    async def speak(model, voice, audio_format, text):
        yield b"ID3"
        yield b"audio"

    _speak(client, speak)
    entity = hass.data[tts.DATA_COMPONENT].get_entity(TTS)

    async def message():
        yield "Ah"
        yield "oj"

    assert [v.voice_id for v in entity.async_get_supported_voices("cs-CZ")] == [
        "george",
        "sarah",
    ]
    response = await entity.async_stream_tts_audio(
        tts.TTSAudioRequest("cs-CZ", {}, message())
    )
    assert response.extension == "mp3"
    assert [chunk async for chunk in response.data_gen] == [b"ID3", b"audio"]
    assert client.speak.call_args.args == (
        "elevenlabs/eleven-flash-v2.5",
        "sarah",
        "mp3",
        "Ahoj",
    )


async def test_tts_wraps_pcm_in_a_wav_stream(hass, entry, client):
    hass.config_entries.async_update_entry(
        entry,
        options={**entry.options, "tts_format": "pcm", "tts_sample_rate": 16000.0},
    )
    await hass.async_block_till_done()

    async def speak(model, voice, audio_format, text):
        yield b"\x01\x00\x02"
        yield b"\x00"

    _speak(client, speak)
    entity = hass.data[tts.DATA_COMPONENT].get_entity(TTS)

    async def message():
        yield "Ahoj"

    response = await entity.async_stream_tts_audio(
        tts.TTSAudioRequest("cs-CZ", {}, message())
    )
    header, *samples = [chunk async for chunk in response.data_gen]

    assert response.extension == "wav"
    assert client.speak.call_args.args[2] == "pcm"
    assert samples == [b"\x01\x00\x02", b"\x00"]
    # A length that is not known yet is written as the largest possible one,
    # which the standard library takes at its word, so it gets a real one here.
    data = b"".join(samples)
    sized = header[:-4] + len(data).to_bytes(4, "little") + data
    with wave.open(io.BytesIO(sized)) as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (
            1,
            2,
            16000,
        )
        assert wav.readframes(2) == data


async def test_tts_audio_plays_before_the_whole_reply_is_synthesized(
    hass, entry, client
):
    rest = asyncio.Event()

    async def speak(model, voice, audio_format, text):
        yield b"ID3"
        await rest.wait()
        yield b"audio"

    async def message():
        yield "Which light do you mean?"

    _speak(client, speak)
    stream = tts.async_create_stream(hass, TTS)
    # A reply set as a stream is not cached, so nothing is written to disk.
    stream.async_set_message_stream(message())
    audio = stream.async_stream_result()

    # The satellite gets the start of the reply while the rest is on its way.
    async with asyncio.timeout(5):
        assert await anext(audio) == b"ID3"
    rest.set()
    assert [chunk async for chunk in audio] == [b"audio"]
