"""Text-to-speech through OpenRouter."""

import struct
from collections.abc import AsyncGenerator, Mapping
from typing import Any

from homeassistant.components.tts import (
    ATTR_VOICE,
    TextToSpeechEntity,
    TTSAudioRequest,
    TTSAudioResponse,
    Voice,
)
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from . import JevAssistConfigEntry
from .const import (
    CONF_TTS_FORMAT,
    CONF_TTS_MODEL,
    CONF_TTS_SAMPLE_RATE,
    CONF_TTS_VOICE,
    DEFAULT_TTS_FORMAT,
    DEFAULT_TTS_MODEL,
    DEFAULT_TTS_SAMPLE_RATE,
    DEFAULT_TTS_VOICE,
    SUPPORTED_LANGUAGES,
)
from .entity import JevAssistEntity

PARALLEL_UPDATES = 0

# The size a WAV header gives when the length is not known yet.
_UNKNOWN_SIZE = 0xFFFFFFFF


def _wav_header(sample_rate: int) -> bytes:
    """Return the header of a 16-bit mono WAV stream of unknown length."""
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        _UNKNOWN_SIZE,
        b"WAVE",
        b"fmt ",
        16,
        1,  # PCM
        1,  # mono
        sample_rate,
        sample_rate * 2,
        2,
        16,
        b"data",
        _UNKNOWN_SIZE,
    )


async def async_setup_entry(
    hass: HomeAssistant,
    entry: JevAssistConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the TTS entity."""
    async_add_entities([JevTTSEntity(entry)])


class JevTTSEntity(JevAssistEntity, TextToSpeechEntity):
    """Speaks Assist replies with an OpenRouter speech model."""

    _attr_translation_key = "tts"
    _attr_supported_languages = SUPPORTED_LANGUAGES
    # Unused by the models, which detect the language, but required by the base.
    _attr_default_language = "en-US"
    _attr_supported_options = [ATTR_VOICE]  # noqa: RUF012

    def __init__(self, entry: JevAssistConfigEntry) -> None:
        """Initialize the entity."""
        super().__init__(entry, "tts")

    @property
    def default_options(self) -> Mapping[str, Any]:
        """Return the default options."""
        return {ATTR_VOICE: self.entry.options.get(CONF_TTS_VOICE, DEFAULT_TTS_VOICE)}

    @callback
    def async_get_supported_voices(self, language: str) -> list[Voice] | None:
        """Return the voices of the configured model."""
        voices = self.entry.runtime_data.voices
        return [Voice(v, v) for v in voices] or None

    async def async_stream_tts_audio(
        self, request: TTSAudioRequest
    ) -> TTSAudioResponse:
        """Synthesize the message and pass the audio on as it arrives.

        A voice satellite only waits so long for a reply to start playing. If
        the audio comes later than that, the satellite takes the reply for
        finished, opens the microphone for the answer to a question and then
        records the reply itself. Sending the first bytes without waiting for
        the whole reply keeps that delay short however long the reply is.
        """
        voice = request.options.get(ATTR_VOICE) or self.default_options[ATTR_VOICE]
        options = self.entry.options
        model = options.get(CONF_TTS_MODEL, DEFAULT_TTS_MODEL)
        audio_format = options.get(CONF_TTS_FORMAT, DEFAULT_TTS_FORMAT)
        is_pcm = audio_format == "pcm"

        async def audio() -> AsyncGenerator[bytes]:
            message = "".join([part async for part in request.message_gen])
            if is_pcm:
                # Bare samples cannot be played; a WAV header says what they are.
                yield _wav_header(
                    int(options.get(CONF_TTS_SAMPLE_RATE, DEFAULT_TTS_SAMPLE_RATE))
                )
            async for chunk in self.client.speak(model, message, voice, audio_format):
                yield chunk

        return TTSAudioResponse("wav" if is_pcm else "mp3", audio())
