"""Text-to-speech through OpenRouter."""

from collections.abc import Mapping
from typing import Any

from homeassistant.components.tts import (
    ATTR_VOICE,
    TextToSpeechEntity,
    TtsAudioType,
    Voice,
)
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from . import JevAssistConfigEntry
from .const import (
    CONF_TTS_MODEL,
    CONF_TTS_VOICE,
    DEFAULT_TTS_MODEL,
    DEFAULT_TTS_VOICE,
    SUPPORTED_LANGUAGES,
)
from .entity import JevAssistEntity

PARALLEL_UPDATES = 0


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

    async def async_get_tts_audio(
        self, message: str, language: str, options: dict[str, Any]
    ) -> TtsAudioType:
        """Synthesize the message and return MP3 audio."""
        voice = options.get(ATTR_VOICE) or self.default_options[ATTR_VOICE]
        audio = await self.client.speak(
            self.entry.options.get(CONF_TTS_MODEL, DEFAULT_TTS_MODEL), message, voice
        )
        return "mp3", audio
