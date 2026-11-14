"""Speech-to-text through OpenRouter."""

import io
import wave
from collections.abc import AsyncIterable

from homeassistant.components import stt
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from . import JevAssistConfigEntry
from .api import OpenRouterError
from .const import CONF_STT_MODEL, DEFAULT_STT_MODEL, LOGGER, SUPPORTED_LANGUAGES
from .entity import JevAssistEntity

PARALLEL_UPDATES = 0


async def async_setup_entry(
    hass: HomeAssistant,
    entry: JevAssistConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the STT entity."""
    async_add_entities([JevSTTEntity(entry)])


class JevSTTEntity(JevAssistEntity, stt.SpeechToTextEntity):
    """Transcribes Assist audio with an OpenRouter transcription model."""

    _attr_translation_key = "stt"

    def __init__(self, entry: JevAssistConfigEntry) -> None:
        """Initialize the entity."""
        super().__init__(entry, "stt")

    @property
    def supported_languages(self) -> list[str]:
        """Return a list of supported languages."""
        return SUPPORTED_LANGUAGES

    @property
    def supported_formats(self) -> list[stt.AudioFormats]:
        """Return a list of supported formats."""
        return [stt.AudioFormats.WAV]

    @property
    def supported_codecs(self) -> list[stt.AudioCodecs]:
        """Return a list of supported codecs."""
        return [stt.AudioCodecs.PCM]

    @property
    def supported_bit_rates(self) -> list[stt.AudioBitRates]:
        """Return a list of supported bit rates."""
        return [stt.AudioBitRates.BITRATE_16]

    @property
    def supported_sample_rates(self) -> list[stt.AudioSampleRates]:
        """Return a list of supported sample rates."""
        return [stt.AudioSampleRates.SAMPLERATE_16000]

    @property
    def supported_channels(self) -> list[stt.AudioChannels]:
        """Return a list of supported channels."""
        return [stt.AudioChannels.CHANNEL_MONO]

    async def async_process_audio_stream(
        self, metadata: stt.SpeechMetadata, stream: AsyncIterable[bytes]
    ) -> stt.SpeechResult:
        """Send the recording to OpenRouter and return the transcript."""
        pcm = bytearray()
        async for chunk in stream:
            pcm.extend(chunk)

        # Assist streams raw PCM; the API wants a complete WAV file.
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(metadata.channel.value)
            wav.setsampwidth(metadata.bit_rate.value // 8)
            wav.setframerate(metadata.sample_rate.value)
            wav.writeframes(bytes(pcm))

        try:
            text = await self.client.transcribe(
                self.entry.options.get(CONF_STT_MODEL, DEFAULT_STT_MODEL),
                buffer.getvalue(),
                "wav",
                metadata.language.split("-")[0],
            )
        except OpenRouterError as err:
            LOGGER.error("Speech-to-text failed: %s", err)
            return stt.SpeechResult(None, stt.SpeechResultState.ERROR)
        return stt.SpeechResult(text.strip(), stt.SpeechResultState.SUCCESS)
