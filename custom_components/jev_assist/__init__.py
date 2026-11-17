"""Jev Assist: a cloud voice assistant for Home Assistant built on OpenRouter."""

from dataclasses import dataclass, field

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_KEY, Platform
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryError, ConfigEntryNotReady
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .api import OpenRouterAuthError, OpenRouterClient, OpenRouterError
from .const import CONF_TTS_MODEL, DEFAULT_TTS_MODEL, LOGGER
from .memory import Memory, async_remove_memory

PLATFORMS = [Platform.CONVERSATION, Platform.STT, Platform.TTS]


@dataclass
class JevAssistData:
    """Runtime data shared by the platforms."""

    client: OpenRouterClient
    # Loaded even when the memory is off, so the options can show what is in it.
    memory: Memory
    voices: list[str] = field(default_factory=list)


type JevAssistConfigEntry = ConfigEntry[JevAssistData]


async def async_setup_entry(hass: HomeAssistant, entry: JevAssistConfigEntry) -> bool:
    """Set up Jev Assist from a config entry."""
    client = OpenRouterClient(async_get_clientsession(hass), entry.data[CONF_API_KEY])
    try:
        await client.validate_key()
    except OpenRouterAuthError as err:
        raise ConfigEntryError("Invalid OpenRouter API key") from err
    except OpenRouterError as err:
        raise ConfigEntryNotReady(str(err)) from err

    memory = Memory(hass, entry.entry_id)
    await memory.async_load()
    data = JevAssistData(client, memory)
    # The voice list only feeds the voice picker, so a failure is not fatal.
    tts_model = entry.options.get(CONF_TTS_MODEL, DEFAULT_TTS_MODEL)
    try:
        for model in await client.list_models("speech"):
            if model["id"] == tts_model:
                data.voices = [
                    v for v in model.get("supported_voices") or [] if isinstance(v, str)
                ]
    except OpenRouterError as err:
        LOGGER.debug("Could not load the voice list: %s", err)

    entry.runtime_data = data
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True


async def async_unload_entry(hass: HomeAssistant, entry: JevAssistConfigEntry) -> bool:
    """Unload a config entry."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)


async def async_remove_entry(hass: HomeAssistant, entry: JevAssistConfigEntry) -> None:
    """Delete what the assistant remembers along with the integration."""
    await async_remove_memory(hass, entry.entry_id)
