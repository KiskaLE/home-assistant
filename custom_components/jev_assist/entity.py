"""Base entity for Jev Assist."""

from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.entity import Entity

from . import JevAssistConfigEntry
from .const import DOMAIN


class JevAssistEntity(Entity):
    """Common device info and option access."""

    _attr_has_entity_name = True

    def __init__(self, entry: JevAssistConfigEntry, key: str) -> None:
        """Initialize the entity."""
        self.entry = entry
        self.client = entry.runtime_data.client
        self._attr_unique_id = f"{entry.entry_id}_{key}"
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=entry.title,
            manufacturer="OpenRouter",
            model="Jev Assist",
            entry_type=dr.DeviceEntryType.SERVICE,
        )
