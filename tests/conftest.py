"""Fixtures for Jev Assist tests."""

from unittest.mock import AsyncMock, patch

import pytest
from homeassistant.const import CONF_API_KEY
from homeassistant.setup import async_setup_component
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.jev_assist.config_flow import DEFAULT_OPTIONS
from custom_components.jev_assist.const import DEFAULT_TTS_MODEL, DOMAIN


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(enable_custom_integrations):
    """Load the custom integration in every test."""


@pytest.fixture
def client():
    """Mock the OpenRouter client everywhere it is constructed."""
    mock = AsyncMock()
    mock.list_models.return_value = [
        {
            "id": DEFAULT_TTS_MODEL,
            "supported_voices": ["george", "sarah"],
            "supported_parameters": ["tools"],
        }
    ]
    with (
        patch("custom_components.jev_assist.OpenRouterClient", return_value=mock),
        patch(
            "custom_components.jev_assist.config_flow.OpenRouterClient",
            return_value=mock,
        ),
    ):
        yield mock


@pytest.fixture
async def entry(hass, client):
    """Set up the integration."""
    assert await async_setup_component(hass, "homeassistant", {})
    config_entry = MockConfigEntry(
        domain=DOMAIN,
        title="Jev Assist",
        data={CONF_API_KEY: "sk-test"},
        options=DEFAULT_OPTIONS,
    )
    config_entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(config_entry.entry_id)
    await hass.async_block_till_done()
    return config_entry
