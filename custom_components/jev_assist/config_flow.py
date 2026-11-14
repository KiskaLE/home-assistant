"""Config flow for Jev Assist."""

from collections.abc import Mapping
from typing import Any

import probatio
from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlowWithReload,
)
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API, CONF_PROMPT
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import llm
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.selector import (
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TemplateSelector,
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
)

from .api import OpenRouterAuthError, OpenRouterClient, OpenRouterError
from .const import (
    CONF_CHAT_MODEL,
    CONF_CONFIDENCE,
    CONF_DECISION_MODEL,
    CONF_STT_MODEL,
    CONF_TTS_MODEL,
    CONF_TTS_VOICE,
    DEFAULT_CHAT_MODEL,
    DEFAULT_CONFIDENCE,
    DEFAULT_DECISION_MODEL,
    DEFAULT_STT_MODEL,
    DEFAULT_TTS_MODEL,
    DEFAULT_TTS_VOICE,
    DOMAIN,
    LOGGER,
)

DEFAULT_OPTIONS = {
    CONF_DECISION_MODEL: DEFAULT_DECISION_MODEL,
    CONF_CHAT_MODEL: DEFAULT_CHAT_MODEL,
    CONF_STT_MODEL: DEFAULT_STT_MODEL,
    CONF_TTS_MODEL: DEFAULT_TTS_MODEL,
    CONF_TTS_VOICE: DEFAULT_TTS_VOICE,
    CONF_CONFIDENCE: DEFAULT_CONFIDENCE,
    CONF_LLM_HASS_API: [llm.LLM_API_ASSIST],
}


def _model_selector(models: list[str], current: str) -> SelectSelector:
    """Dropdown of model ids that still accepts an id typed by hand."""
    return SelectSelector(
        SelectSelectorConfig(
            options=sorted({*models, current}),
            mode=SelectSelectorMode.DROPDOWN,
            custom_value=True,
        )
    )


async def _async_models(client: OpenRouterClient) -> dict[str, list[dict]]:
    """Load the live model catalog; an empty list still allows typing an id."""
    catalog: dict[str, list[dict]] = {}
    for key, modality in (
        ("decisions", "decisions"),
        ("chat", None),
        ("stt", "transcription"),
        ("tts", "speech"),
    ):
        try:
            catalog[key] = await client.list_models(modality)
        except OpenRouterError as err:
            LOGGER.debug("Could not list %s models: %s", key, err)
            catalog[key] = []
    return catalog


async def _async_settings_schema(
    hass: HomeAssistant, client: OpenRouterClient, options: Mapping[str, Any]
) -> probatio.Schema:
    """Build the form for the models, the voice and the assistant behaviour."""
    catalog = await _async_models(client)
    ids = {key: [m["id"] for m in models] for key, models in catalog.items()}
    # The fallback LLM needs tool calling to control Home Assistant.
    ids["chat"] = [
        m["id"]
        for m in catalog["chat"]
        if "tools" in (m.get("supported_parameters") or [])
    ]
    voices = next(
        (
            m.get("supported_voices") or []
            for m in catalog["tts"]
            if m["id"] == options[CONF_TTS_MODEL]
        ),
        [],
    )
    hass_apis = [
        SelectOptionDict(label=api.name, value=api.id)
        for api in llm.async_get_apis(hass)
    ]
    return probatio.Schema(
        {
            probatio.Required(CONF_DECISION_MODEL): _model_selector(
                ids["decisions"], options[CONF_DECISION_MODEL]
            ),
            probatio.Required(CONF_CONFIDENCE): NumberSelector(
                NumberSelectorConfig(
                    min=0.5, max=0.99, step=0.01, mode=NumberSelectorMode.SLIDER
                )
            ),
            probatio.Required(CONF_CHAT_MODEL): _model_selector(
                ids["chat"], options[CONF_CHAT_MODEL]
            ),
            probatio.Optional(CONF_LLM_HASS_API): SelectSelector(
                SelectSelectorConfig(options=hass_apis, multiple=True)
            ),
            probatio.Optional(CONF_PROMPT): TemplateSelector(),
            probatio.Required(CONF_STT_MODEL): _model_selector(
                ids["stt"], options[CONF_STT_MODEL]
            ),
            probatio.Required(CONF_TTS_MODEL): _model_selector(
                ids["tts"], options[CONF_TTS_MODEL]
            ),
            probatio.Required(CONF_TTS_VOICE): _model_selector(
                voices, options[CONF_TTS_VOICE]
            ),
        }
    )


def _without_empty_llm_api(user_input: dict[str, Any]) -> dict[str, Any]:
    """Drop the Home Assistant API key when no API is selected."""
    if not user_input.get(CONF_LLM_HASS_API):
        user_input.pop(CONF_LLM_HASS_API, None)
    return user_input


class JevAssistConfigFlow(ConfigFlow, domain=DOMAIN):
    """Ask for the OpenRouter API key, then for the models to use."""

    VERSION = 1

    _data: dict[str, Any]
    _client: OpenRouterClient

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: ConfigEntry) -> JevAssistOptionsFlow:
        """Return the options flow."""
        return JevAssistOptionsFlow()

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step."""
        errors: dict[str, str] = {}
        if user_input is not None:
            self._async_abort_entries_match(user_input)
            client = OpenRouterClient(
                async_get_clientsession(self.hass), user_input[CONF_API_KEY]
            )
            try:
                await client.validate_key()
            except OpenRouterAuthError:
                errors["base"] = "invalid_auth"
            except OpenRouterError:
                errors["base"] = "cannot_connect"
            else:
                self._data = user_input
                self._client = client
                return await self.async_step_models()
        return self.async_show_form(
            step_id="user",
            data_schema=probatio.Schema(
                {
                    probatio.Required(CONF_API_KEY): TextSelector(
                        TextSelectorConfig(type=TextSelectorType.PASSWORD)
                    )
                }
            ),
            errors=errors,
        )

    async def async_step_models(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Choose the models, the voice and the assistant behaviour."""
        if user_input is not None:
            return self.async_create_entry(
                title="Jev Assist",
                data=self._data,
                options=_without_empty_llm_api(user_input),
            )
        schema = await _async_settings_schema(self.hass, self._client, DEFAULT_OPTIONS)
        return self.async_show_form(
            step_id="models",
            data_schema=self.add_suggested_values_to_schema(schema, DEFAULT_OPTIONS),
        )


class JevAssistOptionsFlow(OptionsFlowWithReload):
    """Change the models, the voice and the assistant behaviour."""

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Manage the options."""
        if user_input is not None:
            return self.async_create_entry(data=_without_empty_llm_api(user_input))

        options = {**DEFAULT_OPTIONS, **self.config_entry.options}
        client = OpenRouterClient(
            async_get_clientsession(self.hass), self.config_entry.data[CONF_API_KEY]
        )
        schema = await _async_settings_schema(self.hass, client, options)
        return self.async_show_form(
            step_id="init",
            data_schema=self.add_suggested_values_to_schema(schema, options),
        )
