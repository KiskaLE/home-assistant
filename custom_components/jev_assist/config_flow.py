"""Config flow for Jev Assist."""

from typing import Any

import probatio
from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlowWithReload,
)
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API, CONF_PROMPT
from homeassistant.core import callback
from homeassistant.helpers import llm
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.selector import (
    BooleanSelector,
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
    CONF_WEB_SEARCH,
    DEFAULT_CHAT_MODEL,
    DEFAULT_CONFIDENCE,
    DEFAULT_DECISION_MODEL,
    DEFAULT_STT_MODEL,
    DEFAULT_TTS_MODEL,
    DEFAULT_TTS_VOICE,
    DEFAULT_WEB_SEARCH,
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
    CONF_WEB_SEARCH: DEFAULT_WEB_SEARCH,
}


class JevAssistConfigFlow(ConfigFlow, domain=DOMAIN):
    """Ask for the OpenRouter API key."""

    VERSION = 1

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
                return self.async_create_entry(
                    title="Jev Assist", data=user_input, options=DEFAULT_OPTIONS
                )
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


def _model_selector(models: list[str], current: str) -> SelectSelector:
    """Dropdown of model ids that still accepts an id typed by hand."""
    return SelectSelector(
        SelectSelectorConfig(
            options=sorted({*models, current}),
            mode=SelectSelectorMode.DROPDOWN,
            custom_value=True,
        )
    )


class JevAssistOptionsFlow(OptionsFlowWithReload):
    """Choose the models, the voice and the assistant behaviour."""

    async def _async_models(self) -> dict[str, list[dict]]:
        """Load the live model catalog; an empty list still allows typing an id."""
        client = OpenRouterClient(
            async_get_clientsession(self.hass), self.config_entry.data[CONF_API_KEY]
        )
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

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Manage the options."""
        if user_input is not None:
            if not user_input.get(CONF_LLM_HASS_API):
                user_input.pop(CONF_LLM_HASS_API, None)
            return self.async_create_entry(data=user_input)

        options = {**DEFAULT_OPTIONS, **self.config_entry.options}
        catalog = await self._async_models()
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
            for api in llm.async_get_apis(self.hass)
        ]

        schema = probatio.Schema(
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
                probatio.Required(CONF_WEB_SEARCH): BooleanSelector(),
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
        return self.async_show_form(
            step_id="init",
            data_schema=self.add_suggested_values_to_schema(schema, options),
        )
