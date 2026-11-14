"""Constants for the Jev Assist integration."""

import logging

DOMAIN = "jev_assist"
LOGGER = logging.getLogger(__package__)

API_BASE = "https://openrouter.ai/api/v1"
# Jev and other decision models are served by OpenRouter's Decisions endpoint.
DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"

CONF_DECISION_MODEL = "decision_model"
CONF_CHAT_MODEL = "chat_model"
CONF_STT_MODEL = "stt_model"
CONF_TTS_MODEL = "tts_model"
CONF_TTS_VOICE = "tts_voice"
CONF_CONFIDENCE = "confidence_threshold"
CONF_WEB_SEARCH = "web_search"

DEFAULT_DECISION_MODEL = "~typesafe/jev-latest"
DEFAULT_CHAT_MODEL = "~anthropic/claude-haiku-latest"
DEFAULT_STT_MODEL = "elevenlabs/scribe-v2"
DEFAULT_TTS_MODEL = "elevenlabs/eleven-flash-v2.5"
DEFAULT_TTS_VOICE = "sarah"
DEFAULT_CONFIDENCE = 0.7
DEFAULT_WEB_SEARCH = True

# Actions that are hard to undo need a stricter threshold before Jev may act
# without the LLM (see TypeSafe's "routing by confidence" pattern).
SENSITIVE_CONFIDENCE = 0.9
# A Noul answer at or above this probability counts as "yes".
NOUL_YES = 0.7

MAX_TOOL_ITERATIONS = 10
# OpenRouter runs this tool on its side: the model decides when to search and
# gets the results back within the same request, so no code here executes it.
WEB_SEARCH_TOOL = {"type": "openrouter:web_search"}
# A Choice question accepts up to 255 options.
MAX_ENTITIES = 250

# The cloud models detect the language themselves; Home Assistant still needs
# an explicit list to offer in the voice pipeline settings.
SUPPORTED_LANGUAGES = [
    "af-ZA", "ar-SA", "bg-BG", "bs-BA", "ca-ES", "cs-CZ", "cy-GB", "da-DK",
    "de-DE", "el-GR", "en-GB", "en-US", "es-ES", "et-EE", "fa-IR", "fi-FI",
    "fr-FR", "gl-ES", "he-IL", "hi-IN", "hr-HR", "hu-HU", "id-ID", "is-IS",
    "it-IT", "ja-JP", "ko-KR", "lt-LT", "lv-LV", "mk-MK", "ms-MY", "nl-NL",
    "no-NO", "pl-PL", "pt-BR", "pt-PT", "ro-RO", "ru-RU", "sk-SK", "sl-SI",
    "sr-RS", "sv-SE", "sw-KE", "ta-IN", "th-TH", "tr-TR", "uk-UA", "ur-PK",
    "vi-VN", "zh-CN",
]  # fmt: skip
