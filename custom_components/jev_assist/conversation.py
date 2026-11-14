"""Conversation agent: Jev decides, code acts, an LLM handles the rest."""

import asyncio
import json
import re
from collections.abc import AsyncGenerator
from typing import Any, Literal

import probatio
from homeassistant.components import conversation
from homeassistant.components.homeassistant.exposed_entities import async_should_expose
from homeassistant.const import CONF_LLM_HASS_API, CONF_PROMPT, MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import (
    area_registry as ar,
)
from homeassistant.helpers import (
    device_registry as dr,
)
from homeassistant.helpers import (
    entity_registry as er,
)
from homeassistant.helpers import (
    llm,
)
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.json import json_dumps

from . import JevAssistConfigEntry
from .api import OpenRouterError
from .const import (
    CONF_CHAT_MODEL,
    CONF_CONFIDENCE,
    CONF_DECISION_MODEL,
    CONF_WEB_SEARCH,
    DEFAULT_CHAT_MODEL,
    DEFAULT_CONFIDENCE,
    DEFAULT_DECISION_MODEL,
    DEFAULT_WEB_SEARCH,
    DOMAIN,
    LOGGER,
    MAX_ENTITIES,
    MAX_TOOL_ITERATIONS,
    WEB_SEARCH_TOOL,
)
from .decision import (
    DOMAIN_ACTIONS,
    EntityInfo,
    Plan,
    build_request,
    describe,
    describe_failure,
    interpret,
)
from .entity import JevAssistEntity

_SPLIT_PROMPT = (
    "Split the user's smart home request into separate commands, one action"
    " each. Every command must be complete on its own and in the user's"
    " language. Reply with only a JSON array of strings."
)

# Appended to the LLM instructions. Home Assistant keeps the microphone open,
# without the wake word, after a reply that ends with a question.
_FOLLOW_UP_PROMPT = (
    "If the request is ambiguous or something you need is missing, such as"
    " which device, which room or what value, do not guess. Ask one short"
    " question and put it at the very end of your reply, ending with a"
    " question mark: the microphone then stays open and the user can answer"
    " without saying the wake word.\n"
    "End a reply with a question only when you need the answer. Never add"
    ' offers of further help such as "Anything else?".'
)
# Appended when web search is on. The reply is spoken, so the links and source
# lists that search results invite have to stay out of it.
_WEB_SEARCH_PROMPT = (
    "You can search the web. Search whenever the answer depends on current or"
    " outside information, such as news, sports results, prices, opening hours"
    " or facts you are not sure about, instead of answering from memory or"
    " saying you cannot look it up. Do not search for anything about the home"
    " and its devices.\n"
    "Your reply is read aloud: give the answer in a sentence or two, without"
    " links, source lists or formatting."
)
# Tools OpenRouter runs itself are named like "openrouter:web_search".
_SERVER_TOOL_PREFIX = "openrouter:"
# A question mark (also the Greek and the full-width one) with nothing after
# it but punctuation, quotes, formatting or emoji.
_ENDS_WITH_QUESTION = re.compile(r"[?;\u037e\uff1f]\W*\Z")


def _expects_answer(text: str | None) -> bool:
    """Return whether a reply ends with a question for the user."""
    return bool(text and _ENDS_WITH_QUESTION.search(text))


def _answers_question(chat_log: conversation.ChatLog) -> bool:
    """Return whether the new message answers a question the assistant asked."""
    # The last entry is the new user message, the one before it the last reply.
    if len(chat_log.content) < 2:
        return False
    previous = chat_log.content[-2]
    return previous.role == "assistant" and _expects_answer(previous.content)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: JevAssistConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the conversation entity."""
    async_add_entities([JevConversationEntity(entry)])


def collect_entities(hass: HomeAssistant) -> list[EntityInfo]:
    """Return the entities Jev may control: exposed to Assist and supported."""
    entities = er.async_get(hass)
    devices = dr.async_get(hass)
    areas = ar.async_get(hass)
    result: list[EntityInfo] = []
    for state in hass.states.async_all(list(DOMAIN_ACTIONS)):
        if not async_should_expose(hass, conversation.DOMAIN, state.entity_id):
            continue
        area_name = None
        aliases: tuple[str, ...] = ()
        if entry := entities.async_get(state.entity_id):
            aliases = tuple(a for a in entry.aliases if isinstance(a, str))
            area_id = entry.area_id
            if not area_id and entry.device_id:
                device = devices.async_get(entry.device_id)
                area_id = device.area_id if device else None
            if area_id and (area := areas.async_get_area(area_id)):
                area_name = area.name
        result.append(
            EntityInfo(state.entity_id, state.name, area_name, state.state, aliases)
        )
    if len(result) > MAX_ENTITIES:
        LOGGER.warning(
            "%s entities are exposed, Jev only sees the first %s",
            len(result),
            MAX_ENTITIES,
        )
        result = sorted(result, key=lambda e: e.entity_id)[:MAX_ENTITIES]
    return result


def _format_tool(tool: llm.Tool, custom_serializer: Any) -> dict[str, Any]:
    schema = probatio.to_openapi(
        tool.parameters, custom_serializer=custom_serializer, openapi_version="3.1.0"
    )
    schema = {k: v for k, v in schema.items() if k not in ("oneOf", "anyOf", "allOf")}
    function: dict[str, Any] = {"name": tool.name, "parameters": schema}
    if tool.description:
        function["description"] = tool.description
    return {"type": "function", "function": function}


def _to_message(content: conversation.Content) -> dict[str, Any] | None:
    """Convert chat log content to a chat completions message."""
    if isinstance(content, conversation.ToolResultContent):
        return {
            "role": "tool",
            "tool_call_id": content.tool_call_id,
            "content": json_dumps(
                {"data": content.result.data, "error": content.result.error}
            ),
        }
    if content.role in ("system", "user"):
        if not content.content:
            return None
        return {"role": content.role, "content": content.content}
    message: dict[str, Any] = {"role": "assistant", "content": content.content}
    if isinstance(content, conversation.AssistantContent) and content.tool_calls:
        message["tool_calls"] = [
            {
                "type": "function",
                "id": call.id,
                "function": {
                    "name": call.tool_name,
                    "arguments": json_dumps(call.tool_args),
                },
            }
            for call in content.tool_calls
        ]
    return message


async def _to_delta(
    message: dict[str, Any], can_call_tools: bool
) -> AsyncGenerator[conversation.AssistantContentDeltaDict]:
    """Convert a chat completions message to a chat log delta.

    Only calls Home Assistant can run are kept. OpenRouter has already run its
    own tools, such as web search, by the time it replies.
    """
    delta: conversation.AssistantContentDeltaDict = {
        "role": "assistant",
        "content": message.get("content"),
    }
    tool_calls = []
    for call in message.get("tool_calls") or []:
        if (
            not can_call_tools
            or call.get("type") != "function"
            or call["function"]["name"].startswith(_SERVER_TOOL_PREFIX)
        ):
            LOGGER.debug("Ignoring a tool call Home Assistant does not run: %s", call)
            continue
        try:
            arguments = json.loads(call["function"].get("arguments") or "{}")
        except ValueError as err:
            raise HomeAssistantError(f"Invalid tool arguments: {err}") from err
        tool_calls.append(
            llm.ToolInput(
                id=call["id"], tool_name=call["function"]["name"], tool_args=arguments
            )
        )
    if tool_calls:
        delta["tool_calls"] = tool_calls
    yield delta


class JevConversationEntity(JevAssistEntity, conversation.ConversationEntity):
    """Conversation agent that routes each request with Jev."""

    _attr_name = None
    _attr_supported_features = conversation.ConversationEntityFeature.CONTROL

    def __init__(self, entry: JevAssistConfigEntry) -> None:
        """Initialize the agent."""
        super().__init__(entry, "conversation")

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Route the request: act directly when Jev is sure, else ask the LLM."""
        # An answer to a question only makes sense together with the question.
        # Jev judges a request on its own, so the LLM, which sees the whole
        # conversation, takes the answer.
        plans = (
            None
            if _answers_question(chat_log)
            else await self._async_fast_path(user_input.text)
        )
        if plans is None:
            return await self._async_run_llm(user_input, chat_log)

        replies = []
        for plan in plans:
            try:
                for call in plan.calls:
                    await self.hass.services.async_call(
                        call.domain,
                        call.service,
                        {"entity_id": list(call.entity_ids), **call.data},
                        blocking=True,
                        context=user_input.context,
                    )
            except (HomeAssistantError, probatio.Invalid) as err:
                LOGGER.warning("Service call failed: %s", err)
                replies.append(describe_failure(str(err), user_input.language))
            else:
                replies.append(describe(plan, user_input.language))

        chat_log.async_add_assistant_content_without_tools(
            conversation.AssistantContent(
                agent_id=user_input.agent_id, content=" ".join(replies)
            )
        )
        return conversation.async_get_result_from_chat_log(user_input, chat_log)

    async def _async_decide(
        self, text: str, entities: list[EntityInfo], allow_split: bool
    ) -> Plan:
        """Send one request with every question to Jev and interpret it."""
        options = self.entry.options
        state, questions = build_request(text, entities)
        answers = await self.client.decide(
            options.get(CONF_DECISION_MODEL, DEFAULT_DECISION_MODEL), state, questions
        )
        plan = interpret(
            text,
            entities,
            answers,
            options.get(CONF_CONFIDENCE, DEFAULT_CONFIDENCE),
            allow_split,
        )
        LOGGER.debug("Jev plan for %r: %s (%s)", text, plan.kind, plan.reason)
        return plan

    async def _async_fast_path(self, text: str) -> list[Plan] | None:
        """Return plans to execute without the LLM, or None to escalate."""
        entities = collect_entities(self.hass)
        if not entities:
            return None
        try:
            plan = await self._async_decide(text, entities, allow_split=True)
            if plan.kind == "split":
                parts = await self._async_split(text)
                if len(parts) < 2:
                    plans = [await self._async_decide(text, entities, False)]
                else:
                    plans = await asyncio.gather(
                        *(self._async_decide(part, entities, False) for part in parts)
                    )
            else:
                plans = [plan]
        except OpenRouterError as err:
            # A broken check never turns into an action: let the LLM handle it.
            LOGGER.warning("Decision request failed, using the LLM: %s", err)
            return None
        # Only act when every part is clear, so nothing runs twice if the LLM
        # takes over the whole request.
        if all(p.kind == "execute" for p in plans):
            return list(plans)
        return None

    async def _async_split(self, text: str) -> list[str]:
        """Ask the LLM to split a compound request into atomic commands."""
        message = await self.client.chat(
            self.entry.options.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL),
            [
                {"role": "system", "content": _SPLIT_PROMPT},
                {"role": "user", "content": text},
            ],
        )
        content = message.get("content") or ""
        start, end = content.find("["), content.rfind("]")
        try:
            parts = json.loads(content[start : end + 1])
        except ValueError:
            return []
        if not isinstance(parts, list):
            return []
        return [p.strip() for p in parts if isinstance(p, str) and p.strip()][:5]

    async def _async_run_llm(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Answer with the conversational LLM, with Home Assistant tools and the web."""
        options = self.entry.options
        web_search = options.get(CONF_WEB_SEARCH, DEFAULT_WEB_SEARCH)
        instructions = [
            (options.get(CONF_PROMPT) or llm.DEFAULT_INSTRUCTIONS_PROMPT).rstrip(),
            _FOLLOW_UP_PROMPT,
        ]
        if web_search:
            instructions.append(_WEB_SEARCH_PROMPT)
        try:
            await chat_log.async_provide_llm_data(
                user_input.as_llm_context(DOMAIN),
                options.get(CONF_LLM_HASS_API) or None,
                "\n".join(instructions),
                user_input.extra_system_prompt,
            )
        except conversation.ConverseError as err:
            return err.as_conversation_result()

        tools: list[dict[str, Any]] = []
        if chat_log.llm_api:
            tools = [
                _format_tool(tool, chat_log.llm_api.custom_serializer)
                for tool in chat_log.llm_api.tools
            ]
        if web_search:
            tools.append(WEB_SEARCH_TOOL)
        messages = [m for c in chat_log.content if (m := _to_message(c))]
        model = options.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL)

        for _ in range(MAX_TOOL_ITERATIONS):
            message = await self.client.chat(model, messages, tools)
            messages.extend(
                [
                    m
                    async for content in chat_log.async_add_delta_content_stream(
                        self.entity_id, _to_delta(message, chat_log.llm_api is not None)
                    )
                    if (m := _to_message(content))
                ]
            )
            if not chat_log.unresponded_tool_results:
                break

        result = conversation.async_get_result_from_chat_log(user_input, chat_log)
        # Home Assistant only looks at the very last character; also keep
        # listening when the question is followed by a quote, formatting or emoji.
        result.continue_conversation = _expects_answer(chat_log.content[-1].content)
        return result
