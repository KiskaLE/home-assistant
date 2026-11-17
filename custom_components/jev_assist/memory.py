"""Memory: what the assistant keeps from one conversation to the next.

The LLM saves and removes facts with two tools and finds them in its prompt on
every request. Jev never sees them; it only sends requests about the memory on
to the LLM.
"""

import re
from dataclasses import dataclass

import probatio
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm
from homeassistant.helpers.storage import Store

from .const import DOMAIN, MAX_MEMORIES, MAX_MEMORY_LENGTH

_STORAGE_VERSION = 1

# Part of the API prompt, which Home Assistant does not render as a template
# the way it does the instructions, so a remembered "{{ ... }}" stays text.
_PROMPT = (
    "You have a memory that lasts between conversations. Save a short fact"
    " with the Remember tool when the user asks you to remember something, or"
    " tells you something lasting about themselves, their household or their"
    " preferences that will help later. Do not save one-off requests, the"
    " state of devices or anything you already remember. When the user asks"
    " you to forget something, or a fact is no longer true, remove it with the"
    " Forget tool; to change a fact, forget the old one and remember the new"
    " one. Write each fact in the user's language.\n"
    "The user cannot see your memory change, so when they ask you to remember"
    " or forget something, confirm it in a few words."
)


def _clean(fact: str) -> str:
    """Return the fact on one line, as it is kept and shown."""
    return " ".join(fact.split())


def _key(fact: str) -> str:
    """Return what tells facts apart: their words, whatever the case or punctuation."""
    return " ".join(re.findall(r"\w+", fact.casefold()))


def _store(hass: HomeAssistant, entry_id: str) -> Store[dict[str, list[str]]]:
    return Store(hass, _STORAGE_VERSION, f"{DOMAIN}.memory.{entry_id}", private=True)


async def async_remove_memory(hass: HomeAssistant, entry_id: str) -> None:
    """Delete everything a config entry remembers."""
    await _store(hass, entry_id).async_remove()


class Memory:
    """The facts a config entry remembers, in Home Assistant's storage."""

    def __init__(self, hass: HomeAssistant, entry_id: str) -> None:
        """Initialize an empty memory; `async_load` fills it."""
        self.hass = hass
        self._store = _store(hass, entry_id)
        self.facts: list[str] = []

    async def async_load(self) -> None:
        """Load the saved facts."""
        data = await self._store.async_load() or {}
        self.facts = [f for f in data.get("facts", []) if isinstance(f, str)]

    async def async_set(self, facts: list[str]) -> None:
        """Replace everything remembered, leaving out blanks and repeats."""
        kept: dict[str, str] = {}
        for fact in facts:
            if (key := _key(fact)) and key not in kept:
                kept[key] = _clean(fact)
        # Set before saving, so calls running at the same time build on it.
        self.facts = list(kept.values())
        await self._store.async_save({"facts": list(self.facts)})

    async def async_remember(self, fact: str) -> None:
        """Add a fact, unless it is already remembered."""
        if not (key := _key(fact)):
            raise HomeAssistantError("The fact is empty")
        if any(_key(f) == key for f in self.facts):
            return
        if len(_clean(fact)) > MAX_MEMORY_LENGTH:
            raise HomeAssistantError(
                f"A fact must be shorter than {MAX_MEMORY_LENGTH} characters"
            )
        if len(self.facts) >= MAX_MEMORIES:
            raise HomeAssistantError(
                "The memory is full. Forget something that no longer matters,"
                " or merge related facts into one, first"
            )
        await self.async_set([*self.facts, fact])

    async def async_forget(self, fact: str) -> None:
        """Remove a fact."""
        key = _key(fact)
        kept = [f for f in self.facts if _key(f) != key]
        if len(kept) == len(self.facts):
            raise HomeAssistantError(
                "Nothing remembered matches. Give the fact word for word as you"
                " remember it"
            )
        await self.async_set(kept)

    def prompt(self) -> str:
        """Return how to use the memory and what is in it, for the LLM."""
        if not self.facts:
            return f"{_PROMPT}\nYou do not remember anything yet."
        return "\n".join(
            [_PROMPT, "What you remember:", *(f"- {f}" for f in self.facts)]
        )

    def llm_api(self, api_ids: list[str]) -> llm.API:
        """Return the Home Assistant APIs chosen for the LLM with the memory added."""
        return _MemoryAPI(
            hass=self.hass,
            id=f"{DOMAIN}.memory",
            name="Jev Assist memory",
            memory=self,
            api_ids=api_ids,
        )


class _RememberTool(llm.Tool):
    name = "Remember"
    description = "Remember a fact in every later conversation."
    parameters = probatio.Schema(
        {
            probatio.Required(
                "fact", description="One short fact that makes sense on its own."
            ): str
        }
    )
    annotations = llm.ToolAnnotations(
        read_only=False, destructive=False, idempotent=True, open_world=False
    )
    integration = DOMAIN

    def __init__(self, memory: Memory) -> None:
        self._memory = memory

    async def async_call(
        self,
        hass: HomeAssistant,
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> llm.ToolResult:
        args = self.parameters(tool_input.tool_args)
        await self._memory.async_remember(args["fact"])
        return llm.ToolResult(data={"success": True})


class _ForgetTool(llm.Tool):
    name = "Forget"
    description = "Forget a fact you remember."
    parameters = probatio.Schema(
        {
            probatio.Required(
                "fact", description="The fact word for word as you remember it."
            ): str
        }
    )
    annotations = llm.ToolAnnotations(
        read_only=False, destructive=True, idempotent=True, open_world=False
    )
    integration = DOMAIN

    def __init__(self, memory: Memory) -> None:
        self._memory = memory

    async def async_call(
        self,
        hass: HomeAssistant,
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> llm.ToolResult:
        args = self.parameters(tool_input.tool_args)
        await self._memory.async_forget(args["fact"])
        return llm.ToolResult(data={"success": True})


@dataclass(kw_only=True)
class _MemoryAPI(llm.API):
    """The Home Assistant APIs chosen for the LLM, plus the memory tools.

    Not built with `llm.MergedAPI`, which would rename every tool.
    """

    memory: Memory
    api_ids: list[str]

    async def async_get_api_instance(
        self, llm_context: llm.LLMContext
    ) -> llm.APIInstance:
        """Return the chosen APIs' tools and prompt with the memory's added."""
        tools: list[llm.Tool] = [_RememberTool(self.memory), _ForgetTool(self.memory)]
        prompt = self.memory.prompt()
        custom_serializer = None
        if self.api_ids:
            chosen = await llm.async_get_api(self.hass, self.api_ids, llm_context)
            tools = [*chosen.tools, *tools]
            prompt = f"{chosen.api_prompt}\n{prompt}"
            custom_serializer = chosen.custom_serializer
        return llm.APIInstance(
            api=self,
            api_prompt=prompt,
            llm_context=llm_context,
            tools=tools,
            custom_serializer=custom_serializer,
        )
