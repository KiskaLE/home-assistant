"""Tests for the memory the assistant keeps between conversations."""

import json

import pytest
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import intent
from pytest_homeassistant_custom_component.common import async_mock_service

from custom_components.jev_assist.const import MAX_MEMORIES

from .test_decision import answers, choice, noul
from .test_integration import _converse


def _key(entry):
    return f"jev_assist.memory.{entry.entry_id}"


def _tool_call(name, fact):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps({"fact": fact})},
            }
        ],
    }


def _tool_names(client):
    return [
        t["function"]["name"] if t["type"] == "function" else t["type"]
        for t in client.chat.call_args.args[2]
    ]


def _system_prompt(client):
    return client.chat.call_args.args[1][0]["content"]


@pytest.fixture
async def memory(hass, entry):
    """Turn the memory on and return it."""
    hass.config_entries.async_update_entry(
        entry, options={**entry.options, "memory": True}
    )
    await hass.async_block_till_done()
    return entry.runtime_data.memory


async def test_memory_is_off_by_default(hass, entry, client):
    hass.states.async_set("light.kitchen", "off", {"friendly_name": "Kitchen light"})
    client.decide.return_value = answers(category=choice("conversation"))
    client.chat.return_value = {"role": "assistant", "content": "Hello."}

    await _converse(hass, "hello")

    assert "Remember" not in _tool_names(client)
    assert "Remember tool" not in _system_prompt(client)
    assert "memory" not in client.decide.call_args.args[2]


async def test_assistant_remembers_between_conversations(
    hass, entry, memory, client, hass_storage
):
    client.chat.side_effect = [
        _tool_call("Remember", "Likes the bedroom at 22 degrees."),
        {"role": "assistant", "content": "Got it."},
    ]

    result = await _converse(hass, "remember I like the bedroom at 22 degrees")

    assert result.response.speech["plain"]["speech"] == "Got it."
    assert {"Remember", "Forget"} <= set(_tool_names(client))
    assert "You do not remember anything yet." in _system_prompt(client)
    assert memory.facts == ["Likes the bedroom at 22 degrees."]
    assert hass_storage[_key(entry)]["data"] == {
        "facts": ["Likes the bedroom at 22 degrees."]
    }

    # A new conversation knows nothing of the first one, except the fact.
    client.chat.side_effect = None
    client.chat.return_value = {"role": "assistant", "content": "At 22 degrees."}
    await _converse(hass, "how warm do I like the bedroom")

    messages = client.chat.call_args.args[1]
    assert [m["content"] for m in messages if m["role"] == "user"] == [
        "how warm do I like the bedroom"
    ]
    assert "What you remember:\n- Likes the bedroom at 22 degrees." in _system_prompt(
        client
    )


async def test_assistant_forgets_a_fact(hass, memory, client):
    await memory.async_set(
        ["Likes the bedroom at 22 degrees.", "Has a cat called Mia."]
    )
    client.chat.side_effect = [
        # Close enough: the case and the full stop do not matter.
        _tool_call("Forget", "has a cat called mia"),
        {"role": "assistant", "content": "Forgotten."},
    ]

    await _converse(hass, "forget about the cat")

    assert memory.facts == ["Likes the bedroom at 22 degrees."]


async def test_forgetting_what_is_not_remembered_is_reported_to_the_llm(
    hass, memory, client
):
    await memory.async_set(["Has a cat called Mia."])
    client.chat.side_effect = [
        _tool_call("Forget", "Has a dog."),
        {"role": "assistant", "content": "I did not remember a dog."},
    ]

    result = await _converse(hass, "forget about the dog")

    assert result.response.speech["plain"]["speech"] == "I did not remember a dog."
    assert memory.facts == ["Has a cat called Mia."]
    # The mock keeps a reference to the list, which also gets the final reply.
    tool_result = client.chat.call_args.args[1][-2]
    assert tool_result["role"] == "tool"
    assert "Nothing remembered matches" in tool_result["content"]


async def test_memory_works_without_home_assistant_tools(hass, entry, client):
    options = {k: v for k, v in entry.options.items() if k != "llm_hass_api"}
    hass.config_entries.async_update_entry(entry, options={**options, "memory": True})
    await hass.async_block_till_done()
    client.chat.side_effect = [
        _tool_call("Remember", "Has a cat called Mia."),
        {"role": "assistant", "content": "Got it."},
    ]

    await _converse(hass, "remember that my cat is called Mia")

    assert _tool_names(client) == ["Remember", "Forget", "openrouter:web_search"]
    assert entry.runtime_data.memory.facts == ["Has a cat called Mia."]


async def test_remembered_text_is_not_rendered_as_a_template(hass, memory, client):
    await memory.async_set(["Says {{ 1 + 1 }} a lot."])
    client.chat.return_value = {"role": "assistant", "content": "Hello."}

    await _converse(hass, "hello")

    assert "- Says {{ 1 + 1 }} a lot." in _system_prompt(client)


async def test_request_to_remember_a_command_is_not_carried_out(hass, memory, client):
    hass.states.async_set("light.kitchen", "off", {"friendly_name": "Kitchen light"})
    hass.states.async_set("light.bedroom", "off", {"friendly_name": "Bedroom light"})
    calls = async_mock_service(hass, "light", "turn_on")
    # Otherwise a clear command to turn on the kitchen light.
    client.decide.return_value = answers(memory=noul(0.95))
    client.chat.return_value = {"role": "assistant", "content": "Noted."}

    await _converse(hass, "remember that good morning means the kitchen light on")

    assert not calls
    assert "memory" in client.decide.call_args.args[2]
    client.chat.assert_called_once()


async def test_missing_home_assistant_api_is_an_error_reply(hass, entry, client):
    hass.config_entries.async_update_entry(
        entry, options={**entry.options, "memory": True, "llm_hass_api": ["gone"]}
    )
    await hass.async_block_till_done()

    result = await _converse(hass, "hello")

    assert result.response.response_type is intent.IntentResponseType.ERROR
    client.chat.assert_not_called()


async def test_facts_stay_short_and_are_not_repeated(hass, entry):
    memory = entry.runtime_data.memory

    await memory.async_remember("Has a cat  called\nMia.")
    await memory.async_remember("has a cat called mia")

    assert memory.facts == ["Has a cat called Mia."]
    with pytest.raises(HomeAssistantError, match="shorter"):
        await memory.async_remember("x" * 301)
    with pytest.raises(HomeAssistantError, match="empty"):
        await memory.async_remember(" . ")
    await memory.async_set([f"Fact {i}" for i in range(MAX_MEMORIES)])
    with pytest.raises(HomeAssistantError, match="full"):
        await memory.async_remember("One more fact")


def _shown(result):
    return next(
        key.description["suggested_value"]
        for key in result["data_schema"].schema
        if key == "memories"
    )


async def test_options_show_and_edit_what_is_remembered(hass, entry):
    await entry.runtime_data.memory.async_set(
        ["Has a cat called Mia.", "Likes it warm."]
    )

    result = await hass.config_entries.options.async_init(entry.entry_id)
    shown = _shown(result)
    assert shown == "Has a cat called Mia.\nLikes it warm."

    # Remembered while the form is open; an untouched field keeps it.
    await entry.runtime_data.memory.async_remember("Wakes up at six.")
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {**entry.options, "memory": True, "memories": shown}
    )
    await hass.async_block_till_done()

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert entry.options["memory"] is True
    assert "memories" not in entry.options
    # The entry was reloaded and read the memory back from storage.
    assert entry.runtime_data.memory.facts == [
        "Has a cat called Mia.",
        "Likes it warm.",
        "Wakes up at six.",
    ]

    result = await hass.config_entries.options.async_init(entry.entry_id)
    await hass.config_entries.options.async_configure(
        result["flow_id"],
        {**entry.options, "memories": "Likes it warm.\n\n  Wakes up  at six. \n"},
    )
    assert entry.runtime_data.memory.facts == ["Likes it warm.", "Wakes up at six."]

    # An emptied field is left out of what the form sends.
    result = await hass.config_entries.options.async_init(entry.entry_id)
    await hass.config_entries.options.async_configure(result["flow_id"], entry.options)
    assert entry.runtime_data.memory.facts == []


async def test_removing_the_integration_deletes_the_memory(hass, entry, hass_storage):
    await entry.runtime_data.memory.async_set(["Has a cat called Mia."])
    assert _key(entry) in hass_storage

    await hass.config_entries.async_remove(entry.entry_id)

    assert _key(entry) not in hass_storage
