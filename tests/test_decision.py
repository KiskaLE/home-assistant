"""Tests for the routing rules that turn Jev's answers into a plan."""

from custom_components.jev_assist.decision import (
    EntityInfo,
    build_request,
    describe,
    interpret,
)

ENTITIES = [
    EntityInfo("light.kitchen", "Kitchen light", "Kitchen", "off"),
    EntityInfo("light.bedroom", "Bedroom light", "Bedroom", "on"),
    EntityInfo("lock.front_door", "Front door", "Hall", "locked"),
    EntityInfo("climate.living_room", "Thermostat", "Living room", "heat"),
]


def choice(option, confidence=0.95):
    return {"type": "choice", "choice": option, "confidence": confidence}


def noul(probability):
    return {"type": "noul", "noul": probability}


def answers(**overrides):
    base = {
        "category": choice("command"),
        "compound": noul(0.02),
        "names_device": noul(0.97),
        "names_area": noul(0.05),
        "whole_home": noul(0.03),
        "level": choice("none"),
        "device": choice("light.kitchen"),
        "area": choice("Kitchen"),
        "device_type": choice("light"),
        "action_light": choice("turn_on"),
        "action_lock": choice("lock"),
        "action_climate": choice("set_temperature"),
    }
    return {**base, **overrides}


def test_build_request_fans_out_one_action_question_per_device_type():
    state, questions = build_request("turn on the kitchen light", ENTITIES)
    assert state["request"] == "turn on the kitchen light"
    assert set(questions["device"]["criteria"]) == {e.entity_id for e in ENTITIES}
    assert {"action_light", "action_lock", "action_climate"} <= set(questions)
    assert questions["compound"]["type"] == "noul"


def test_single_option_questions_are_left_out():
    _, questions = build_request("on", ENTITIES[:1])
    assert "device" not in questions
    assert "device_type" not in questions
    plan = interpret(
        "on",
        ENTITIES[:1],
        {
            "category": choice("command"),
            "names_device": noul(0.9),
            "action_light": choice("turn_on"),
        },
        0.7,
    )
    assert plan.kind == "execute"
    assert plan.calls[0].entity_ids == ("light.kitchen",)


def test_confident_command_executes_directly():
    plan = interpret("turn on the kitchen light", ENTITIES, answers(), 0.7)
    assert plan.kind == "execute"
    call = plan.calls[0]
    assert (call.domain, call.service) == ("light", "turn_on")
    assert call.entity_ids == ("light.kitchen",)
    assert describe(plan, "en") == "Turning on Kitchen light."
    assert describe(plan, "cs-CZ") == "Zapínám: Kitchen light."


def test_conversation_goes_to_llm():
    plan = interpret(
        "tell me a joke", ENTITIES, answers(category=choice("conversation")), 0.7
    )
    assert plan.kind == "llm"


def test_uncertain_category_goes_to_llm():
    plan = interpret("lights?", ENTITIES, answers(category=choice("command", 0.5)), 0.7)
    assert plan.kind == "llm"


def test_compound_request_is_split_once():
    compound = answers(compound=noul(0.93))
    assert interpret("a and b", ENTITIES, compound, 0.7).kind == "split"
    assert interpret("a", ENTITIES, compound, 0.7, allow_split=False).kind == "execute"


def test_whole_home_targets_every_device_of_the_type():
    plan = interpret(
        "turn off all the lights",
        ENTITIES,
        answers(
            names_device=noul(0.04),
            whole_home=noul(0.96),
            action_light=choice("turn_off"),
        ),
        0.7,
    )
    assert plan.kind == "execute"
    assert plan.calls[0].entity_ids == ("light.kitchen", "light.bedroom")


def test_area_targets_devices_of_the_type_in_that_area():
    plan = interpret(
        "lights off in the bedroom",
        ENTITIES,
        answers(
            names_device=noul(0.2),
            names_area=noul(0.9),
            area=choice("Bedroom"),
            action_light=choice("turn_off"),
        ),
        0.7,
    )
    assert plan.calls[0].entity_ids == ("light.bedroom",)


def test_unclear_target_goes_to_llm():
    plan = interpret("turn it on", ENTITIES, answers(names_device=noul(0.4)), 0.7)
    assert plan.kind == "llm"


def test_sensitive_action_needs_stricter_confidence():
    unlock = {"device": choice("lock.front_door")}
    plan = interpret(
        "unlock the front door",
        ENTITIES,
        answers(**unlock, action_lock=choice("unlock", 0.8)),
        0.7,
    )
    assert plan.kind == "llm"
    plan = interpret(
        "unlock the front door",
        ENTITIES,
        answers(**unlock, action_lock=choice("unlock", 0.95)),
        0.7,
    )
    assert plan.kind == "execute"
    assert plan.calls[0].service == "unlock"


def test_number_is_read_by_code():
    plan = interpret(
        "set the kitchen light to 40 percent",
        ENTITIES,
        answers(action_light=choice("set_brightness")),
        0.7,
    )
    assert plan.calls[0].data == {"brightness_pct": 40}
    assert describe(plan, "en") == "Setting Kitchen light to 40 percent."


def test_level_word_is_used_when_there_is_no_number():
    plan = interpret(
        "dim the kitchen light",
        ENTITIES,
        answers(action_light=choice("set_brightness"), level=choice("low")),
        0.7,
    )
    assert plan.calls[0].data == {"brightness_pct": 20}


def test_digits_in_device_names_are_not_values():
    entities = [
        EntityInfo("light.lamp_2", "Lamp 2"),
        EntityInfo("light.lamp_3", "Lamp 3"),
    ]
    plan = interpret(
        "dim lamp 2",
        entities,
        answers(device=choice("light.lamp_2"), action_light=choice("set_brightness")),
        0.7,
    )
    assert plan.kind == "llm"


def test_temperature_without_a_number_goes_to_llm():
    plan = interpret(
        "make it warmer",
        ENTITIES,
        answers(device=choice("climate.living_room"), level=choice("high")),
        0.7,
    )
    assert plan.kind == "llm"
    plan = interpret(
        "set the thermostat to 21,5",
        ENTITIES,
        answers(device=choice("climate.living_room")),
        0.7,
    )
    assert plan.calls[0].data == {"temperature": 21.5}
