"""Decision layer: Jev judges the request, this code decides what happens.

The design follows TypeSafe's smart home demo. Every request goes to Jev once
with a speculative fan-out of narrow questions (what kind of request, which
device, which area, which action for each device type, ...). The answers come
back as calibrated probabilities and plain code combines them:

* confident command  -> call the Home Assistant service directly, no LLM
* compound request   -> an LLM splits it, each part is judged on its own
* anything uncertain -> escalate to the conversational LLM

This module has no Home Assistant dependencies so the routing is easy to test.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from .const import NOUL_YES, SENSITIVE_CONFIDENCE


@dataclass(frozen=True)
class Action:
    """One thing that can be done to a device type."""

    key: str
    description: str
    service: str
    # (service data field, multiplier) when the action needs a number.
    value: tuple[str, float] | None = None
    sensitive: bool = False


@dataclass(frozen=True)
class EntityInfo:
    """An exposed entity as shown to Jev."""

    entity_id: str
    name: str
    area: str | None = None
    state: str = ""
    aliases: tuple[str, ...] = ()

    @property
    def domain(self) -> str:
        """Return the entity domain."""
        return self.entity_id.split(".", 1)[0]


@dataclass(frozen=True)
class ServiceCall:
    """A Home Assistant service call the plan wants to make."""

    domain: str
    service: str
    entity_ids: tuple[str, ...]
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Plan:
    """What the code decided to do with a request."""

    kind: Literal["execute", "split", "llm"]
    reason: str
    calls: tuple[ServiceCall, ...] = ()
    action: str = ""
    targets: tuple[EntityInfo, ...] = ()
    value: float | None = None


_ON = Action("turn_on", "Switch it on, start it, enable it.", "turn_on")
_OFF = Action("turn_off", "Switch it off, disable it.", "turn_off")
_TOGGLE = Action("toggle", "Flip it to the opposite of its current state.", "toggle")
_ON_OFF = (_ON, _OFF, _TOGGLE)

DOMAIN_LABELS: dict[str, str] = {
    "light": "Lights and lamps.",
    "switch": "Switches, plugs and sockets.",
    "input_boolean": "Virtual on/off helpers.",
    "fan": "Fans and ventilation.",
    "cover": "Blinds, shades, shutters, curtains, garage doors and gates.",
    "climate": "Heating, thermostats and air conditioning.",
    "lock": "Door locks.",
    "media_player": "TVs, speakers and other media players.",
    "vacuum": "Robot vacuum cleaners.",
    "scene": "Scenes that set several devices at once.",
    "script": "Scripts and routines.",
}

DOMAIN_ACTIONS: dict[str, tuple[Action, ...]] = {
    "light": (
        *_ON_OFF,
        Action(
            "set_brightness",
            "Make it brighter or dimmer, or set a brightness level.",
            "turn_on",
            ("brightness_pct", 1),
        ),
    ),
    "switch": _ON_OFF,
    "input_boolean": _ON_OFF,
    "fan": (
        *_ON_OFF,
        Action("set_speed", "Set the fan speed.", "set_percentage", ("percentage", 1)),
    ),
    "cover": (
        Action("open", "Open it or raise it.", "open_cover", sensitive=True),
        Action("close", "Close it or lower it.", "close_cover"),
        Action("stop", "Stop it where it is.", "stop_cover"),
        Action(
            "set_position",
            "Move it to a given position or percentage.",
            "set_cover_position",
            ("position", 1),
            sensitive=True,
        ),
    ),
    "climate": (
        Action(
            "set_temperature",
            "Set a target temperature.",
            "set_temperature",
            ("temperature", 1),
        ),
        _ON,
        _OFF,
    ),
    "lock": (
        Action("lock", "Lock it.", "lock"),
        Action("unlock", "Unlock it.", "unlock", sensitive=True),
    ),
    "media_player": (
        Action("play", "Start or resume playback.", "media_play"),
        Action("pause", "Pause playback.", "media_pause"),
        Action("stop", "Stop playback.", "media_stop"),
        Action("next", "Skip to the next track.", "media_next_track"),
        Action("previous", "Go to the previous track.", "media_previous_track"),
        Action(
            "set_volume",
            "Set the volume to a given level.",
            "volume_set",
            ("volume_level", 0.01),
        ),
        _ON,
        _OFF,
    ),
    "vacuum": (
        Action("start", "Start or resume cleaning.", "start"),
        Action("stop", "Stop cleaning.", "stop"),
        Action("dock", "Send it back to its dock.", "return_to_base"),
    ),
    "scene": (Action("activate", "Activate the scene.", "turn_on"),),
    "script": (Action("activate", "Run the script.", "turn_on"),),
}

# Used when an action needs a number and the request gives a word instead.
_LEVELS = {"low": 20.0, "medium": 50.0, "high": 80.0, "maximum": 100.0}
_PERCENT_FIELDS = {"brightness_pct", "percentage", "position", "volume_level"}
_NUMBER = re.compile(r"-?\d+(?:[.,]\d+)?")


def _choice(instructions: str, criteria: dict[str, str | None]) -> dict[str, Any]:
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def _noul(instructions: str) -> dict[str, Any]:
    return {"type": "noul", "instructions": instructions}


def _describe(entity: EntityInfo) -> str:
    text = entity.name
    if entity.aliases:
        text += f" (also called {', '.join(entity.aliases)})"
    if entity.area:
        text += f", in {entity.area}"
    return text


def build_request(
    text: str, entities: list[EntityInfo], memory: bool = False
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Build the `state` and the full question fan-out for one user request.

    With `memory`, Jev is also asked whether the request is about the memory,
    which only the LLM can use.
    """
    areas = sorted({e.area for e in entities if e.area})
    domains = sorted({e.domain for e in entities})

    state = {
        "request": text,
        "devices": [
            {
                "id": e.entity_id,
                "name": e.name,
                "type": e.domain,
                "area": e.area,
                "state": e.state,
            }
            for e in entities
        ],
        "areas": areas,
    }

    questions: dict[str, dict[str, Any]] = {
        "category": _choice(
            "What kind of request is `request`?",
            {
                "command": (
                    "Asks to control or change a smart home device: switch it on"
                    " or off, open, close, lock, play, or set a value."
                ),
                "query": "Asks about the current state of the home or a device.",
                "conversation": (
                    "General conversation, a general knowledge question, or"
                    " anything that is not about the devices in `devices`."
                ),
            },
        ),
        "compound": _noul("`request` asks for more than one distinct action."),
        "names_device": _noul(
            "`request` refers to exactly one specific device listed in `devices`."
        ),
        "names_area": _noul(
            "`request` names a room or area listed in `areas` and targets the"
            " devices of one kind in it rather than one specific device."
        ),
        "whole_home": _noul(
            "`request` targets every device of one kind in the whole home, such"
            " as all lights or everything."
        ),
        "level": _choice(
            "Which level does `request` ask for in words?",
            {
                "low": "A low, dim, quiet or small level.",
                "medium": "A medium level or half.",
                "high": "A high, bright or loud level.",
                "maximum": "The maximum or full level.",
                "none": "`request` names no level in words.",
            },
        ),
    }
    if memory:
        # "Remember that I want 22 degrees in the bedroom" names a device, a
        # room and a value, but asks for none of it to happen now.
        questions["memory"] = _noul(
            "`request` asks the assistant to remember or forget something,"
            " rather than to do it now."
        )
    # Choice questions need at least two options; a single option is implied.
    if len(entities) > 1:
        questions["device"] = _choice(
            "Which device in `devices` does `request` refer to?",
            {e.entity_id: _describe(e) for e in entities},
        )
    if len(areas) > 1:
        questions["area"] = _choice(
            "Which area in `areas` does `request` refer to?",
            dict.fromkeys(areas),
        )
    if len(domains) > 1:
        questions["device_type"] = _choice(
            "Which type of device does `request` target?",
            {d: DOMAIN_LABELS[d] for d in domains},
        )
    # Speculative: asked for every device type before the target is known, so
    # one round trip covers every kind of command.
    for domain in domains:
        actions = DOMAIN_ACTIONS[domain]
        if len(actions) > 1:
            questions[f"action_{domain}"] = _choice(
                f"Assuming `request` targets a device of type {domain}, what"
                " should be done to it?",
                {a.key: a.description for a in actions},
            )
    return state, questions


def _pick(
    answers: dict[str, dict], name: str, only: str | None
) -> tuple[str | None, float]:
    """Return (option, confidence) of a Choice answer, or the implied option."""
    answer = answers.get(name)
    if answer is None:
        return only, 1.0 if only is not None else 0.0
    choice = answer.get("choice")
    confidence = answer.get("confidence")
    if not isinstance(confidence, (int, float)):
        probabilities = answer.get("probabilities") or {}
        confidence = probabilities.get(choice, 0.0)
    return choice, float(confidence)


def _yes(answers: dict[str, dict], name: str) -> bool:
    value = answers.get(name, {}).get("noul")
    return isinstance(value, (int, float)) and value >= NOUL_YES


def _number(text: str, targets: list[EntityInfo]) -> float | None:
    """Find a number in the request, ignoring digits in device and area names."""
    lowered = text.lower()
    for target in targets:
        for name in (target.name, target.area, *target.aliases):
            if name:
                lowered = lowered.replace(name.lower(), " ")
    if match := _NUMBER.search(lowered):
        return float(match.group().replace(",", "."))
    return None


def interpret(
    text: str,
    entities: list[EntityInfo],
    answers: dict[str, dict],
    threshold: float,
    allow_split: bool = True,
) -> Plan:
    """Turn Jev's answers into a plan. All thresholds and rules live here."""
    # Before splitting, so the LLM gets the whole request, which may also say
    # what to do now.
    if _yes(answers, "memory"):
        return Plan("llm", "request is about the memory")
    if allow_split and _yes(answers, "compound"):
        return Plan("split", "request asks for more than one action")

    category, confidence = _pick(answers, "category", None)
    if category != "command" or confidence < threshold:
        return Plan("llm", f"category={category} confidence={confidence:.2f}")

    by_id = {e.entity_id: e for e in entities}
    areas = sorted({e.area for e in entities if e.area})
    domains = sorted({e.domain for e in entities})
    only_entity = entities[0].entity_id if len(entities) == 1 else None
    only_area = areas[0] if len(areas) == 1 else None
    only_domain = domains[0] if len(domains) == 1 else None

    device_id, device_conf = _pick(answers, "device", only_entity)
    domain, domain_conf = _pick(answers, "device_type", only_domain)
    area, area_conf = _pick(answers, "area", only_area)

    targets: list[EntityInfo]
    if (
        _yes(answers, "names_device")
        and device_id in by_id
        and device_conf >= threshold
    ):
        targets = [by_id[device_id]]
    elif domain_conf < threshold or domain not in DOMAIN_ACTIONS:
        return Plan("llm", f"unclear target (device_type confidence={domain_conf:.2f})")
    elif _yes(answers, "names_area") and area_conf >= threshold:
        targets = [e for e in entities if e.area == area and e.domain == domain]
    elif _yes(answers, "whole_home"):
        targets = [e for e in entities if e.domain == domain]
    else:
        return Plan("llm", "unclear target")
    if not targets:
        return Plan("llm", "no matching device")

    domain = targets[0].domain
    actions = DOMAIN_ACTIONS[domain]
    key, action_conf = _pick(
        answers, f"action_{domain}", actions[0].key if len(actions) == 1 else None
    )
    action = next((a for a in actions if a.key == key), None)
    if action is None:
        return Plan("llm", "unclear action")
    needed = max(threshold, SENSITIVE_CONFIDENCE) if action.sensitive else threshold
    if action_conf < needed:
        return Plan("llm", f"action={key} confidence={action_conf:.2f} < {needed:.2f}")

    data: dict[str, Any] = {}
    value: float | None = None
    if action.value:
        data_field, scale = action.value
        value = _number(text, targets)
        if value is None and data_field in _PERCENT_FIELDS:
            level, level_conf = _pick(answers, "level", None)
            if level_conf >= threshold:
                value = _LEVELS.get(level or "")
        if value is None:
            return Plan("llm", f"action={key} needs a value")
        if data_field in _PERCENT_FIELDS:
            value = min(max(value, 0.0), 100.0)
        data[data_field] = round(value * scale, 2)

    call = ServiceCall(
        domain, action.service, tuple(e.entity_id for e in targets), data
    )
    return Plan(
        "execute",
        f"action={key} confidence={action_conf:.2f}",
        calls=(call,),
        action=key,
        targets=tuple(targets),
        value=value,
    )


_RESPONSES: dict[str, dict[str, str]] = {
    "en": {
        "turn_on": "Turning on {names}.",
        "turn_off": "Turning off {names}.",
        "toggle": "Toggling {names}.",
        "set_brightness": "Setting {names} to {value} percent.",
        "set_speed": "Setting {names} to {value} percent.",
        "set_position": "Setting {names} to {value} percent.",
        "set_volume": "Setting the volume of {names} to {value} percent.",
        "set_temperature": "Setting {names} to {value} degrees.",
        "open": "Opening {names}.",
        "close": "Closing {names}.",
        "stop": "Stopping {names}.",
        "lock": "Locking {names}.",
        "unlock": "Unlocking {names}.",
        "play": "Playing on {names}.",
        "pause": "Pausing {names}.",
        "next": "Skipping to the next track on {names}.",
        "previous": "Going back a track on {names}.",
        "start": "Starting {names}.",
        "dock": "Sending {names} back to the dock.",
        "activate": "Activating {names}.",
        "many": "{count} devices",
        "and": " and ",
        "failed": "Sorry, that did not work: {error}",
    },
    "cs": {
        "turn_on": "Zapínám: {names}.",
        "turn_off": "Vypínám: {names}.",
        "toggle": "Přepínám: {names}.",
        "set_brightness": "Nastavuji jas na {value} procent: {names}.",
        "set_speed": "Nastavuji rychlost na {value} procent: {names}.",
        "set_position": "Nastavuji polohu na {value} procent: {names}.",
        "set_volume": "Nastavuji hlasitost na {value} procent: {names}.",
        "set_temperature": "Nastavuji teplotu na {value} stupňů: {names}.",
        "open": "Otevírám: {names}.",
        "close": "Zavírám: {names}.",
        "stop": "Zastavuji: {names}.",
        "lock": "Zamykám: {names}.",
        "unlock": "Odemykám: {names}.",
        "play": "Spouštím přehrávání: {names}.",
        "pause": "Pozastavuji: {names}.",
        "next": "Další skladba: {names}.",
        "previous": "Předchozí skladba: {names}.",
        "start": "Spouštím: {names}.",
        "dock": "Posílám zpět do nabíječky: {names}.",
        "activate": "Aktivuji: {names}.",
        "many": "{count} zařízení",
        "and": " a ",
        "failed": "Omlouvám se, nepodařilo se to: {error}",
    },
}


def _strings(language: str) -> dict[str, str]:
    return _RESPONSES.get(language.split("-")[0].lower(), _RESPONSES["en"])


def describe(plan: Plan, language: str) -> str:
    """Return a short spoken confirmation for an executed plan."""
    strings = _strings(language)
    names = [t.name for t in plan.targets]
    if len(names) > 3:
        joined = strings["many"].format(count=len(names))
    elif len(names) > 1:
        joined = ", ".join(names[:-1]) + strings["and"] + names[-1]
    else:
        joined = names[0]
    value = f"{plan.value:g}" if plan.value is not None else ""
    return strings[plan.action].format(names=joined, value=value)


def describe_failure(error: str, language: str) -> str:
    """Return a spoken error message."""
    return _strings(language)["failed"].format(error=error)
