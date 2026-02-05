from __future__ import annotations
from typing import List, Dict
import re
from .common import Event, normalize

# Simple event taxonomy for demo (you can add more)
TRIGGERS = {
    "Attack": ["attacked", "attack", "assault", "strike", "bombed", "explosion", "exploded"],
    "Arrest": ["arrested", "detained", "custody"],
    "Agreement": ["signed", "agreement", "deal", "accord"],
    "Disaster": ["earthquake", "flood", "storm", "wildfire"],
    "Death": ["killed", "dead", "died", "fatalities"],
}

ORG_HINTS = ["police", "government", "ministry", "army", "officials", "company", "court"]

def extract_entities_heuristic(text: str) -> Dict[str, str]:
    # Quick heuristic: grab first capitalized span as PERSON/ORG and locations as in "in <X>"
    args = {}
    # actor: first capitalized phrase
    m = re.search(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b", text)
    if m:
        args["actor"] = m.group(1)

    # location: "in/at <ProperNoun...>"
    m2 = re.search(r"\b(?:in|at)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b", text)
    if m2:
        args["location"] = m2.group(1)

    # target: "on <X>" or "against <X>"
    m3 = re.search(r"\b(?:on|against)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b", text)
    if m3:
        args["target"] = m3.group(1)

    return args

def run(text_id: str, text: str) -> List[Event]:
    text = normalize(text)
    lower = text.lower()
    events: List[Event] = []

    for etype, words in TRIGGERS.items():
        for w in words:
            if re.search(rf"\b{re.escape(w)}\b", lower):
                args = extract_entities_heuristic(text)
                events.append(Event(
                    event_type=etype,
                    trigger=w,
                    arguments=args,
                    model="rules",
                    model_conf=0.55,   # prior confidence for rule model
                    text_id=text_id
                ))
                break

    return events
