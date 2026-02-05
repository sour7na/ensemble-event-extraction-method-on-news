from __future__ import annotations
from typing import List, Dict
import re
from .common import Event, normalize

# A different heuristic model:
# It detects events mainly via verbs patterns + time/location hints.
PATTERNS = [
    ("Attack", r"\b(attacked|strike|struck|bombed|exploded|explosion)\b"),
    ("Arrest", r"\b(arrested|detained)\b"),
    ("Agreement", r"\b(signed|deal|agreement|accord)\b"),
    ("Disaster", r"\b(earthquake|flood|wildfire|storm)\b"),
    ("Death", r"\b(killed|died|dead|fatalities)\b"),
]

def extract_args(text: str) -> Dict[str, str]:
    args = {}
    # time: crude date like 2025-01-31 or "on Monday"
    m = re.search(r"\b(20\d{2}[-/]\d{2}[-/]\d{2})\b", text)
    if m:
        args["time"] = m.group(1)
    m2 = re.search(r"\bon\s+(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b", text)
    if m2:
        args["time"] = m2.group(0).replace("on ", "")

    # location: after "in"
    m3 = re.search(r"\bin\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b", text)
    if m3:
        args["location"] = m3.group(1)

    # actor: before verb (very rough)
    m4 = re.search(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\s+(?:has\s+)?(attacked|arrested|signed|bombed|struck|killed)\b", text)
    if m4:
        args["actor"] = m4.group(1)

    return args

def run(text_id: str, text: str) -> List[Event]:
    text = normalize(text)
    lower = text.lower()
    events: List[Event] = []

    for etype, pat in PATTERNS:
        m = re.search(pat, lower)
        if m:
            trigger = m.group(1)
            args = extract_args(text)
            events.append(Event(
                event_type=etype,
                trigger=trigger,
                arguments=args,
                model="heuristic",
                model_conf=0.60,  # prior confidence for heuristic model
                text_id=text_id
            ))
    return events
