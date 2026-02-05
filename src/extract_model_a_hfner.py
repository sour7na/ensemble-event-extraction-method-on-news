from __future__ import annotations
from typing import List, Dict
from .common import Event, normalize

# Uses HF token classification (NER) to get entities, then maps to event arguments.
# If HF can't load (no internet / missing torch), it returns [] and pipeline will still work.

TRIGGERS = {
    "Attack": ["attacked", "attack", "assault", "strike", "bombed", "explosion", "exploded"],
    "Arrest": ["arrested", "detained"],
    "Agreement": ["signed", "agreement", "deal", "accord"],
    "Disaster": ["earthquake", "flood", "storm", "wildfire"],
    "Death": ["killed", "dead", "died", "fatalities"],
}

def _lazy_pipeline():
    from transformers import pipeline
    return pipeline("token-classification", model="dslim/bert-base-NER", aggregation_strategy="simple")

def _pick_trigger(text_lower: str):
    for etype, words in TRIGGERS.items():
        for w in words:
            if f" {w} " in f" {text_lower} ":
                return etype, w
    return None, None

def run(text_id: str, text: str) -> List[Event]:
    text = normalize(text)
    lower = text.lower()

    etype, trigger = _pick_trigger(lower)
    if not etype:
        return []

    try:
        nlp = _lazy_pipeline()
        ents = nlp(text)
    except Exception:
        # HF not available; skip
        return []

    # Map entities to arguments (very simple)
    persons = [e["word"] for e in ents if e.get("entity_group") == "PER"]
    orgs = [e["word"] for e in ents if e.get("entity_group") == "ORG"]
    locs = [e["word"] for e in ents if e.get("entity_group") == "LOC"]

    args: Dict[str, str] = {}
    if persons:
        args["actor"] = persons[0]
    elif orgs:
        args["actor"] = orgs[0]
    if locs:
        args["location"] = locs[0]
    if len(persons) >= 2:
        args["target"] = persons[1]

    return [Event(
        event_type=etype,
        trigger=trigger,
        arguments=args,
        model="hf_ner",
        model_conf=0.72,  # prior confidence for hf model
        text_id=text_id
    )]
