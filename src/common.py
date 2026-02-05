from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import re

@dataclass
class Event:
    event_type: str
    trigger: str
    arguments: Dict[str, str]
    model: str
    model_conf: float
    text_id: str

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())

def simple_tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9']+|[^\w\s]", text)

def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

def arg_overlap_f1(a: Dict[str, str], b: Dict[str, str]) -> float:
    # Token-based overlap over argument values (rough but fast)
    def val_tokens(d: Dict[str, str]) -> set:
        s = " ".join([v for v in d.values() if v])
        toks = set(t.lower() for t in simple_tokenize(s) if re.match(r"[A-Za-z0-9']+$", t))
        return toks

    A = val_tokens(a)
    B = val_tokens(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    tp = len(A & B)
    prec = tp / max(len(A), 1)
    rec = tp / max(len(B), 1)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)

def event_similarity(e1: Event, e2: Event) -> float:
    # Weighted similarity: type + trigger + args overlap
    type_sim = 1.0 if e1.event_type == e2.event_type else 0.0
    trig_sim = 1.0 if e1.trigger.lower() == e2.trigger.lower() else 0.0
    args_sim = arg_overlap_f1(e1.arguments, e2.arguments)
    return 0.5 * type_sim + 0.2 * trig_sim + 0.3 * args_sim
