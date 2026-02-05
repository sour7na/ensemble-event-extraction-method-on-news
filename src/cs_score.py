from __future__ import annotations
from typing import List, Dict, Tuple
from .common import Event, event_similarity

def compute_cs(events_by_model: Dict[str, List[Event]], iters: int = 3, alpha: float = 0.6) -> Dict[Tuple[str, int], float]:
    """
    Returns CS per (model_name, index_in_that_model_list) after iterative update.
    Uses pairwise consistency between models + prior confidences.
    """
    # Initialize CS with model_conf priors
    cs: Dict[Tuple[str, int], float] = {}
    for m, evs in events_by_model.items():
        for i, e in enumerate(evs):
            cs[(m, i)] = float(e.model_conf)

    models = list(events_by_model.keys())

    for _ in range(iters):
        new_cs = dict(cs)
        for m in models:
            evs_m = events_by_model[m]
            for i, e in enumerate(evs_m):
                sims = []
                for m2 in models:
                    if m2 == m:
                        continue
                    evs_2 = events_by_model[m2]
                    if not evs_2:
                        continue
                    # best-match similarity with other model outputs
                    best = max(event_similarity(e, e2) for e2 in evs_2)
                    sims.append(best)

                consistency = sum(sims) / len(sims) if sims else 0.0
                prior = cs[(m, i)]
                new_cs[(m, i)] = alpha * prior + (1 - alpha) * consistency
        cs = new_cs

    return cs

def aggregate_events(events_by_model: Dict[str, List[Event]], cs: Dict[Tuple[str, int], float]) -> List[dict]:
    """
    Merge events: cluster by event_type + trigger (simple), aggregate CS as weighted average.
    """
    merged = {}
    for m, evs in events_by_model.items():
        for i, e in enumerate(evs):
            key = (e.event_type, e.trigger.lower(), e.text_id)
            item = merged.get(key)
            score = cs.get((m, i), e.model_conf)
            if item is None:
                merged[key] = {
                    "text_id": e.text_id,
                    "event_type": e.event_type,
                    "trigger": e.trigger,
                    "arguments": dict(e.arguments),
                    "support": [{"model": m, "cs": score, "prior": e.model_conf}],
                }
            else:
                item["support"].append({"model": m, "cs": score, "prior": e.model_conf})
                # simple argument fill: keep existing, add missing
                for k, v in e.arguments.items():
                    if k not in item["arguments"] or not item["arguments"][k]:
                        item["arguments"][k] = v

    # final score: avg of cs in support
    out = []
    for key, item in merged.items():
        s = [x["cs"] for x in item["support"]]
        item["confidence_score"] = sum(s) / max(len(s), 1)
        out.append(item)
    # sort by confidence desc
    out.sort(key=lambda x: x["confidence_score"], reverse=True)
    return out
