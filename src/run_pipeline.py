from __future__ import annotations
import argparse
import json
import pandas as pd
from tqdm import tqdm

from . import extract_model_b_rules as model_b
from . import extract_model_c_heuristic as model_c
from . import extract_model_a_hfner as model_a
from .cs_score import compute_cs, aggregate_events

def run_one(text_id: str, text: str):
    events_by_model = {
        "rules": model_b.run(text_id, text),
        "heuristic": model_c.run(text_id, text),
        "hf_ner": model_a.run(text_id, text),  # may return [] if not available
    }
    cs = compute_cs(events_by_model, iters=3, alpha=0.6)
    merged = aggregate_events(events_by_model, cs)
    return merged

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV with columns: id,text")
    ap.add_argument("--output", required=True, help="Output JSONL file")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    assert "id" in df.columns and "text" in df.columns, "CSV must have columns: id,text"

    with open(args.output, "w", encoding="utf-8") as f:
        for _, row in tqdm(df.iterrows(), total=len(df)):
            text_id = str(row["id"])
            text = str(row["text"])
            merged = run_one(text_id, text)
            for ev in merged:
                f.write(json.dumps(ev, ensure_ascii=False) + "\n")

    print(f"Done. Wrote JSONL to: {args.output}")

if __name__ == "__main__":
    main()
