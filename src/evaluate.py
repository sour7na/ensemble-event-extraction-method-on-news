from __future__ import annotations
import argparse
import json
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_jsonl", required=True)
    ap.add_argument("--labels_csv", required=True, help="CSV with columns: id,true_event_type (one label per text)")
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    labels = pd.read_csv(args.labels_csv)
    lab = dict(zip(labels["id"].astype(str), labels["true_event_type"].astype(str)))

    # For each text_id, pick top confidence event_type as prediction (or None)
    best = {}
    with open(args.pred_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            tid = str(obj["text_id"])
            score = float(obj["confidence_score"])
            et = str(obj["event_type"])
            if tid not in best or score > best[tid][0]:
                best[tid] = (score, et)

    y_true, y_pred = [], []
    for tid, true_et in lab.items():
        pred_et = best.get(tid, (0.0, "None"))[1]
        y_true.append(true_et)
        y_pred.append(pred_et)

    labels_set = sorted(list(set(y_true + y_pred)))
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels_set, average="macro", zero_division=0)

    out = pd.DataFrame([{
        "metric": "macro",
        "precision": p,
        "recall": r,
        "f1": f1,
        "num_samples": len(y_true)
    }])
    out.to_csv(args.out_csv, index=False)
    print(out)

if __name__ == "__main__":
    main()
