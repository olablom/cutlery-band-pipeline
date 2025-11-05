#!/usr/bin/env python3
# analyze_inference_log.py

import json
from pathlib import Path
import statistics

LOG_PATH = Path("inference_log.jsonl")


def main():
    lat = []
    decisions = {}
    labels = {}
    n = 0
    
    with LOG_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            n += 1
            obj = json.loads(line)
            
            # latency
            lm = obj.get("latency_ms")
            if lm is not None:
                lat.append(lm)
            
            # decision
            d = obj.get("decision", {})
            dc = d.get("decision_class", "UNKNOWN")
            decisions[dc] = decisions.get(dc, 0) + 1
            
            # label
            pl = obj.get("pred_label", "UNKNOWN")
            labels[pl] = labels.get(pl, 0) + 1
    
    print(f"total entries: {n}")
    print("decisions:", decisions)
    print("pred_labels:", labels)
    
    if lat:
        lat_sorted = sorted(lat)
        
        def pct(p):
            idx = int(len(lat_sorted) * p)
            idx = min(idx, len(lat_sorted) - 1)
            return lat_sorted[idx]
        
        print(f"latency ms: min={min(lat):.2f} max={max(lat):.2f} avg={statistics.mean(lat):.2f}")
        print(f"p50={pct(0.50):.2f} p90={pct(0.90):.2f} p95={pct(0.95):.2f} p99={pct(0.99):.2f}")


if __name__ == "__main__":
    main()

