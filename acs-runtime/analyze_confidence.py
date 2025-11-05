#!/usr/bin/env python3
# analyze_confidence.py
"""Analyze confidence distribution per class."""

import json
import statistics
from pathlib import Path
from collections import defaultdict


def main():
    """Analyze confidence distribution."""
    log_path = Path("inference_log.jsonl")
    
    if not log_path.exists():
        print(f"Error: {log_path} not found")
        return 1
    
    # Collect confidence values per class
    conf_by_class = defaultdict(list)
    conf_by_decision = defaultdict(list)
    
    # Thresholds
    bg_threshold = 0.50
    cutlery_threshold = 0.85
    
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            
            pred_label = obj.get("pred_label", "UNKNOWN")
            conf = obj.get("conf", 0.0)
            decision_class = obj.get("decision", {}).get("decision_class", "UNKNOWN")
            
            conf_by_class[pred_label].append(conf)
            conf_by_decision[decision_class].append(conf)
    
    print("=" * 60)
    print("CONFIDENCE DISTRIBUTION BY CLASS")
    print("=" * 60)
    
    for class_name in sorted(conf_by_class.keys()):
        confs = conf_by_class[class_name]
        if not confs:
            continue
        
        confs_sorted = sorted(confs)
        
        def pct(p):
            idx = int(len(confs_sorted) * p)
            idx = min(idx, len(confs_sorted) - 1)
            return confs_sorted[idx]
        
        mean_conf = statistics.mean(confs)
        min_conf = min(confs)
        max_conf = max(confs)
        
        # Check threshold proximity
        threshold = bg_threshold if class_name == "BACKGROUND" else cutlery_threshold
        
        below_threshold = sum(1 for c in confs if c < threshold)
        above_threshold = len(confs) - below_threshold
        
        print(f"\n{class_name}:")
        print(f"  Count: {len(confs)}")
        print(f"  Mean: {mean_conf:.3f}")
        print(f"  Min: {min_conf:.3f}")
        print(f"  Max: {max_conf:.3f}")
        print(f"  p10: {pct(0.10):.3f}")
        print(f"  p50: {pct(0.50):.3f}")
        print(f"  p90: {pct(0.90):.3f}")
        print(f"  Threshold: {threshold:.2f}")
        print(f"  Below threshold: {below_threshold} ({below_threshold/len(confs)*100:.1f}%)")
        print(f"  Above threshold: {above_threshold} ({above_threshold/len(confs)*100:.1f}%)")
        
        # Check if many are close to threshold
        margin = 0.05  # 5% margin
        near_threshold = sum(1 for c in confs if abs(c - threshold) < margin)
        if near_threshold > 0:
            print(f"  ⚠️  Near threshold (±{margin}): {near_threshold} ({near_threshold/len(confs)*100:.1f}%)")
    
    print("\n" + "=" * 60)
    print("CONFIDENCE DISTRIBUTION BY DECISION")
    print("=" * 60)
    
    for decision_class in sorted(conf_by_decision.keys()):
        confs = conf_by_decision[decision_class]
        if not confs:
            continue
        
        mean_conf = statistics.mean(confs)
        min_conf = min(confs)
        max_conf = max(confs)
        
        print(f"\n{decision_class}:")
        print(f"  Count: {len(confs)}")
        print(f"  Mean: {mean_conf:.3f}")
        print(f"  Min: {min_conf:.3f}")
        print(f"  Max: {max_conf:.3f}")
        
        # For BACKGROUND_TRASH, check how many are actually low
        if decision_class == "BACKGROUND_TRASH":
            very_low = sum(1 for c in confs if c < 0.3)
            low = sum(1 for c in confs if 0.3 <= c < bg_threshold)
            print(f"  Very low (<0.3): {very_low} ({very_low/len(confs)*100:.1f}%)")
            print(f"  Low (0.3-{bg_threshold}): {low} ({low/len(confs)*100:.1f}%)")
        
        # For UNKNOWN_VARIANT, check distribution
        if decision_class == "UNKNOWN_VARIANT":
            high_conf = sum(1 for c in confs if c >= cutlery_threshold)
            medium_conf = sum(1 for c in confs if bg_threshold <= c < cutlery_threshold)
            print(f"  High conf (≥{cutlery_threshold}): {high_conf} ({high_conf/len(confs)*100:.1f}%)")
            print(f"  Medium conf ({bg_threshold}-{cutlery_threshold}): {medium_conf} ({medium_conf/len(confs)*100:.1f}%)")
    
    return 0


if __name__ == "__main__":
    exit(main())

