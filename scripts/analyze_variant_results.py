#!/usr/bin/env python3
"""
Analyze variant matching results from inference log.

This script reads inference_log.jsonl and generates statistics
for variant matching performance.
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import numpy as np


def load_inference_log(log_path: str) -> List[Dict]:
    """Load inference log entries."""
    entries = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def extract_variant_stats(entries: List[Dict]) -> Dict:
    """Extract variant matching statistics."""
    stats = {
        "total": len(entries),
        "by_type": defaultdict(int),
        "by_variant": defaultdict(int),
        "by_decision": defaultdict(int),
        "scores": defaultdict(list),
        "latencies": {
            "type_classification": [],
            "feature_extraction": [],
            "total": [],
        },
        "confusion_matrix": defaultdict(lambda: defaultdict(int)),
    }
    
    for entry in entries:
        decision = entry.get("decision", {})
        pred_type = decision.get("pred_type")
        decision_class = decision.get("decision_class")
        variant_name = decision.get("manufacturer")
        variant_score = decision.get("variant_score", 0.0)
        latency_ms = entry.get("latency_ms", 0.0)
        
        # Count by type
        if pred_type:
            stats["by_type"][pred_type] += 1
        
        # Count by decision
        if decision_class:
            stats["by_decision"][decision_class] += 1
        
        # Count by variant (if matched)
        if variant_name:
            stats["by_variant"][variant_name] += 1
            stats["scores"][variant_name].append(variant_score)
        
        # Latency
        stats["latencies"]["total"].append(latency_ms)
        
        # Confusion matrix (if we have ground truth)
        # TODO: Add ground truth comparison when available
    
    return stats


def calculate_accuracy_metrics(stats: Dict, ground_truth: Dict = None) -> Dict:
    """Calculate accuracy metrics (if ground truth available)."""
    if not ground_truth:
        return {"note": "Ground truth not available"}
    
    # TODO: Implement accuracy calculation when ground truth is available
    return {}


def generate_report(stats: Dict, output_path: Path = None):
    """Generate text report from statistics."""
    report_lines = []
    
    report_lines.append("=" * 60)
    report_lines.append("Variant Matching Results Report")
    report_lines.append("=" * 60)
    report_lines.append("")
    
    # Overall statistics
    report_lines.append(f"Total Images Processed: {stats['total']}")
    report_lines.append("")
    
    # By type
    report_lines.append("Distribution by Type:")
    for type_name, count in sorted(stats["by_type"].items()):
        percentage = 100 * count / stats["total"]
        report_lines.append(f"  {type_name}: {count} ({percentage:.1f}%)")
    report_lines.append("")
    
    # By decision
    report_lines.append("Distribution by Decision:")
    for decision, count in sorted(stats["by_decision"].items()):
        percentage = 100 * count / stats["total"]
        report_lines.append(f"  {decision}: {count} ({percentage:.1f}%)")
    report_lines.append("")
    
    # By variant
    if stats["by_variant"]:
        report_lines.append("Variant Matches:")
        total_matched = sum(stats["by_variant"].values())
        for variant, count in sorted(stats["by_variant"].items(), key=lambda x: -x[1]):
            percentage = 100 * count / total_matched if total_matched > 0 else 0
            scores = stats["scores"][variant]
            if scores:
                mean_score = np.mean(scores)
                min_score = np.min(scores)
                max_score = np.max(scores)
                report_lines.append(
                    f"  {variant}: {count} ({percentage:.1f}%) "
                    f"[score: {mean_score:.3f} (min: {min_score:.3f}, max: {max_score:.3f})]"
                )
            else:
                report_lines.append(f"  {variant}: {count} ({percentage:.1f}%)")
        report_lines.append("")
    
    # Score statistics
    all_scores = []
    for scores in stats["scores"].values():
        all_scores.extend(scores)
    
    if all_scores:
        report_lines.append("Score Statistics:")
        report_lines.append(f"  Mean: {np.mean(all_scores):.3f}")
        report_lines.append(f"  Std Dev: {np.std(all_scores):.3f}")
        report_lines.append(f"  Median: {np.median(all_scores):.3f}")
        report_lines.append(f"  Min: {np.min(all_scores):.3f}")
        report_lines.append(f"  Max: {np.max(all_scores):.3f}")
        report_lines.append("")
    
    # Latency statistics
    if stats["latencies"]["total"]:
        latencies = stats["latencies"]["total"]
        report_lines.append("Latency Statistics:")
        report_lines.append(f"  Mean: {np.mean(latencies):.2f} ms")
        report_lines.append(f"  P50: {np.percentile(latencies, 50):.2f} ms")
        report_lines.append(f"  P95: {np.percentile(latencies, 95):.2f} ms")
        report_lines.append(f"  P99: {np.percentile(latencies, 99):.2f} ms")
        report_lines.append(f"  Max: {np.max(latencies):.2f} ms")
        report_lines.append("")
    
    # Variant matching success rate
    total_images = stats["total"]
    matched_count = sum(stats["by_variant"].values())
    unknown_count = stats["by_decision"].get("UNKNOWN_VARIANT", 0)
    
    report_lines.append("Variant Matching Success Rate:")
    report_lines.append(f"  Matched (above threshold): {matched_count} ({100*matched_count/total_images:.1f}%)")
    report_lines.append(f"  Unknown Variant: {unknown_count} ({100*unknown_count/total_images:.1f}%)")
    report_lines.append(f"  Background/Other: {total_images - matched_count - unknown_count} ({100*(total_images-matched_count-unknown_count)/total_images:.1f}%)")
    report_lines.append("")
    
    report_text = "\n".join(report_lines)
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"Report saved to: {output_path}")
    else:
        print(report_text)
    
    return report_text


def main():
    parser = argparse.ArgumentParser(description="Analyze variant matching results")
    parser.add_argument(
        "--log",
        type=str,
        default="acs-runtime/logs/inference_log.jsonl",
        help="Path to inference log file",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for report (default: stdout)",
    )
    parser.add_argument(
        "--json",
        type=str,
        help="Save statistics as JSON to this path",
    )
    args = parser.parse_args()
    
    log_path = Path(args.log)
    if not log_path.exists():
        print(f"Error: Log file not found: {log_path}")
        return 1
    
    print(f"[Load] Loading inference log: {log_path}")
    entries = load_inference_log(str(log_path))
    print(f"  Loaded {len(entries)} entries")
    
    print(f"\n[Analyze] Computing statistics...")
    stats = extract_variant_stats(entries)
    
    # Generate report
    output_path = Path(args.output) if args.output else None
    report_text = generate_report(stats, output_path)
    
    # Save JSON if requested
    if args.json:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to native Python types for JSON
        json_stats = {
            "total": stats["total"],
            "by_type": dict(stats["by_type"]),
            "by_variant": dict(stats["by_variant"]),
            "by_decision": dict(stats["by_decision"]),
            "scores": {
                k: [float(s) for s in v] for k, v in stats["scores"].items()
            },
            "latencies": {
                k: [float(l) for l in v] for k, v in stats["latencies"].items()
            },
        }
        
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_stats, f, indent=2)
        print(f"\nStatistics saved to: {json_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

