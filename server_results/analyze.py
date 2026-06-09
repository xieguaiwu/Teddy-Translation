#!/usr/bin/env python3
"""SMT vs LLM: Feature comparison analysis on remote server."""
import sys, os, csv, math, json
from collections import defaultdict

def main():
    print("Loading feature matrix...", flush=True)
    with open("data/feature_matrix.csv", "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f"  {len(rows)} translations", flush=True)

    arch_counts = defaultdict(int)
    for r in rows:
        arch_counts[r["architecture"]] += 1
    print(f"  SMT: {arch_counts.get('smt', 0)}, LLM: {arch_counts.get('llm', 0)}", flush=True)

    # We need at least the feature matrix to do analysis
    key_features = ["sttr", "mean_sent_len", "sent_polarity", "pos_entropy", "func_word_ratio", "alpha_ratio"]

    # Overview
    sep = "-" * 80
    print("\n=== SMT vs LLM Feature Comparison ===", flush=True)
    print(f"{'Feature':<20} {'SMT mean':<12} {'LLM mean':<12} {'Diff':<12} {'SMT std':<12} {'LLM std':<12}", flush=True)
    print(sep, flush=True)

    for feat in key_features:
        smt_vals = []
        llm_vals = []
        for r in rows:
            try:
                val = float(r[feat])
                if math.isnan(val) or math.isinf(val):
                    continue
                if r["architecture"] == "smt":
                    smt_vals.append(val)
                else:
                    llm_vals.append(val)
            except (ValueError, KeyError):
                continue

        if smt_vals and llm_vals:
            n_smt = len(smt_vals)
            n_llm = len(llm_vals)
            smt_mean = sum(smt_vals) / n_smt
            llm_mean = sum(llm_vals) / n_llm
            smt_std = (sum((v - smt_mean)**2 for v in smt_vals) / n_smt) ** 0.5
            llm_std = (sum((v - llm_mean)**2 for v in llm_vals) / n_llm) ** 0.5
            diff = smt_mean - llm_mean
            print(f"{feat:<20} {smt_mean:<12.4f} {llm_mean:<12.4f} {diff:<+12.4f} {smt_std:<12.4f} {llm_std:<12.4f}", flush=True)

    # By direction
    for direction in ["zh2en", "en2zh"]:
        print(f"\n=== Direction: {direction} ===", flush=True)
        print(f"{'Feature':<20} {'SMT mean':<12} {'LLM mean':<12} {'Diff':<12}", flush=True)
        print(sep, flush=True)
        for feat in key_features:
            smt_vals = []
            llm_vals = []
            for r in rows:
                if r["direction"] != direction:
                    continue
                try:
                    val = float(r[feat])
                    if math.isnan(val) or math.isinf(val):
                        continue
                    if r["architecture"] == "smt":
                        smt_vals.append(val)
                    else:
                        llm_vals.append(val)
                except (ValueError, KeyError):
                    continue
            if smt_vals and llm_vals:
                smt_m = sum(smt_vals) / len(smt_vals)
                llm_m = sum(llm_vals) / len(llm_vals)
                print(f"{feat:<20} {smt_m:<12.4f} {llm_m:<12.4f} {smt_m-llm_m:<+12.4f}", flush=True)

    # By genre
    print("\n=== By Genre ===", flush=True)
    for feat in key_features:
        for genre in ["lit", "news"]:
            smt_vals = []
            llm_vals = []
            for r in rows:
                if r["genre"] != genre:
                    continue
                try:
                    val = float(r[feat])
                    if math.isnan(val) or math.isinf(val):
                        continue
                    if r["architecture"] == "smt":
                        smt_vals.append(val)
                    else:
                        llm_vals.append(val)
                except (ValueError, KeyError):
                    continue
            if smt_vals and llm_vals:
                smt_m = sum(smt_vals) / len(smt_vals)
                llm_m = sum(llm_vals) / len(llm_vals)
                print(f"  {feat:<18} {genre:<5} SMT={smt_m:.4f}  LLM={llm_m:.4f}  diff={smt_m-llm_m:+.4f}", flush=True)

    print("\n=== Done ===", flush=True)

if __name__ == "__main__":
    main()
