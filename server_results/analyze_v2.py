#!/usr/bin/env python3
"""SMT vs LLM: Advanced statistical analysis with KS tests, Cohen's d, and visualizations."""
import sys, os, csv, math, json
from collections import defaultdict

try:
    import numpy as np
    HAS_NP = True
except ImportError:
    HAS_NP = False

try:
    from scipy import stats as sp_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def cohens_d(vals1, vals2):
    """Cohen's d effect size."""
    n1, n2 = len(vals1), len(vals2)
    if n1 < 2 or n2 < 2:
        return 0
    m1, m2 = sum(vals1) / n1, sum(vals2) / n2
    v1 = sum((v - m1)**2 for v in vals1) / (n1 - 1)
    v2 = sum((v - m2)**2 for v in vals2) / (n2 - 1)
    s_pooled = math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    if s_pooled == 0:
        return 0
    return (m1 - m2) / s_pooled


def load_data(path="data/feature_matrix.csv"):
    print(f"Loading: {path}", flush=True)
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f"  {len(rows)} translations", flush=True)

    # Validate
    arch_counts = defaultdict(int)
    for r in rows:
        arch_counts[r["architecture"]] += 1
    print(f"  SMT: {arch_counts.get('smt', 0)}, LLM: {arch_counts.get('llm', 0)}", flush=True)
    return rows


def extract_vals(rows, feature, architecture=None, direction=None, genre=None, skip_nan=True):
    """Extract values for a feature with optional filters."""
    vals = []
    for r in rows:
        if architecture and r["architecture"] != architecture:
            continue
        if direction and r["direction"] != direction:
            continue
        if genre and r["genre"] != genre:
            continue
        try:
            val = float(r[feature])
            if skip_nan and (math.isnan(val) or math.isinf(val)):
                continue
            vals.append(val)
        except (ValueError, KeyError):
            continue
    return vals


def analyze_feature(rows, feature, label, output_dir="results"):
    """Run full analysis on one feature: means, Cohen's d, KS test, histogram."""
    smt_all = extract_vals(rows, feature, architecture="smt")
    llm_all = extract_vals(rows, feature, architecture="llm")

    if not smt_all or not llm_all:
        return None

    n_smt, n_llm = len(smt_all), len(llm_all)
    m_smt, m_llm = sum(smt_all) / n_smt, sum(llm_all) / n_llm
    d = cohens_d(smt_all, llm_all)

    dirs = {"all": (None, None)}
    genres = {"all": None}
    results_data = {"feature": feature, "label": label, "overall": {}}

    # Overall stats
    result = {
        "smt_mean": round(m_smt, 4), "llm_mean": round(m_llm, 4),
        "smt_n": n_smt, "llm_n": n_llm,
        "diff": round(m_smt - m_llm, 4),
        "cohens_d": round(d, 4),
    }
    results_data["overall"] = result

    # KS test on overall
    if HAS_SCIPY and n_smt > 0 and n_llm > 0:
        ks_stat, ks_p = sp_stats.ks_2samp(smt_all, llm_all)
        result["ks_stat"] = round(ks_stat, 4)
        result["ks_pvalue"] = round(ks_p, 6)
        sig = "***" if ks_p < 0.001 else "**" if ks_p < 0.01 else "*" if ks_p < 0.05 else "ns"
        result["ks_sig"] = sig

    # By direction
    for direction in ["zh2en", "en2zh"]:
        smt_d = extract_vals(rows, feature, architecture="smt", direction=direction)
        llm_d = extract_vals(rows, feature, architecture="llm", direction=direction)
        if smt_d and llm_d:
            m_s = sum(smt_d) / len(smt_d)
            m_l = sum(llm_d) / len(llm_d)
            d_d = cohens_d(smt_d, llm_d)
            res = {"smt_mean": round(m_s, 4), "llm_mean": round(m_l, 4),
                   "diff": round(m_s - m_l, 4), "cohens_d": round(d_d, 4),
                   "smt_n": len(smt_d), "llm_n": len(llm_d)}
            if HAS_SCIPY:
                ks_s, ks_p = sp_stats.ks_2samp(smt_d, llm_d)
                res["ks_stat"] = round(ks_s, 4)
                res["ks_pvalue"] = round(ks_p, 6)
                res["ks_sig"] = "***" if ks_p < 0.001 else "**" if ks_p < 0.01 else "*" if ks_p < 0.05 else "ns"
            results_data[f"dir_{direction}"] = res

    # By genre
    for genre in ["lit", "news"]:
        smt_g = extract_vals(rows, feature, architecture="smt", genre=genre)
        llm_g = extract_vals(rows, feature, architecture="llm", genre=genre)
        if smt_g and llm_g:
            m_s = sum(smt_g) / len(smt_g)
            m_l = sum(llm_g) / len(llm_g)
            d_g = cohens_d(smt_g, llm_g)
            res = {"smt_mean": round(m_s, 4), "llm_mean": round(m_l, 4),
                   "diff": round(m_s - m_l, 4), "cohens_d": round(d_g, 4),
                   "smt_n": len(smt_g), "llm_n": len(llm_g)}
            if HAS_SCIPY:
                ks_s, ks_p = sp_stats.ks_2samp(smt_g, llm_g)
                res["ks_stat"] = round(ks_s, 4)
                res["ks_pvalue"] = round(ks_p, 6)
                res["ks_sig"] = "***" if ks_p < 0.001 else "**" if ks_p < 0.01 else "*" if ks_p < 0.05 else "ns"
            results_data[f"genre_{genre}"] = res

    # Histogram
    if HAS_MPL:
        os.makedirs(output_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(smt_all, bins=30, alpha=0.6, label=f"SMT (n={n_smt})", color="#E74C3C")
        ax.hist(llm_all, bins=30, alpha=0.6, label=f"LLM (n={n_llm})", color="#3498DB")
        ax.set_xlabel(label)
        ax.set_ylabel("Frequency")
        ax.set_title(f"{label}: SMT vs LLM (d={d:.3f})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"{output_dir}/{feature}_hist.png", dpi=100)
        plt.close(fig)
        print(f"  Saved: {output_dir}/{feature}_hist.png", flush=True)

    return results_data


def print_results_table(all_results):
    """Print formatted results table."""
    sep = "-" * 100
    print("\n" + "=" * 100, flush=True)
    print("SMT vs LLM: Full Statistical Comparison", flush=True)
    print("=" * 100, flush=True)

    for feat_name in ["sttr", "mean_sent_len", "sent_polarity", "pos_entropy", "func_word_ratio", "alpha_ratio"]:
        res = all_results.get(feat_name)
        if not res:
            continue
        label_map = {
            "sttr": "STTR (Lexical Diversity)",
            "mean_sent_len": "Mean Sentence Length",
            "sent_polarity": "Sentiment Polarity",
            "pos_entropy": "POS Tag Entropy",
            "func_word_ratio": "Function Word Ratio",
            "alpha_ratio": "Alphabetic Ratio",
        }
        label = label_map.get(feat_name, feat_name)

        print(f"\n--- {label} ---", flush=True)
        overall = res.get("overall", {})
        ks_str = f" | KS={overall.get('ks_stat', '?')} p={overall.get('ks_pvalue', '?')} {overall.get('ks_sig', '')}"
        print(f"  Overall: SMT={overall.get('smt_mean','')} LLM={overall.get('llm_mean','')} "
              f"d={overall.get('cohens_d','')}{ks_str}", flush=True)

        for dir_key in ["dir_zh2en", "dir_en2zh"]:
            d = res.get(dir_key)
            if d:
                dir_label = dir_key.replace("dir_", "")
                ks_s = f" KS={d.get('ks_stat','?')} p={d.get('ks_pvalue','?')} {d.get('ks_sig','')}"
                print(f"  [{dir_label}] SMT={d['smt_mean']} LLM={d['llm_mean']} "
                      f"d={d['cohens_d']}{ks_s}", flush=True)

        for g_key in ["genre_lit", "genre_news"]:
            g = res.get(g_key)
            if g:
                g_label = g_key.replace("genre_", "")
                ks_s = f" KS={g.get('ks_stat','?')} p={g.get('ks_pvalue','?')} {g.get('ks_sig','')}"
                print(f"  [{g_label}]  SMT={g['smt_mean']} LLM={g['llm_mean']} "
                      f"d={g['cohens_d']}{ks_s}", flush=True)


def save_json(all_results, path="results/analysis_results.json"):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved: {path}", flush=True)


def main():
    rows = load_data()

    features = [
        ("sttr", "STTR (Lexical Diversity)"),
        ("mean_sent_len", "Mean Sentence Length"),
        ("sent_polarity", "Sentiment Polarity"),
        ("pos_entropy", "POS Tag Entropy"),
        ("func_word_ratio", "Function Word Ratio"),
        ("alpha_ratio", "Alphabetic Ratio"),
    ]

    all_results = {}
    for feat_key, feat_label in features:
        print(f"\nAnalyzing: {feat_label}...", flush=True)
        res = analyze_feature(rows, feat_key, feat_label)
        if res:
            all_results[feat_key] = res

    print_results_table(all_results)
    save_json(all_results)

    # Summary of significant findings
    print("\n" + "=" * 100, flush=True)
    print("SUMMARY: Statistically Significant Differences", flush=True)
    print("=" * 100, flush=True)
    print(f"{'Feature':<25} {'d (Cohen)':<10} {'KS p':<10} {'Direction'}", flush=True)
    print("-" * 100, flush=True)
    for feat_key, res in all_results.items():
        o = res.get("overall", {})
        d_val = o.get("cohens_d", 0)
        ks_p = o.get("ks_pvalue", 1)
        ks_sig = o.get("ks_sig", "ns")
        label = [l for k, l in features if k == feat_key][0]
        # Determine direction
        diff = o.get("diff", 0)
        direction = "LLM higher" if diff < 0 else "SMT higher"
        print(f"{label:<25} {d_val:<+10.3f} {ks_p:<10.6f} {ks_sig:<4} {direction}", flush=True)

    print("\nDone!", flush=True)


if __name__ == "__main__":
    main()
