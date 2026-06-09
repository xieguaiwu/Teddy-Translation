#!/usr/bin/env python3
"""SMT vs LLM: SVM classification + Mixed effects + Paper-ready tables."""
import sys, os, csv, math, json, warnings
warnings.filterwarnings("ignore")
from collections import defaultdict

import numpy as np

HAS_SKLEARN = False
HAS_STATSMODELS = False
HAS_MPL = False

try:
    from sklearn.svm import SVC
    from sklearn.model_selection import GroupKFold, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, confusion_matrix
    HAS_SKLEARN = True
except ImportError:
    pass

try:
    import statsmodels.api as sm
    from statsmodels.formula.api import mixedlm
    HAS_STATSMODELS = True
except ImportError:
    pass

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MPL = True
except ImportError:
    pass


def load_data(path="data/feature_matrix.csv"):
    print(f"Loading: {path}", flush=True)
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f"  {len(rows)} translations", flush=True)
    arch_counts = defaultdict(int)
    for r in rows:
        arch_counts[r["architecture"]] += 1
    print(f"  SMT: {arch_counts.get('smt', 0)}, LLM: {arch_counts.get('llm', 0)}", flush=True)
    return rows


def safe_float(v):
    try:
        val = float(v)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    except (ValueError, TypeError):
        return None


def run_svm(rows, output_dir="results"):
    """Linear SVM with GroupKFold cross-validation."""
    print("\n=== SVM Classification: SMT vs LLM ===", flush=True)
    if not HAS_SKLEARN:
        print("  sklearn not available. Skipping.", flush=True)
        return

    # Only use rows with valid features
    feature_names = ["sttr", "mean_sent_len", "sent_polarity", "pos_entropy", "func_word_ratio"]
    X, y, groups = [], [], []

    for r in rows:
        vec = [safe_float(r.get(f)) for f in feature_names]
        if any(v is None for v in vec):
            continue
        X.append(vec)
        y.append(0 if r["architecture"] == "smt" else 1)
        # Group by source_file
        groups.append(r.get("source_file", "unknown"))

    X = np.array(X)
    y = np.array(y)
    n_smt = sum(1 for v in y if v == 0)
    n_llm = sum(1 for v in y if v == 1)
    print(f"  Samples: {len(y)} ({n_smt} SMT, {n_llm} LLM)", flush=True)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # GroupKFold
    gkf = GroupKFold(n_splits=5)
    acc_scores = []
    f1_scores = []
    mcc_scores = []
    all_y_true, all_y_pred = [], []

    for train_idx, test_idx in gkf.split(X_scaled, y, groups):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        svm = SVC(kernel="linear", C=1.0, class_weight="balanced")
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)

        acc_scores.append(accuracy_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))
        mcc_scores.append(matthews_corrcoef(y_test, y_pred))
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

    # Feature importance from SVM weights
    svm_final = SVC(kernel="linear", C=1.0, class_weight="balanced")
    svm_final.fit(X_scaled, y)
    feature_weights = svm_final.coef_[0]

    print(f"  Accuracy:  {np.mean(acc_scores):.3f} +/- {np.std(acc_scores):.3f}", flush=True)
    print(f"  F1 Score:   {np.mean(f1_scores):.3f} +/- {np.std(f1_scores):.3f}", flush=True)
    print(f"  MCC:        {np.mean(mcc_scores):.3f} +/- {np.std(mcc_scores):.3f}", flush=True)
    print(f"  Chance:     50% (binary classification)", flush=True)
    print(f"\n  Feature weights (SMT vs LLM):", flush=True)
    for name, weight in sorted(zip(feature_names, feature_weights), key=lambda x: -abs(x[1])):
        direction = "SMT higher" if weight < 0 else "LLM higher"
        print(f"    {name:<20} {weight:+.4f} ({direction})", flush=True)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    svm_results = {
        "accuracy_mean": round(np.mean(acc_scores), 4),
        "accuracy_std": round(np.std(acc_scores), 4),
        "f1_mean": round(np.mean(f1_scores), 4),
        "f1_std": round(np.std(f1_scores), 4),
        "mcc_mean": round(np.mean(mcc_scores), 4),
        "mcc_std": round(np.std(mcc_scores), 4),
        "feature_weights": {n: round(w, 4) for n, w in zip(feature_names, feature_weights)},
        "n_samples": len(y),
        "n_smt": n_smt,
        "n_llm": n_llm,
    }
    with open(f"{output_dir}/svm_results.json", "w") as f:
        json.dump(svm_results, f, indent=2)
    print(f"  Saved: {output_dir}/svm_results.json", flush=True)

    # Confusion matrix plot
    if HAS_MPL:
        cm = confusion_matrix(all_y_true, all_y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["SMT", "LLM"])
        ax.set_yticklabels(["SMT", "LLM"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=16)
        plt.title(f"SVM Confusion Matrix (Acc={np.mean(acc_scores):.2f})")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/svm_confusion.png", dpi=120)
        plt.close()
        print(f"  Saved: {output_dir}/svm_confusion.png", flush=True)


def run_mixedlm(rows, output_dir="results"):
    """Mixed effects models for key features."""
    print("\n=== Mixed Effects Models ===", flush=True)
    if not HAS_STATSMODELS:
        print("  statsmodels not available. Trying sklearn linear regression instead.", flush=True)
        return

    features = ["sttr", "mean_sent_len", "sent_polarity", "pos_entropy", "func_word_ratio"]
    results = {}

    for feat in features:
        data = []
        for r in rows:
            val = safe_float(r.get(feat))
            if val is None:
                continue
            data.append({
                "value": val,
                "architecture": r["architecture"],
                "direction": r["direction"],
                "genre": r["genre"],
                "source_id": r.get("source_file", "unknown").split("/")[-1].replace(".txt", ""),
            })

        if len(data) < 10:
            continue

        import pandas as pd
        df = pd.DataFrame(data)
        df["is_smt"] = (df["architecture"] == "smt").astype(int)

        try:
            model = mixedlm(
                "value ~ is_smt * direction * genre",
                df,
                groups="source_id",
            ).fit(reml=False, maxiter=100)
            results[feat] = {
                "n": len(df),
                "coef_smt": round(model.params.get("is_smt", 0), 4),
                "p_smt": round(model.pvalues.get("is_smt", 1), 6),
                "coef_smt_dir": round(model.params.get("is_smt:direction[T.zh2en]", 0), 4) if "is_smt:direction[T.zh2en]" in model.params else 0,
                "coef_smt_genre": round(model.params.get("is_smt:genre[T.news]", 0), 4) if "is_smt:genre[T.news]" in model.params else 0,
            }
            sig = "***" if results[feat]["p_smt"] < 0.001 else "**" if results[feat]["p_smt"] < 0.01 else "*" if results[feat]["p_smt"] < 0.05 else "ns"
            print(f"  {feat:<20} SMT coef={results[feat]['coef_smt']:+8.4f} p={results[feat]['p_smt']:.6f} {sig} (n={len(df)})", flush=True)
        except Exception as e:
            print(f"  {feat:<20} ERROR: {e}", flush=True)

    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/mixedlm_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {output_dir}/mixedlm_results.json", flush=True)


def generate_paper_tables(rows, output_dir="results"):
    """Generate paper-ready LaTeX tables."""
    print("\n=== Generating Paper Tables ===", flush=True)

    features = [
        ("sttr", "STTR", "Lexical Diversity"),
        ("mean_sent_len", "Mean Sent. Length", "Sentence Complexity"),
        ("sent_polarity", "Sentiment Polarity", "Sentiment"),
        ("pos_entropy", "POS Entropy", "Stylometry"),
        ("func_word_ratio", "Func. Word Ratio", "Stylometry"),
    ]

    # Table 1: Overall comparison
    table1 = []
    table1.append("\\begin{table}[h]")
    table1.append("\\centering")
    table1.append("\\caption{SMT vs LLM: Feature Comparison}")
    table1.append("\\begin{tabular}{lrrrrr}")
    table1.append("\\toprule")
    table1.append("Feature & SMT Mean & LLM Mean & Cohen's d & KS stat & p-value \\\\")
    table1.append("\\midrule")

    for feat_key, feat_name, feat_group in features:
        smt_vals = []
        llm_vals = []
        for r in rows:
            val = safe_float(r.get(feat_key))
            if val is None:
                continue
            if r["architecture"] == "smt":
                smt_vals.append(val)
            else:
                llm_vals.append(val)

        if smt_vals and llm_vals:
            m_s = sum(smt_vals) / len(smt_vals)
            m_l = sum(llm_vals) / len(llm_vals)
            n_s, n_l = len(smt_vals), len(llm_vals)
            v_s = sum((v - m_s)**2 for v in smt_vals) / (n_s - 1) if n_s > 1 else 0
            v_l = sum((v - m_l)**2 for v in llm_vals) / (n_l - 1) if n_l > 1 else 0
            s_p = math.sqrt(((n_s - 1) * v_s + (n_l - 1) * v_l) / (n_s + n_l - 2))
            d = (m_s - m_l) / s_p if s_p > 0 else 0

            from scipy import stats as sp_stats
            ks, ks_p = sp_stats.ks_2samp(smt_vals, llm_vals)
            sig = "$^{***}$" if ks_p < 0.001 else "$^{**}$" if ks_p < 0.01 else "$^{*}$" if ks_p < 0.05 else ""
            table1.append(f"  {feat_name} & {m_s:.3f} & {m_l:.3f} & {d:.3f} & {ks:.3f} & {ks_p:.2e}{sig} \\\\")

    table1.append("\\bottomrule")
    table1.append("\\end{tabular}")
    table1.append("\\end{table}")

    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/paper_table1.tex", "w") as f:
        f.write("\n".join(table1))
    print(f"  Saved: {output_dir}/paper_table1.tex", flush=True)

    # Table 2: By direction
    table2 = []
    table2.append("\\begin{table}[h]")
    table2.append("\\centering")
    table2.append("\\caption{SMT vs LLM: Feature Comparison by Direction}")
    table2.append("\\begin{tabular}{lrrrr}")
    table2.append("\\toprule")
    table2.append("Direction & Feature & SMT Mean & LLM Mean & Cohen's d \\\\")
    table2.append("\\midrule")

    for direction in ["zh2en", "en2zh"]:
        first_row = True
        for feat_key, feat_name, _ in features:
            smt_vals = []
            llm_vals = []
            for r in rows:
                if r["direction"] != direction:
                    continue
                val = safe_float(r.get(feat_key))
                if val is None:
                    continue
                if r["architecture"] == "smt":
                    smt_vals.append(val)
                else:
                    llm_vals.append(val)
            if smt_vals and llm_vals:
                m_s = sum(smt_vals) / len(smt_vals)
                m_l = sum(llm_vals) / len(llm_vals)
                n_s, n_l = len(smt_vals), len(llm_vals)
                v_s = sum((v - m_s)**2 for v in smt_vals) / (n_s - 1) if n_s > 1 else 0
                v_l = sum((v - m_l)**2 for v in llm_vals) / (n_l - 1) if n_l > 1 else 0
                s_p = math.sqrt(((n_s - 1) * v_s + (n_l - 1) * v_l) / (n_s + n_l - 2))
                d = (m_s - m_l) / s_p if s_p > 0 else 0
                dir_label = direction if first_row else ""
                table2.append(f"  {dir_label} & {feat_name} & {m_s:.3f} & {m_l:.3f} & {d:.3f} \\\\")
                first_row = False

    table2.append("\\bottomrule")
    table2.append("\\end{tabular}")
    table2.append("\\end{table}")

    with open(f"{output_dir}/paper_table2.tex", "w") as f:
        f.write("\n".join(table2))
    print(f"  Saved: {output_dir}/paper_table2.tex", flush=True)


def main():
    rows = load_data()

    # 1. SVM Classification
    run_svm(rows)

    # 2. Mixed Effects Models
    run_mixedlm(rows)

    # 3. Paper Tables
    generate_paper_tables(rows)

    print("\n=== All analyses complete! ===", flush=True)


if __name__ == "__main__":
    main()
