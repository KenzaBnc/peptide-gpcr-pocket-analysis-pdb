#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_interaction_signature_from_consensus.py

Génère une figure "interaction signature" par classe à partir des fichiers
consensus_*.validable.tsv.

Entrées attendues dans chaque fichier consensus :
- gpcrdb_pos
- gpcrdb_interaction_types
Optionnel :
- label
- readable_label
- display_label
- segment_gemmi / segment_gpcrdb

Les catégories d'interaction GPCRdb sont conservées telles quelles :
  hydrophobic / aromatic / polar / vdw / other

Sorties :
- Class_A_interaction_signature_thr50.png
- Class_B_interaction_signature_thr50.png

Usage :
python3 scripts/plot_interaction_signature_from_consensus.py \
  --consensus_a out/consensus_validable/consensus_Class_A_thr50.validable.tsv \
  --consensus_b out/consensus_validable/consensus_Class_B_thr50.validable.tsv \
  --outdir out/interaction_signatures \
  --threshold 50
"""

import argparse
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


INTERACTION_ORDER = ["hydrophobic", "aromatic", "polar", "vdw", "other"]
COLORS = {
    "hydrophobic": "#1f77b4",
    "aromatic":    "#2ca02c",
    "polar":       "#ff7f0e",
    "vdw":         "#d62728",
    "other":       "#8c564b",
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--consensus_a", required=True, help="consensus_Class_A_thr50.validable.tsv")
    ap.add_argument("--consensus_b", required=True, help="consensus_Class_B_thr50.validable.tsv")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--threshold", type=int, default=50)
    return ap.parse_args()


def segment_from_gpcrdb_pos(gp: str) -> str:
    if not gp:
        return "NA"
    gp = str(gp).strip()

    m_tm = re.fullmatch(r"([1-8])x(\d+)", gp)
    if m_tm:
        h = int(m_tm.group(1))
        if 1 <= h <= 7:
            return f"TM{h}"
        if h == 8:
            return "H8"
        return "NA"

    m_loop = re.fullmatch(r"(\d{2})x(\d+)", gp)
    if m_loop:
        code = m_loop.group(1)
        return {
            "12": "ICL1",
            "23": "ECL1",
            "34": "ICL2",
            "45": "ECL2",
            "56": "ICL3",
            "67": "ECL3",
            "78": "C-term/ECL3",
        }.get(code, f"Loop{code}")

    return "NA"


def make_readable_label(row: pd.Series) -> str:
    for col in ["display_label", "readable_label", "label"]:
        if col in row.index:
            val = str(row[col]).strip()
            if val and val.lower() != "nan":
                return val

    gp = str(row.get("gpcrdb_pos", "")).strip()
    seg = None
    for col in ["segment_gemmi", "segment_gpcrdb"]:
        if col in row.index:
            val = str(row[col]).strip()
            if val and val.lower() != "nan":
                seg = val
                break

    if seg is None:
        seg = segment_from_gpcrdb_pos(gp)

    return f"{gp}\n({seg})"


def load_consensus(consensus_tsv: str) -> pd.DataFrame:
    df = pd.read_csv(consensus_tsv, sep="\t", dtype=str)

    needed = {"gpcrdb_pos", "gpcrdb_interaction_types"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(
            f"{consensus_tsv} manque les colonnes: {sorted(missing)} ; colonnes={df.columns.tolist()}"
        )

    df["gpcrdb_pos"] = df["gpcrdb_pos"].astype(str).str.strip()
    df["gpcrdb_interaction_types"] = df["gpcrdb_interaction_types"].fillna("").astype(str)

    df["plot_label"] = df.apply(make_readable_label, axis=1)

    rows = []
    for _, r in df.iterrows():
        gp = r["gpcrdb_pos"]
        label = r["plot_label"]
        raw = r["gpcrdb_interaction_types"]

        seen = set()
        for t in re.split(r"[;,]\s*", raw):
            t = t.strip().lower()
            if t:
                seen.add(t)

        if not seen:
            seen = {"other"}

        row = {"gpcrdb_pos": gp, "plot_label": label}
        for it in INTERACTION_ORDER:
            row[it] = 1 if it in seen else 0
        rows.append(row)

    out = pd.DataFrame(rows)

    freq_col = None
    for cand in ["freq_structures", "n_structures_with_pos"]:
        if cand in df.columns:
            freq_col = cand
            break

    if freq_col is not None:
        out = out.merge(df[["gpcrdb_pos", freq_col]], on="gpcrdb_pos", how="left")
        out = out.sort_values(by=freq_col, ascending=False, kind="stable")
    else:
        out = out.sort_values(by="gpcrdb_pos", kind="stable")

    return out.reset_index(drop=True)


def plot_signature(df: pd.DataFrame, class_label: str, threshold: int, out_png: Path):
    x = np.arange(len(df))
    labels = df["plot_label"].tolist()

    fig_w = max(12, 0.9 * len(df) + 4)
    fig, ax = plt.subplots(figsize=(fig_w, 5.5))

    bottom = np.zeros(len(df))
    for it in INTERACTION_ORDER:
        vals = df[it].values.astype(float)
        ax.bar(
            x,
            vals,
            bottom=bottom,
            color=COLORS[it],
            label=it,
            width=0.85,
        )
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Interaction types present (0/1)", fontsize=12)
    ax.set_title(
        f"Interaction signature (GPCRdb) — {class_label} consensus (thr={threshold}%)",
        fontsize=15
    )
    ax.set_ylim(0, max(1, int(bottom.max())) + 0.2)
    ax.legend(frameon=True, fontsize=10)
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    specs = [
        ("Class A", Path(args.consensus_a)),
        ("Class B", Path(args.consensus_b)),
    ]

    for class_label, path in specs:
        df = load_consensus(str(path))
        out_png = outdir / f"{class_label.replace(' ', '_')}_interaction_signature_thr{args.threshold}.png"
        plot_signature(df, class_label, args.threshold, out_png)
        print(f"[DONE] {class_label}: {out_png}")


if __name__ == "__main__":
    main()
