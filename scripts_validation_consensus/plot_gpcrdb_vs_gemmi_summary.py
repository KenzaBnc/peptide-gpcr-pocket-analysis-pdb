#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_gpcrdb_vs_gemmi_summary.py

Figure d'annexe pour comparer Gemmi vs GPCRdb à partir des tableaux :
- gpcrdb_vs_gemmi.signature_by_segment.tsv
- gpcrdb_vs_gemmi.signature_by_segment_and_interaction.tsv

Les catégories d'interaction GPCRdb sont conservées telles quelles :
  vdw / hydrophobic / aromatic / polar / other

Sortie :
- gpcrdb_vs_gemmi_summary.png

Usage :
python3 scripts_validation_consensus/plot_gpcrdb_vs_gemmi_summary.py \
  --segment_tsv out/gpcrdb_vs_gemmi.signature_by_segment.tsv \
  --segment_interaction_tsv out/gpcrdb_vs_gemmi.signature_by_segment_and_interaction.tsv \
  --out_png out/gpcrdb_vs_gemmi_summary.png
"""

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


SEGMENT_ORDER = ["ECL1", "ECL2", "TM1", "TM2", "TM3", "TM4", "TM5", "TM6", "TM7"]
INTERACTION_ORDER = ["vdw", "polar", "hydrophobic", "aromatic", "other"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segment_tsv", required=True)
    ap.add_argument("--segment_interaction_tsv", required=True)
    ap.add_argument("--out_png", required=True)
    return ap.parse_args()


def main():
    args = parse_args()

    seg_df = pd.read_csv(args.segment_tsv, sep="\t")
    inter_df = pd.read_csv(args.segment_interaction_tsv, sep="\t")

    # Harmonisation ordre segments
    seg_df["segment_final"] = pd.Categorical(
        seg_df["segment_final"], categories=SEGMENT_ORDER, ordered=True
    )
    inter_df["segment_final"] = pd.Categorical(
        inter_df["segment_final"], categories=SEGMENT_ORDER, ordered=True
    )

    if "interaction" not in inter_df.columns:
        raise ValueError("La colonne 'interaction' est absente du fichier --segment_interaction_tsv")

    # ---------- Panel A : both by segment and interaction type ----------
    both_inter = inter_df[inter_df["source"] == "both"].copy()

    pivot_inter = (
        both_inter.pivot_table(
            index="segment_final",
            columns="interaction",
            values="n_positions",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(index=SEGMENT_ORDER)
    )

    # garder uniquement colonnes présentes, dans l’ordre voulu
    present_cols = [c for c in INTERACTION_ORDER if c in pivot_inter.columns]
    pivot_inter = pivot_inter[present_cols]

    # ---------- Panel B : both vs gemmi_only ----------
    pivot_seg = (
        seg_df.pivot_table(
            index="segment_final",
            columns="source",
            values="n_positions",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(index=SEGMENT_ORDER)
    )

    for col in ["both", "gemmi_only"]:
        if col not in pivot_seg.columns:
            pivot_seg[col] = 0
    pivot_seg = pivot_seg[["both", "gemmi_only"]]

    # ---------- Plot ----------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

    # Panel A: stacked bars
    ax = axes[0]
    bottom = np.zeros(len(pivot_inter))
    colors = {
        "vdw":         "#4C78A8",
        "polar":       "#E45756",
        "hydrophobic": "#72B7B2",
        "aromatic":    "#2ca02c",
        "other":       "#B0B0B0",
    }

    for col in present_cols:
        vals = pivot_inter[col].values
        ax.bar(
            pivot_inter.index.astype(str),
            vals,
            bottom=bottom,
            label=col,
            color=colors.get(col, "#B0B0B0"),
            edgecolor="black",
            linewidth=0.4,
        )
        bottom += vals

    ax.set_title("Shared GPCRdb/Gemmi positions by segment\nand interaction category", fontsize=12)
    ax.set_ylabel("Number of positions", fontsize=11)
    ax.set_xlabel("GPCR segment", fontsize=11)
    ax.tick_params(axis="x", rotation=45)
    ax.legend(frameon=False, title="Interaction")

    # Panel B: grouped bars
    ax = axes[1]
    x = np.arange(len(pivot_seg.index))
    width = 0.38

    ax.bar(
        x - width / 2,
        pivot_seg["both"].values,
        width,
        label="both",
        color="#4C78A8",
        edgecolor="black",
        linewidth=0.4,
    )
    ax.bar(
        x + width / 2,
        pivot_seg["gemmi_only"].values,
        width,
        label="gemmi_only",
        color="#B0B0B0",
        edgecolor="black",
        linewidth=0.4,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(pivot_seg.index.astype(str), rotation=45)
    ax.set_title("Positions detected by both methods\nversus Gemmi only", fontsize=12)
    ax.set_ylabel("Number of positions", fontsize=11)
    ax.set_xlabel("GPCR segment", fontsize=11)
    ax.legend(frameon=False)

    # Global title
    fig.suptitle(
        "Comparison of peptide-contact positions detected by GPCRdb and Gemmi",
        fontsize=14,
        y=1.02
    )

    fig.tight_layout()

    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

    print("[DONE] Figure saved:", out_png)


if __name__ == "__main__":
    main()
