#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_leave_one_out_stability.py

Figure "papier" de stabilité Leave-One-Out (LOO) pour la poche consensus.

Montre:
(A) Taille de la poche consensus recalculée (n_consensus_LOO) par structure retirée
(B) Similarité Jaccard avec la poche de référence
(C) Estimation automatique du "core" ultra-robuste (intersection 100% LOO)
(D) Overlap constant avec la référence (ligne horizontale)

Entrée:
- leave_one_out.tsv (colonnes attendues):
    removed_pdb_id
    n_consensus_LOO
    overlap_with_ref
    jaccard_with_ref
  (n_structures_remaining optionnel)

Sorties:
- outdir/loo_stability_clean.png
- outdir/loo_stability_estimates.tsv

Usage:
python3 scripts_validation_consensus/plot_leave_one_out_stability.py \
  --loo_tsv out/consensus_validation/Class_A/leave_one_out.tsv \
  --outdir out/consensus_validation/Class_A/figures \
  --highlight 9M1O
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def estimate_n_ref_and_core(df: pd.DataFrame):
    required = {"n_consensus_LOO", "overlap_with_ref", "jaccard_with_ref"}
    miss = required - set(df.columns)
    if miss:
        raise ValueError(f"Colonnes manquantes dans loo_tsv: {sorted(miss)}")

    L = pd.to_numeric(df["n_consensus_LOO"], errors="coerce").astype(float)
    over = pd.to_numeric(df["overlap_with_ref"], errors="coerce").astype(float)
    j = pd.to_numeric(df["jaccard_with_ref"], errors="coerce").astype(float)

    denom = over * (1.0 + j) - j
    R = (j * L) / denom
    R = R.replace([np.inf, -np.inf], np.nan)
    R = R.where(R > 0)

    I = over * R
    I = I.replace([np.inf, -np.inf], np.nan)
    I = I.where(I >= 0)

    n_ref = float(np.nanmedian(R))
    n_core = float(np.nanmedian(I))

    return int(round(n_ref)), int(round(n_core))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loo_tsv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--highlight", default=None)
    ap.add_argument("--sort_by", choices=["n_consensus_LOO", "jaccard_with_ref"], default="n_consensus_LOO")
    return ap.parse_args()


def choose_default_highlight(df: pd.DataFrame) -> str:
    d = df.copy()
    d["n_consensus_LOO"] = pd.to_numeric(d["n_consensus_LOO"], errors="coerce")
    d["jaccard_with_ref"] = pd.to_numeric(d["jaccard_with_ref"], errors="coerce")

    min_size = d["n_consensus_LOO"].min()
    cand = d[d["n_consensus_LOO"] == min_size].copy()
    if cand.empty:
        return str(df["removed_pdb_id"].iloc[0])
    cand = cand.sort_values("jaccard_with_ref", ascending=False)
    return str(cand["removed_pdb_id"].iloc[0])


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.loo_tsv, sep="\t", dtype=str)

    needed = {"removed_pdb_id", "n_consensus_LOO", "overlap_with_ref", "jaccard_with_ref"}
    miss = needed - set(df.columns)
    if miss:
        raise ValueError(f"Colonnes manquantes dans {args.loo_tsv}: {sorted(miss)}")

    df["n_consensus_LOO"] = pd.to_numeric(df["n_consensus_LOO"], errors="coerce")
    df["overlap_with_ref"] = pd.to_numeric(df["overlap_with_ref"], errors="coerce")
    df["jaccard_with_ref"] = pd.to_numeric(df["jaccard_with_ref"], errors="coerce")

    n_ref, n_core = estimate_n_ref_and_core(df)

    df = df.sort_values(args.sort_by, ascending=True).reset_index(drop=True)

    x = np.arange(len(df))
    labels = df["removed_pdb_id"].astype(str).tolist()
    sizes = df["n_consensus_LOO"].astype(float).values
    jacc = df["jaccard_with_ref"].astype(float).values
    overlap_const = float(df["overlap_with_ref"].dropna().iloc[0])

    highlight = args.highlight or choose_default_highlight(df)
    highlight = str(highlight).upper()

    fig, ax = plt.subplots(figsize=(13, 7.5))

    # Barres = taille de poche
    bars = ax.bar(
        x, sizes,
        color="#4C78A8",
        alpha=0.90,
        edgecolor="black",
        linewidth=0.4,
        label="Consensus pocket size after LOO"
    )
    ax.set_ylabel("Consensus pocket size (LOO)", fontsize=13)

    # Ligne core
    core_line = ax.axhline(
        n_core,
        linestyle="--",
        linewidth=2.2,
        color="#1f77b4",
        label=f"Ultra-robust core (≈ {n_core} positions)"
    )

    # Annotation core
    ax.text(
        0.2, n_core + 0.35,
        "stable shared core",
        color="#1f77b4",
        fontsize=11,
        fontweight="bold"
    )

    # Axe droit
    ax2 = ax.twinx()

    # Courbe Jaccard
    jacc_line = ax2.plot(
        x, jacc,
        marker="o",
        linewidth=2.2,
        color="#d62728",
        label="Jaccard similarity vs reference"
    )[0]

    # Ligne overlap constant
    overlap_line = ax2.axhline(
        overlap_const,
        linestyle=":",
        linewidth=2.5,
        color="#2ca02c",
        label=f"Constant overlap with reference ({overlap_const:.4f})"
    )

    # Annotation overlap
    ax2.text(
        len(x) - 5.7, overlap_const + 0.002,
        "constant overlap\n= invariant core relative to ref",
        color="#2ca02c",
        fontsize=10,
        ha="left",
        va="bottom",
        fontweight="bold"
    )

    ax2.set_ylabel("Similarity vs reference pocket", fontsize=13)

    # Highlight discret
    upper_labels = [s.upper() for s in labels]
    if highlight in upper_labels:
        idx = upper_labels.index(highlight)
        ax2.scatter(x[idx], jacc[idx], s=95, color="black", zorder=6)
        ax2.text(
            x[idx] + 0.15, jacc[idx] + 0.001,
            highlight,
            fontsize=10,
            fontweight="bold",
            color="black"
        )

    # Ticks X
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=11)

    # Titre
    ax.set_title(
        "Leave-one-out stability of the peptide-binding consensus pocket\n"
        f"Reference pocket size ≈ {n_ref} | Ultra-robust core ≈ {n_core}",
        fontsize=16,
        pad=18
    )

    # Légende en haut, hors zone du titre
    handles = [bars, core_line, jacc_line, overlap_line]
    labels_leg = [
        "Consensus pocket size after LOO",
        f"Ultra-robust core (≈ {n_core} positions)",
        "Jaccard similarity vs reference",
        f"Constant overlap with reference ({overlap_const:.4f})"
    ]
    fig.legend(
        handles, labels_leg,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=2,
        frameon=False,
        fontsize=11
    )

    # Un peu d'espace en haut pour éviter tout chevauchement
    fig.subplots_adjust(top=0.83, bottom=0.18, left=0.08, right=0.92)

    out_png = outdir / "loo_stability_clean.png"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("[DONE] Saved figure:", out_png)


if __name__ == "__main__":
    main()
