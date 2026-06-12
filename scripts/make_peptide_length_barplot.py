"""
Barplot: peptide length per structure, colored by GPCR class (A / B1),
with secondary structure shown as stacked segments (helix vs coil/other).
Special reference structures (9IQV, 9MNI) are annotated.

Output: out/figures/peptide_length_barplot.svg  (and .png)
"""

import argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── colours ──────────────────────────────────────────────────────────────────
CLASS_COLORS = {
    "Class A": "#4C72B0",   # blue
    "Class B1": "#DD8452",  # orange
}
HELIX_ALPHA  = 1.0          # solid for helix portion
COIL_ALPHA   = 0.30         # light for coil / other portion

SPECIAL_REFS = {"9mni": "ECD-interface\nbinder", "9iqv": "knottin\nreference"}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ligands",   default="run_out/peptide_ligands_gpcr.tsv")
    ap.add_argument("--features",  default="run_out/biophys_annotations/peptide_structure_features.tsv")
    ap.add_argument("--biophys",   default="run_out/biophys_annotations/pocket_biophys_by_pocket.tsv")
    ap.add_argument("--nature",    default="out/peptide_nature_from_cif.tsv")
    ap.add_argument("--outdir",    default="out/figures")
    return ap.parse_args()


def normalise_class(raw: str) -> str:
    raw = str(raw).strip()
    if "B" in raw:
        return "Class B1"
    return "Class A"


def short_name(entity_desc: str, pdb: str) -> str:
    """Return a readable short label (≤20 chars) for the x-axis."""
    name = str(entity_desc).strip().strip("'")
    # known shortcuts
    shortcuts = {
        "Orexigenic neuropeptide QRFP": "QRFP",
        "peptide from Protachykinin-1": "Substance P",
        "De novo designed minibinder - dC2_050": "dC2_050",
        "Muscarinic toxin 3": "MT3",
        "Morphine-modulating neuropeptide B": "RFRP-3",
        "TRH peptide": "TRH",
        "Neuropeptide Y": "NPY",
    }
    if name in shortcuts:
        return shortcuts[name]
    if len(name) > 18:
        return name[:16] + "…"
    return name


def main():
    args = parse_args()
    base = Path(__file__).parent.parent

    # ── load data ────────────────────────────────────────────────────────────
    ligands  = pd.read_csv(base / args.ligands,  sep="\t")
    features = pd.read_csv(base / args.features, sep="\t")
    biophys  = pd.read_csv(base / args.biophys,  sep="\t")
    nature   = pd.read_csv(base / args.nature,   sep="\t")

    ligands["pdb_id"]  = ligands["pdb_id"].str.upper()
    features["pdb_id"] = features["pdb_id"].str.upper()
    biophys["pdb_id"]  = biophys["pdb_id"].str.upper()
    nature["pdb_id"]   = nature["pdb_id"].str.upper()

    # keep one row per pdb
    biophys_class = (biophys[["pdb_id", "gpcr_class"]]
                     .drop_duplicates("pdb_id")
                     .assign(gpcr_class=lambda d: d["gpcr_class"].apply(normalise_class)))

    # merge everything
    df = (ligands[["pdb_id", "peptide_length", "peptide_entity_desc"]]
          .merge(features[["pdb_id", "helix_fraction", "coil_fraction"]], on="pdb_id", how="left")
          .merge(biophys_class, on="pdb_id", how="left")
          .merge(nature[["pdb_id", "entity_description"]], on="pdb_id", how="left"))

    df["helix_fraction"] = df["helix_fraction"].fillna(0.0)
    df["coil_fraction"]  = df["coil_fraction"].fillna(1.0)

    # build display label
    df["label"] = df.apply(
        lambda r: short_name(
            r["entity_description"] if pd.notna(r["entity_description"]) else r["peptide_entity_desc"],
            r["pdb_id"]),
        axis=1)

    # sort: Class A first (by length desc), then Class B1 (by length desc)
    df["_class_order"] = df["gpcr_class"].map({"Class A": 0, "Class B1": 1}).fillna(2)
    df = df.sort_values(["_class_order", "peptide_length"], ascending=[True, False]).reset_index(drop=True)

    # ── plot ─────────────────────────────────────────────────────────────────
    n_bars   = len(df)
    fig_w    = max(14, n_bars * 0.85)
    fig, ax  = plt.subplots(figsize=(fig_w, 7))
    bar_width = 0.6
    tick_labels = []

    for i, row in df.iterrows():
        pdb    = row["pdb_id"].lower()
        length = row["peptide_length"]
        cls    = row["gpcr_class"]
        color  = CLASS_COLORS.get(cls, "#888888")

        helix_len = length * row["helix_fraction"]
        coil_len  = length - helix_len
        is_special = pdb in SPECIAL_REFS

        # coil segment (light, hatched)
        ax.bar(i, coil_len, bottom=helix_len, width=bar_width,
               color=color, alpha=COIL_ALPHA, linewidth=0,
               hatch="///" if coil_len > 0 else None)

        # helix segment (solid)
        if helix_len > 0:
            ax.bar(i, helix_len, width=bar_width,
                   color=color, alpha=HELIX_ALPHA, linewidth=0)

        # special reference: thick border + star above bar
        if is_special:
            ax.bar(i, length, width=bar_width,
                   color="none", edgecolor="black", linewidth=2.2, zorder=5)
            ax.text(i, length + 1.5, "★", ha="center", va="bottom",
                    fontsize=11, color="black", zorder=6)

        # x-tick label: peptide name on first line, PDB ID on second
        name = row["label"]
        tick_labels.append(f"{name}\n{row['pdb_id']}")

    # ── axes decoration ───────────────────────────────────────────────────────
    max_len = df["peptide_length"].max()
    ax.set_xlim(-0.8, n_bars - 0.2)
    ax.set_ylim(-3, max_len + 10)
    ax.set_xticks(list(range(n_bars)))
    ax.set_xticklabels(tick_labels, rotation=50, ha="right", fontsize=7.5)
    ax.set_ylabel("Peptide length (aa)", fontsize=11)
    ax.set_title("Peptide ligand lengths across GPCR structures\n"
                 "(solid = α-helix fraction, hatched = coil/other fraction)",
                 fontsize=12, pad=12)
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.tick_params(axis="x", length=0)

    # class separator line + header bands
    n_classA = (df["_class_order"] == 0).sum()
    if 0 < n_classA < n_bars:
        ax.axvline(n_classA - 0.5, color="gray", linewidth=1.5,
                   linestyle="--", alpha=0.7)
        ax.axvspan(-0.8, n_classA - 0.5, ymin=0, ymax=1,
                   color=CLASS_COLORS["Class A"], alpha=0.04, zorder=0)
        ax.axvspan(n_classA - 0.5, n_bars - 0.2, ymin=0, ymax=1,
                   color=CLASS_COLORS["Class B1"], alpha=0.07, zorder=0)
        ax.text(n_classA / 2 - 0.5, max_len + 7,
                "Class A (Rhodopsin)", ha="center", fontsize=10,
                color=CLASS_COLORS["Class A"], fontweight="bold")
        n_classB = n_bars - n_classA
        ax.text(n_classA + n_classB / 2 - 0.5, max_len + 7,
                "Class B1 (Secretin)", ha="center", fontsize=10,
                color=CLASS_COLORS["Class B1"], fontweight="bold")

    # ── legend ────────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(color=CLASS_COLORS["Class A"],  label="Class A  (solid = helix)"),
        mpatches.Patch(color=CLASS_COLORS["Class B1"], label="Class B1 (solid = helix)"),
        mpatches.Patch(facecolor="lightgray", hatch="///", label="Coil / other"),
        mpatches.Patch(facecolor="white", edgecolor="black", linewidth=2,
                       label="Reference structure (★)"),
    ]
    ax.legend(handles=legend_handles, fontsize=8.5, framealpha=0.85,
              loc="upper left", bbox_to_anchor=(1.01, 1.0),
              borderaxespad=0)

    # ── save ─────────────────────────────────────────────────────────────────
    outdir = base / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(bottom=0.28, right=0.83)

    for ext in ("svg", "png"):
        out = outdir / f"peptide_length_barplot.{ext}"
        fig.savefig(out, dpi=180)
        print(f"Saved: {out}")

    plt.close(fig)


if __name__ == "__main__":
    main()
