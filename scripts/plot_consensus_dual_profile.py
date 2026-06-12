#!/usr/bin/env python3
"""
plot_consensus_dual_profile.py

Pour chaque position de la poche consensus, affiche côte à côte :
  - Panel gauche  : distribution des AA biophysiques du RÉCEPTEUR
  - Panel droit   : distribution des classes biophysiques du PEPTIDE

Produit deux PNG (Class A + Class B).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Biophysical colours (same palette as the rest of the pipeline) ──────────
BIOPHYS_COLORS = {
    "aromatic":  "#f28e2b",
    "hydrophobic":"#222222",
    "positive":  "#1f4ed8",
    "negative":  "#e31a1c",
    "polar":     "#c61fc6",
    "structural":   "#6dbe6d",
    "other":     "#aaaaaa",
}

BIOPHYS_ORDER = ["aromatic", "hydrophobic", "positive", "negative", "polar", "structural", "other"]

# Normalise aliases from different files → canonical labels above
BIOPHYS_ALIAS = {
    "nonpolar_aliphatic": "hydrophobic",
    "polar_uncharged":    "polar",
}

SEGMENT_COLORS = {
    "TM1": "#b7d4f0", "TM2": "#8ab8e0", "TM3": "#5d9ccf",
    "TM4": "#3a7fbf", "TM5": "#1d62af", "TM6": "#0d4a9e",
    "TM7": "#06337a", "TM8": "#001f5b",
    "ECL1": "#fdd8a4", "ECL2": "#f9a623", "ECL3": "#d47e00",
    "ICL1": "#d4f0c8", "ICL2": "#94d481", "ICL3": "#54b840",
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def simplify_gpcrdb_pos(s):
    import re
    s = str(s).strip().replace(" ", "").replace("×", "x")
    m0 = re.fullmatch(r"(\d{1,2})x(\d{1,3})", s)
    if m0:
        return f"{int(m0.group(1))}x{int(m0.group(2))}"
    m = re.fullmatch(r"(\d+)\.(\d+)x(\d+)", s)
    if m:
        a, b, c = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return f"{b}x{c}" if b >= 40 else f"{a}x{c}"
    m2 = re.findall(r"(\d{1,2})x(\d{1,3})", s)
    if m2:
        x, y = m2[-1]
        return f"{int(x)}x{int(y)}"
    return None


def gpcrdb_sort_key(pos):
    try:
        a, b = pos.split("x")
        return (int(a), int(b))
    except Exception:
        return (999, 999)


def build_percent_bars(df_sub, group_col, class_col, order):
    """Return a DataFrame (positions × classes) with % per position."""
    if df_sub.empty:
        return pd.DataFrame()
    counts = (
        df_sub.groupby([group_col, class_col])
        .size().reset_index(name="n")
    )
    counts["pct"] = counts.groupby(group_col)["n"].transform(
        lambda x: x / x.sum() * 100
    )
    pivot = counts.pivot(index=group_col, columns=class_col, values="pct").fillna(0)
    for cls in order:
        if cls not in pivot.columns:
            pivot[cls] = 0.0
    # keep only classes that have any value
    present = [c for c in order if pivot[c].sum() > 0]
    return pivot[present]


# ── Core plot ─────────────────────────────────────────────────────────────────

def plot_dual_profile(
    class_label,
    positions,        # list of gpcrdb_pos strings, sorted
    cons_df,          # consensus frequencies DataFrame
    receptor_pivot,   # DataFrame: positions × receptor biophys classes (%)
    peptide_pivot,    # DataFrame: positions × peptide biophys classes (%)
    out_png,
):
    n = len(positions)
    if n == 0:
        print(f"[WARN] no positions for {class_label}")
        return

    fig, axes = plt.subplots(
        1, 2,
        figsize=(14, max(5, n * 0.52 + 2.5)),
        sharey=True,
    )
    fig.subplots_adjust(wspace=0.06, left=0.18, right=0.96, top=0.92, bottom=0.10)

    y = np.arange(n)

    for ax_idx, (ax, pivot, title, side) in enumerate(zip(
        axes,
        [receptor_pivot, peptide_pivot],
        ["Récepteur — AA biophysique\n(distribution sur structures)",
         "Peptide — classe biophysique\n(distribution des contacts)"],
        ["receptor", "peptide"],
    )):
        left = np.zeros(n)
        cols = [c for c in BIOPHYS_ORDER if c in pivot.columns]
        for cls in cols:
            vals = pivot.reindex(positions)[cls].fillna(0).values
            color = BIOPHYS_COLORS.get(cls, "#aaaaaa")
            ax.barh(y, vals, left=left, height=0.65, color=color,
                    label=cls, zorder=2)
            # percentage label inside bar if large enough
            for i, (v, l) in enumerate(zip(vals, left)):
                if v >= 15:
                    ax.text(l + v / 2, i, f"{v:.0f}%", ha="center", va="center",
                            fontsize=6.5, color="white", fontweight="bold", zorder=3)
            left += vals

        ax.set_xlim(0, 105)
        ax.set_xlabel("Fréquence (%)", fontsize=9)
        ax.set_title(title, fontsize=9, pad=6)
        ax.axvline(50, color="gray", linewidth=0.7, linestyle=":", zorder=1)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", length=0)

        # show receptor top_aa + freq on the left panel
        if side == "receptor":
            cons_idx = cons_df.set_index("gpcrdb_pos")
            for i, pos in enumerate(positions):
                if pos in cons_idx.index:
                    row = cons_idx.loc[pos]
                    aa = row.get("top_aa", "?")
                    freq = row.get("freq_top_aa", row.get("freq_structures", ""))
                    try:
                        freq_str = f"{float(freq):.0%}"
                    except Exception:
                        freq_str = ""
                    ax.text(-3, i, f"{aa}  {freq_str}", ha="right", va="center",
                            fontsize=7.5, color="#333333")

    # y-axis labels with segment colour bands
    ax_left = axes[0]
    cons_idx = cons_df.set_index("gpcrdb_pos")
    for i, pos in enumerate(positions):
        seg = ""
        if pos in cons_idx.index:
            seg = str(cons_idx.loc[pos].get("segment_gpcrdb", ""))
        seg_color = SEGMENT_COLORS.get(seg, "#cccccc")
        # coloured band behind the label
        ax_left.barh(i, -18, left=0, height=0.85,
                     color=seg_color, alpha=0.35, zorder=0)
        ax_left.text(-20, i, pos, ha="right", va="center",
                     fontsize=8, fontweight="bold")
        if seg:
            ax_left.text(-20, i - 0.27, seg, ha="right", va="center",
                         fontsize=6.5, color="#555555", style="italic")

    ax_left.set_yticks(y)
    ax_left.set_yticklabels([""] * n)
    ax_left.set_ylim(-0.7, n - 0.3)
    axes[1].invert_xaxis()   # peptide panel grows rightward from centre

    # shared legend
    seen = {}
    for pivot in [receptor_pivot, peptide_pivot]:
        for c in pivot.columns:
            if c not in seen:
                seen[c] = mpatches.Patch(color=BIOPHYS_COLORS.get(c, "#aaa"), label=c)
    handles = [seen[k] for k in BIOPHYS_ORDER if k in seen]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               fontsize=8, frameon=False,
               bbox_to_anchor=(0.57, -0.01))

    fig.suptitle(
        f"{class_label} — profil biophysique des positions consensus\n"
        f"(récepteur à gauche · peptide à droite)",
        fontsize=11, y=0.98,
    )

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[DONE] {class_label} -> {out_png}")


# ── Data builders ─────────────────────────────────────────────────────────────

def load_data(args):
    pocket = pd.read_csv(args.pocket_tsv, sep="\t", dtype=str)
    contacts = pd.read_csv(args.contacts_tsv, sep="\t", dtype=str)

    # normalise position format in pocket file
    if "gpcrdb_display_generic_number" in pocket.columns:
        pos_map = (
            pocket[["gpcrdb_display_generic_number", "gpcrdb"]]
            .dropna().drop_duplicates()
            .set_index("gpcrdb_display_generic_number")["gpcrdb"].to_dict()
        )
    else:
        pos_map = {}

    contacts["gpcrdb_short"] = contacts["gpcrdb_pos"].map(pos_map)

    pdb_class = pocket[["pdb_id", "gpcr_class"]].drop_duplicates()
    pdb_class["pdb_id"] = pdb_class["pdb_id"].str.upper()
    contacts["pdb_id"] = contacts["pdb_id"].str.upper()
    contacts = contacts.merge(pdb_class, on="pdb_id", how="left")

    # receptor: use biophys columns from pocket (already has gpcrdb short)
    # map class labels to simple A/B
    pocket["class_simple"] = pocket["gpcr_class"].apply(
        lambda x: "A" if "A" in str(x) else ("B" if "B" in str(x) else "other")
    )
    contacts["class_simple"] = contacts["gpcr_class"].apply(
        lambda x: "A" if "A" in str(x) else ("B" if "B" in str(x) else "other")
    )

    return pocket, contacts


def build_receptor_pivot(pocket, positions, class_simple):
    sub = pocket[
        pocket["gpcrdb"].isin(positions) &
        (pocket["class_simple"] == class_simple)
    ].copy()
    # derive biophys label from boolean columns
    def biophys(row):
        if str(row.get("is_aromatic","0")) == "1":
            return "aromatic"
        if str(row.get("is_pos","0")) == "1":
            return "positive"
        if str(row.get("is_neg","0")) == "1":
            return "negative"
        if str(row.get("is_polar","0")) == "1":
            return "polar_uncharged"
        if str(row.get("is_hydrophobic","0")) == "1":
            return "hydrophobic"
        return "other"
    sub["biophys"] = sub.apply(biophys, axis=1).map(
        lambda x: BIOPHYS_ALIAS.get(x, x)
    )
    return build_percent_bars(sub, "gpcrdb", "biophys", BIOPHYS_ORDER)


def build_peptide_pivot(contacts, positions, class_simple):
    sub = contacts[
        contacts["gpcrdb_short"].isin(positions) &
        (contacts["class_simple"] == class_simple)
    ].drop_duplicates(subset=["pdb_id", "gpcrdb_short", "peptide_resnum"]).copy()
    sub["peptide_class"] = sub["peptide_class"].map(lambda x: BIOPHYS_ALIAS.get(x, x))
    return build_percent_bars(sub, "gpcrdb_short", "peptide_class", BIOPHYS_ORDER)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pocket_tsv",   required=True)
    ap.add_argument("--contacts_tsv", required=True)
    ap.add_argument("--consensus_a",  required=True)
    ap.add_argument("--consensus_b",  required=True)
    ap.add_argument("--outdir",       required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pocket, contacts = load_data(args)

    for cls_label, cls_simple, cons_path, fname in [
        ("Class A (Rhodopsin-like)", "A", args.consensus_a,
         "Class_A.consensus_dual_profile.png"),
        ("Class B1 (Secretin-like)", "B", args.consensus_b,
         "Class_B.consensus_dual_profile.png"),
    ]:
        cons_df = pd.read_csv(cons_path, sep="\t")
        positions = sorted(cons_df["gpcrdb_pos"].dropna().tolist(),
                           key=gpcrdb_sort_key)

        rec_pivot = build_receptor_pivot(pocket, positions, cls_simple)
        pep_pivot = build_peptide_pivot(contacts, positions, cls_simple)

        plot_dual_profile(
            class_label=cls_label,
            positions=positions,
            cons_df=cons_df,
            receptor_pivot=rec_pivot,
            peptide_pivot=pep_pivot,
            out_png=str(outdir / fname),
        )


if __name__ == "__main__":
    main()
