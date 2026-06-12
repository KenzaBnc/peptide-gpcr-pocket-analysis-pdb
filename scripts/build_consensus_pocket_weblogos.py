#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_consensus_pocket_weblogos.py

Objectif
--------
Pour Class A et Class B :
1. lire les positions consensus (thr=50%)
2. extraire les AA observés dans la poche depuis :
   pocket_biophys_by_residue.with_gpcrdb_segments.with_target_resnum.tsv
3. construire un pseudo-alignement de poche
4. générer un weblogo avec couleurs biophysiques personnalisées
5. exporter un tableau de fréquences par position

Dépendances
-----------
pip install pandas numpy matplotlib logomaker

Exemple
-------
python3 scripts/build_consensus_pocket_weblogos.py \
  --pocket_tsv run_out/pocket_biophys_by_residue.with_gpcrdb_segments.with_target_resnum.tsv \
  --consensus_a out/consensus_class_A_thr.validable.tsv \
  --consensus_b out/consensus_class_B_thr.validable.tsv \
  --outdir out/pocket_weblogos
"""

import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import logomaker


# ============================================================
# Helpers
# ============================================================

AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")

AA_TO_BIOPHYS = {
    # Schéma personnalisé 6 classes (Lehninger 7e éd. comme référence générale).
    # C → hydrophobic (KD = +2.5). G/P → structural (contraintes conformationnelles).
    "A": "hydrophobic", "V": "hydrophobic", "I": "hydrophobic",
    "L": "hydrophobic", "M": "hydrophobic", "C": "hydrophobic",

    "F": "aromatic", "W": "aromatic", "Y": "aromatic",

    "K": "positive", "R": "positive", "H": "positive",

    "D": "negative", "E": "negative",

    "S": "polar", "T": "polar", "N": "polar", "Q": "polar",

    "G": "structural", "P": "structural",
}

BIOPHYS_COLORS_LOGO = {
    "hydrophobic": "#222222",   # noir foncé
    "aromatic":    "#f28e2b",   # orange
    "positive":    "#1f4ed8",   # bleu
    "negative":    "#e31a1c",   # rouge
    "polar":       "#c61fc6",   # violet
    "structural":     "#6dbe6d",   # vert
}


def norm_pdb(x: str) -> str:
    return str(x).strip().upper()


def simplify_gpcrdb_pos(pos: str):
    if pos is None or (isinstance(pos, float) and np.isnan(pos)):
        return None
    s = str(pos).strip().replace(" ", "").replace("×", "x")
    if not s:
        return None

    m0 = re.fullmatch(r"(\d{1,2})x(\d{1,3})", s)
    if m0:
        return f"{int(m0.group(1))}x{int(m0.group(2))}"

    m = re.fullmatch(r"(\d+)\.(\d+)x(\d+)", s)
    if m:
        a = int(m.group(1))
        b = int(m.group(2))
        c = int(m.group(3))
        if b >= 40:
            return f"{b}x{c}"
        return f"{a}x{c}"

    m2 = re.findall(r"(\d{1,2})x(\d{1,3})", s)
    if m2:
        x, y = m2[-1]
        return f"{int(x)}x{int(y)}"

    return None


def gpcrdb_sort_key(pos: str):
    try:
        a, b = pos.split("x")
        return (int(a), int(b))
    except Exception:
        return (999999, 999999)


def simplify_class_label(s: str) -> str:
    s = str(s)
    if "Class A" in s:
        return "Class A"
    if "Class B" in s:
        return "Class B"
    return "Other"


def major_biophys_class(aa_series: pd.Series) -> str:
    vals = []
    for aa in aa_series.dropna().astype(str):
        aa = aa.strip().upper()[:1]
        if aa in AA_TO_BIOPHYS:
            vals.append(AA_TO_BIOPHYS[aa])
    if not vals:
        return "NA"
    return pd.Series(vals).mode().iloc[0]


def get_logo_color_scheme():
    """
    Retourne un dict AA -> couleur pour logomaker.
    """
    color_scheme = {}
    for aa in AA_ORDER:
        biophys = AA_TO_BIOPHYS.get(aa, "polar")
        color_scheme[aa] = BIOPHYS_COLORS_LOGO[biophys]
    return color_scheme

# ============================================================
# Loaders
# ============================================================

def load_pocket_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str)

    required = ["pdb_id", "gpcr_class", "aa"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[pocket_tsv] colonnes manquantes: {missing}")

    df["pdb_id"] = df["pdb_id"].map(norm_pdb)
    df["class_simple"] = df["gpcr_class"].map(simplify_class_label)

    # Choix de la meilleure colonne GPCRdb disponible
    if "gpcrdb" in df.columns:
        df["gpcrdb_pos"] = df["gpcrdb"].map(simplify_gpcrdb_pos)
    elif "gpcrdb_display_generic_number" in df.columns:
        df["gpcrdb_pos"] = df["gpcrdb_display_generic_number"].map(simplify_gpcrdb_pos)
    else:
        raise ValueError("[pocket_tsv] il faut une colonne gpcrdb ou gpcrdb_display_generic_number")

    df["aa"] = df["aa"].astype(str).str.strip().str.upper().str[:1]

    # Garder seulement les vrais AA standards
    df = df[df["aa"].isin(AA_ORDER)].copy()

    # Écarter les non mappés si la colonne existe
    if "is_unmapped_gpcrdb" in df.columns:
        df["is_unmapped_gpcrdb"] = pd.to_numeric(df["is_unmapped_gpcrdb"], errors="coerce").fillna(1).astype(int)
        df = df[df["is_unmapped_gpcrdb"] == 0].copy()

    df = df.dropna(subset=["pdb_id", "class_simple", "gpcrdb_pos", "aa"]).copy()

    return df


def load_consensus_positions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    if "gpcrdb_pos" not in df.columns:
        raise ValueError(f"[consensus] colonne gpcrdb_pos manquante dans {path}")

    df["gpcrdb_pos"] = df["gpcrdb_pos"].map(simplify_gpcrdb_pos)
    df = df.dropna(subset=["gpcrdb_pos"]).copy()
    df = df.drop_duplicates(subset=["gpcrdb_pos"]).copy()

    keep_cols = ["gpcrdb_pos"]
    for c in ["freq_structures", "segment_gpcrdb", "gpcrdb_interaction_types"]:
        if c in df.columns:
            keep_cols.append(c)

    return df[keep_cols].copy()


# ============================================================
# Core builders
# ============================================================

def build_class_pocket_alignment(
    pocket_df: pd.DataFrame,
    consensus_df: pd.DataFrame,
    class_label: str,
):
    """
    Retourne :
    - pivot_df : lignes = structures, colonnes = positions consensus, valeurs = AA
    - positions : liste ordonnée des positions consensus
    - long_df : table longue filtrée / dédupliquée
    """
    positions = sorted(consensus_df["gpcrdb_pos"].unique().tolist(), key=gpcrdb_sort_key)

    sub = pocket_df[pocket_df["class_simple"] == class_label].copy()
    sub = sub[sub["gpcrdb_pos"].isin(positions)].copy()

    if sub.empty:
        return pd.DataFrame(), positions, sub

    # Une seule ligne par pdb_id + gpcrdb_pos
    sub_dedup = (
        sub.groupby(["pdb_id", "gpcrdb_pos"], as_index=False)
        .agg({
            "aa": lambda x: x.mode().iloc[0] if len(x.mode()) else x.iloc[0]
        })
    )

    pivot = sub_dedup.pivot(index="pdb_id", columns="gpcrdb_pos", values="aa")
    pivot = pivot.reindex(columns=positions)
    pivot = pivot.dropna(how="all")

    return pivot, positions, sub_dedup


def make_count_matrix(pivot_df: pd.DataFrame, positions: list[str]) -> pd.DataFrame:
    """
    Produit la matrice de comptage AA par position pour logomaker.
    Index = positions consensus
    Colonnes = AA
    """
    rows = []
    for pos in positions:
        aa_counts = {aa: 0 for aa in AA_ORDER}
        if pos in pivot_df.columns:
            vals = pivot_df[pos].dropna().astype(str)
            for aa in vals:
                aa = aa.strip().upper()[:1]
                if aa in aa_counts:
                    aa_counts[aa] += 1
        rows.append(aa_counts)

    count_df = pd.DataFrame(rows, index=positions)
    return count_df


def build_frequency_table(
    pivot_df: pd.DataFrame,
    positions: list[str],
    consensus_df: pd.DataFrame,
) -> pd.DataFrame:
    meta = consensus_df.set_index("gpcrdb_pos").to_dict(orient="index")

    rows = []
    for pos in positions:
        vals = pivot_df[pos].dropna().astype(str) if pos in pivot_df.columns else pd.Series(dtype=str)
        n_obs = len(vals)

        if n_obs == 0:
            top_aa = "NA"
            freq_top = np.nan
            n_distinct = 0
            major_bio = "NA"
        else:
            vc = vals.value_counts()
            top_aa = vc.index[0]
            freq_top = vc.iloc[0] / n_obs
            n_distinct = vals.nunique()
            major_bio = major_biophys_class(vals)

        row = {
            "gpcrdb_pos": pos,
            "n_observed_structures": n_obs,
            "top_aa": top_aa,
            "freq_top_aa": freq_top,
            "n_distinct_aa": n_distinct,
            "major_biophys_class": major_bio,
        }

        if pos in meta:
            for k, v in meta[pos].items():
                if k != "gpcrdb_pos":
                    row[k] = v

        rows.append(row)

    out = pd.DataFrame(rows)
    cols_front = [
        "gpcrdb_pos",
        "n_observed_structures",
        "top_aa",
        "freq_top_aa",
        "n_distinct_aa",
        "major_biophys_class",
    ]
    other_cols = [c for c in out.columns if c not in cols_front]
    out = out[cols_front + other_cols]
    return out


def export_pseudo_sequences(pivot_df: pd.DataFrame, positions: list[str], out_tsv: Path):
    """
    Exporte une table avec :
    - pdb_id
    - pseudo_sequence (sans gaps)
    - pseudo_sequence_with_gaps
    """
    rows = []
    for pdb_id, row in pivot_df.iterrows():
        chars = []
        chars_with_gap = []
        for pos in positions:
            aa = row.get(pos, np.nan)
            if pd.isna(aa):
                chars_with_gap.append("-")
            else:
                aa = str(aa).strip().upper()[:1]
                chars.append(aa)
                chars_with_gap.append(aa)

        rows.append({
            "pdb_id": pdb_id,
            "pseudo_sequence": "".join(chars),
            "pseudo_sequence_with_gaps": "".join(chars_with_gap),
        })

    pd.DataFrame(rows).to_csv(out_tsv, sep="\t", index=False)


# ============================================================
# Plot weblogo
# ============================================================

def plot_weblogo(
    count_df: pd.DataFrame,
    class_label: str,
    positions: list[str],
    consensus_df: pd.DataFrame,
    out_png: Path,
    out_pdf: Path | None = None,
):
    if count_df.empty or count_df.values.sum() == 0:
        print(f"[WARN] weblogo non généré pour {class_label}: aucune donnée.")
        return

    plot_df = count_df.copy().reset_index(drop=True)

    seg_map = {}
    if "segment_gpcrdb" in consensus_df.columns:
        seg_map = (
            consensus_df[["gpcrdb_pos", "segment_gpcrdb"]]
            .dropna()
            .drop_duplicates(subset=["gpcrdb_pos"])
            .set_index("gpcrdb_pos")["segment_gpcrdb"]
            .to_dict()
        )

    segment_labels = [seg_map.get(pos, "") for pos in positions]

    fig_w = max(12, min(28, 0.72 * len(positions) + 7))
    fig_h = 6.4
    fig = plt.figure(figsize=(fig_w, fig_h))

    gs = fig.add_gridspec(
        nrows=1,
        ncols=2,
        width_ratios=[14, 3.8],
        left=0.06,
        right=0.97,
        top=0.90,
        bottom=0.18,
        wspace=0.08
    )

    ax = fig.add_subplot(gs[0, 0])
    ax_leg = fig.add_subplot(gs[0, 1])

    color_scheme = get_logo_color_scheme()

    logomaker.Logo(
        plot_df,
        ax=ax,
        color_scheme=color_scheme,
        shade_below=.5,
        fade_below=.5,
        stack_order='big_on_top'
    )

    ax.set_title(f"Consensus pocket weblogo — {class_label}", fontsize=16)
    ax.set_ylabel("Counts", fontsize=12)
    ax.set_xlabel("Consensus GPCRdb pocket positions", fontsize=12, labelpad=28)

    ax.set_xticks(np.arange(len(positions)))
    ax.set_xticklabels(positions, rotation=60, ha="right", fontsize=10)

    # segments sous les positions
    y_text = -0.12
    for i, seg in enumerate(segment_labels):
        ax.text(
            i,
            y_text,
            seg,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=9,
            color="dimgray",
            clip_on=False
        )

    ax.text(
        1.01,
        y_text,
        "segments",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="dimgray"
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Légende
    ax_leg.axis("off")
    legend_handles = [
        Patch(facecolor=BIOPHYS_COLORS_LOGO["hydrophobic"], edgecolor="black", label="hydrophobic (A/V/I/L/M/C)"),
        Patch(facecolor=BIOPHYS_COLORS_LOGO["aromatic"],   edgecolor="black", label="aromatic (F/W/Y)"),
        Patch(facecolor=BIOPHYS_COLORS_LOGO["positive"],   edgecolor="black", label="positive (K/R/H)"),
        Patch(facecolor=BIOPHYS_COLORS_LOGO["negative"],   edgecolor="black", label="negative (D/E)"),
        Patch(facecolor=BIOPHYS_COLORS_LOGO["polar"],      edgecolor="black", label="polar (S/T/N/Q)"),
        Patch(facecolor=BIOPHYS_COLORS_LOGO["structural"],    edgecolor="black", label="structural (G/P)"),
    ]

    ax_leg.legend(
        handles=legend_handles,
        title="Residue classes",
        loc="upper left",
        fontsize=10,
        title_fontsize=11,
        frameon=True,
        borderpad=0.7,
        labelspacing=0.6
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=250, bbox_inches="tight")
    if out_pdf is not None:
        fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Runner
# ============================================================

def run_for_one_class(
    pocket_df: pd.DataFrame,
    consensus_path: str,
    class_label: str,
    outdir: Path,
):
    class_tag = class_label.replace(" ", "_")

    consensus_df = load_consensus_positions(consensus_path)
    pivot_df, positions, _ = build_class_pocket_alignment(
        pocket_df=pocket_df,
        consensus_df=consensus_df,
        class_label=class_label,
    )

    class_out = outdir / class_tag
    class_out.mkdir(parents=True, exist_ok=True)

    if pivot_df.empty:
        print(f"[WARN] aucune donnée exploitable pour {class_label}")
        return

    # 1. matrice pivot pocket
    pivot_out = class_out / f"{class_tag}.consensus_pocket_alignment.tsv"
    pivot_df.to_csv(pivot_out, sep="\t", index=True, index_label="pdb_id")

    # 2. pseudo-séquences
    pseudo_out = class_out / f"{class_tag}.pseudo_sequences.tsv"
    export_pseudo_sequences(pivot_df, positions, pseudo_out)

    # 3. matrice comptage pour weblogo
    count_df = make_count_matrix(pivot_df, positions)
    count_out = class_out / f"{class_tag}.weblogo_count_matrix.tsv"
    count_df.to_csv(count_out, sep="\t", index=True, index_label="gpcrdb_pos")

    # 4. tableau de fréquences
    freq_df = build_frequency_table(pivot_df, positions, consensus_df)
    freq_out = class_out / f"{class_tag}.consensus_pocket_frequencies.tsv"
    freq_df.to_csv(freq_out, sep="\t", index=False)

    # 5. logo
    logo_png = class_out / f"{class_tag}.consensus_pocket_weblogo.png"
    logo_pdf = class_out / f"{class_tag}.consensus_pocket_weblogo.pdf"
    plot_weblogo(
        count_df=count_df,
        class_label=class_label,
        positions=positions,
        consensus_df=consensus_df,
        out_png=logo_png,
        out_pdf=logo_pdf,
    )

    print(f"[DONE] {class_label}")
    print(f"       alignment : {pivot_out}")
    print(f"       pseudo    : {pseudo_out}")
    print(f"       counts    : {count_out}")
    print(f"       freqs     : {freq_out}")
    print(f"       logo      : {logo_png}")


# ============================================================
# CLI
# ============================================================

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pocket_tsv", required=True)
    ap.add_argument("--consensus_a", required=True)
    ap.add_argument("--consensus_b", required=True)
    ap.add_argument("--outdir", required=True)
    return ap.parse_args()


def main():
    args = parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pocket_df = load_pocket_table(args.pocket_tsv)

    run_for_one_class(
        pocket_df=pocket_df,
        consensus_path=args.consensus_a,
        class_label="Class A",
        outdir=outdir,
    )

    run_for_one_class(
        pocket_df=pocket_df,
        consensus_path=args.consensus_b,
        class_label="Class B",
        outdir=outdir,
    )

    print(f"[DONE] All outputs written to: {outdir}")


if __name__ == "__main__":
    main()
