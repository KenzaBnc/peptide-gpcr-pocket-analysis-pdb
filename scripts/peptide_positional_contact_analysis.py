#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
peptide_positional_contact_analysis.py

Deux analyses structuralement valides des contacts peptide–GPCR.
Aucune normalisation de position peptidique n'est utilisée (cf. problème
de mélanger des peptides de 3 à 65 aa sur un axe [0,1] unique).

Analyse 1 — Profil biophysique receptor-centrique
  Quel segment GPCR est contacté par quel type de résidu peptidique
  (classe biophysique) ? Class A vs Class B.
  Sortie : contact_profile_classA_classB.png

Analyse 2 — Structure secondaire peptidique × segment GPCR
  Quelle fraction des contacts vers chaque segment GPCR provient de
  peptides hélicoïdaux (helix_fraction > 0.5) vs. en coil ?
  Sortie : dssp_segment_profile.png

Analyse 3 — Positions GPCRdb les plus contactées par classe
  Top 15 positions GPCRdb les plus contactées, coloriées par la classe
  biophysique du résidu peptidique contact.
  Sortie : gpcrdb_hotspots_classA.png, gpcrdb_hotspots_classB.png

Analyse 4 — Stratification intra-Classe A
  Class A divisée en : coil-dominant (≤ 0.5 helix) vs helix-dominant
  vs 9IQV microprotéine (traitement séparé).
  Sortie : classA_stratification.png

Table de référence pour comparaison future microprotéines :
  receptor_reference_table.tsv

Usage :
python3 scripts/peptide_positional_contact_analysis.py \\
    --contacts_tsv  run_out/biophys_annotations/peptide_contacts.contacts_pairs.tsv \\
    --sequences_tsv run_out/biophys_annotations/peptide_contacts.peptide_sequences.tsv \\
    --pocket_tsv    run_out/biophys_annotations/pocket_biophys_by_pocket.tsv \\
    --dssp_tsv      run_out/biophys_annotations/peptide_structure_features.tsv \\
    --outdir        out/positional_analysis
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ─────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────

SEGMENT_ORDER = ["TM1", "TM2", "TM3", "TM4", "TM5", "TM6", "TM7",
                 "ECL1", "ECL2", "ECL3", "N-term"]

BIOPHYS_ORDER = ["aromatic", "hydrophobic", "polar", "positive",
                 "negative", "structural", "other"]
BIOPHYS_COLORS = {
    "aromatic":    "#2ca02c",
    "hydrophobic": "#1f77b4",
    "polar":       "#ff7f0e",
    "positive":    "#d62728",
    "negative":    "#9467bd",
    "structural":     "#8c564b",
    "other":       "#7f7f7f",
}

# 9IQV : microprotéine knottin à 65 aa liée à un récepteur Class A (alpha-1A)
# Traitée séparément comme référence pour la comparaison avec les microprotéines
MICROPROTEIN_ID = "9IQV"

CMAP_A = "Blues"
CMAP_B = "Oranges"


# ─────────────────────────────────────────────────────────────
# ARGUMENTS
# ─────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(
        description="Analyse des contacts peptide–GPCR (approche receptor-centrique + DSSP)."
    )
    ap.add_argument("--contacts_tsv",  required=True)
    ap.add_argument("--sequences_tsv", required=True)
    ap.add_argument("--pocket_tsv",    required=True)
    ap.add_argument("--dssp_tsv",      required=True,
                    help="peptide_structure_features.tsv (helix_fraction par peptide)")
    ap.add_argument("--outdir",        required=True)
    return ap.parse_args()


# ─────────────────────────────────────────────────────────────
# CHARGEMENT ET PRÉPARATION
# ─────────────────────────────────────────────────────────────

def load_and_prepare(contacts_tsv, sequences_tsv, pocket_tsv, dssp_tsv) -> pd.DataFrame:
    contacts = pd.read_csv(contacts_tsv, sep="\t", dtype=str)
    seqs     = pd.read_csv(sequences_tsv, sep="\t", dtype=str)
    pocket   = pd.read_csv(pocket_tsv,   sep="\t", dtype=str)
    dssp     = pd.read_csv(dssp_tsv,     sep="\t", dtype=str)

    for df in [contacts, seqs, pocket, dssp]:
        df["pdb_id"] = df["pdb_id"].str.upper().str.strip()

    # Classe GPCR
    pocket["class_simple"] = pocket["gpcr_class"].str.extract(r"(Class [AB])")
    class_map = pocket[["pdb_id", "peptide_chain", "class_simple", "gpcr_class"]].drop_duplicates()

    # Longueur peptide
    seqs["peptide_length"] = pd.to_numeric(seqs["peptide_length"], errors="coerce")
    len_map = seqs[["pdb_id", "peptide_chain", "peptide_length"]].drop_duplicates()

    # DSSP agrégé (helix_fraction par peptide)
    dssp["helix_fraction"] = pd.to_numeric(dssp["helix_fraction"], errors="coerce")
    dssp_map = dssp[["pdb_id", "peptide_chain", "helix_fraction"]].drop_duplicates()

    # Jointures
    contacts = contacts.merge(class_map, on=["pdb_id", "peptide_chain"], how="left")
    contacts = contacts.merge(len_map,   on=["pdb_id", "peptide_chain"], how="left")
    contacts = contacts.merge(dssp_map,  on=["pdb_id", "peptide_chain"], how="left")

    contacts["peptide_length"]    = pd.to_numeric(contacts["peptide_length"], errors="coerce")
    contacts["peptide_pos_index"] = pd.to_numeric(contacts["peptide_pos_index"], errors="coerce")
    contacts["min_dist"]          = pd.to_numeric(contacts["min_dist"], errors="coerce")

    # Type structurel du peptide (niveau peptide, pas résidu)
    # helix_fraction > 0.5 → peptide hélicoïdal dominant
    contacts["peptide_ss_type"] = contacts["helix_fraction"].apply(
        lambda x: "helix-dominant" if pd.notna(x) and x > 0.5 else "coil-dominant"
    )

    # Filtre segments connus
    contacts = contacts[contacts["segment"].isin(SEGMENT_ORDER)].copy()
    contacts = contacts.dropna(subset=["class_simple"]).copy()

    return contacts


def split_groups(df: pd.DataFrame):
    """Retourne les sous-ensembles d'analyse."""
    iqv  = df[df["pdb_id"] == MICROPROTEIN_ID].copy()
    df_ex = df[df["pdb_id"] != MICROPROTEIN_ID].copy()
    classA = df_ex[df_ex["class_simple"] == "Class A"].copy()
    classB = df_ex[df_ex["class_simple"] == "Class B"].copy()
    # Stratification intra-Class A par type DSSP
    classA_coil  = classA[classA["peptide_ss_type"] == "coil-dominant"].copy()
    classA_helix = classA[classA["peptide_ss_type"] == "helix-dominant"].copy()
    return classA, classB, classA_coil, classA_helix, iqv


# ─────────────────────────────────────────────────────────────
# ANALYSE 1 — Profil biophysique receptor-centrique
# ─────────────────────────────────────────────────────────────

def compute_segment_biophys(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pour chaque segment, calcule la distribution (%) des classes biophysiques
    des résidus peptidiques en contact.
    """
    grp = (
        df.groupby(["segment", "peptide_class"])
          .size()
          .reset_index(name="n")
    )
    tot = grp.groupby("segment")["n"].transform("sum")
    grp["pct"] = 100.0 * grp["n"] / tot

    pivot = (
        grp.pivot(index="segment", columns="peptide_class", values="pct")
           .fillna(0)
    )
    for col in BIOPHYS_ORDER:
        if col not in pivot.columns:
            pivot[col] = 0.0
    pivot = pivot[BIOPHYS_ORDER]
    pivot = pivot.reindex(
        [s for s in SEGMENT_ORDER if s in pivot.index]
    )
    return pivot


def plot_contact_profile(classA: pd.DataFrame, classB: pd.DataFrame,
                         outdir: Path):
    """
    Figure principale : profil biophysique receptor-centrique.
    Barres horizontales empilées (segment en Y, classes biophys en X).
    Class A (gauche) vs Class B (droite).
    """
    pivotA = compute_segment_biophys(classA)
    pivotB = compute_segment_biophys(classB)

    nA_str = classA["pdb_id"].nunique()
    nB_str = classB["pdb_id"].nunique()
    nA_c   = len(classA)
    nB_c   = len(classB)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=False)

    for ax, pivot, label, n_str, n_cont, color in [
        (axes[0], pivotA, "Class A (rhodopsin-like)", nA_str, nA_c, "#2c5f9e"),
        (axes[1], pivotB, "Class B (secretin-like)",  nB_str, nB_c, "#a63603"),
    ]:
        segs = list(pivot.index)[::-1]  # TM1 en haut
        y = np.arange(len(segs))
        left = np.zeros(len(segs))

        for cls in BIOPHYS_ORDER:
            vals = np.array([pivot.loc[s, cls] if s in pivot.index else 0.0
                             for s in segs])
            ax.barh(y, vals, left=left,
                    color=BIOPHYS_COLORS[cls], label=cls,
                    height=0.75, edgecolor="white", linewidth=0.5)
            for yi, (v, l) in enumerate(zip(vals, left)):
                if v >= 8:
                    ax.text(l + v / 2, yi, f"{v:.0f}%",
                            ha="center", va="center", fontsize=8,
                            color="white", fontweight="bold")
            left += vals

        ax.set_yticks(y)
        ax.set_yticklabels(segs, fontsize=11)
        ax.set_xlim(0, 105)
        ax.set_xlabel("% des contacts par classe biophysique", fontsize=11)
        ax.set_title(f"{label}\nn={n_str} structures · {n_cont} contacts",
                     fontsize=12, fontweight="bold", color=color)
        ax.axvline(50, color="grey", linewidth=0.7, linestyle="--", alpha=0.5)

    legend_patches = [
        mpatches.Patch(color=BIOPHYS_COLORS[c], label=c) for c in BIOPHYS_ORDER
    ]
    fig.legend(handles=legend_patches, loc="lower center",
               ncol=len(BIOPHYS_ORDER), fontsize=10, frameon=False,
               title="Classe biophysique du résidu peptidique", title_fontsize=10,
               bbox_to_anchor=(0.5, -0.04))

    fig.suptitle(
        "Profil biophysique receptor-centrique des contacts peptide–GPCR\n"
        "Chaque barre = distribution des classes des résidus peptidiques "
        "contactant ce segment",
        fontsize=13, y=1.01
    )
    fig.tight_layout()

    out = outdir / "contact_profile_classA_classB.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[DONE] {out}")


# ─────────────────────────────────────────────────────────────
# ANALYSE 2 — Structure secondaire peptidique × segment GPCR
# ─────────────────────────────────────────────────────────────

def compute_dssp_segment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pour chaque segment, fraction des contacts provenant de peptides
    hélicoïdaux vs coil. Valeurs en % du total de contacts vers ce segment.
    """
    grp = (
        df.groupby(["segment", "peptide_ss_type"])
          .size()
          .reset_index(name="n")
    )
    tot = grp.groupby("segment")["n"].transform("sum")
    grp["pct"] = 100.0 * grp["n"] / tot

    pivot = (
        grp.pivot(index="segment", columns="peptide_ss_type", values="pct")
           .fillna(0)
    )
    for col in ["helix-dominant", "coil-dominant"]:
        if col not in pivot.columns:
            pivot[col] = 0.0
    pivot = pivot[["helix-dominant", "coil-dominant"]]
    pivot = pivot.reindex([s for s in SEGMENT_ORDER if s in pivot.index])
    return pivot


def plot_dssp_segment_profile(classA: pd.DataFrame, classB: pd.DataFrame,
                               outdir: Path):
    """
    Figure : fraction des contacts par segment issus de peptides hélicoïdaux
    ou en coil. Permet de voir quels segments GPCR attirent des peptides
    structurés vs désordonnés.
    """
    pivotA = compute_dssp_segment(classA)
    pivotB = compute_dssp_segment(classB)

    nA_str = classA["pdb_id"].nunique()
    nB_str = classB["pdb_id"].nunique()

    dssp_colors = {
        "helix-dominant": "#c7522a",
        "coil-dominant":  "#74a892",
    }

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    for ax, pivot, label, n_str, cmap_color in [
        (axes[0], pivotA, "Class A (ex-9IQV)", nA_str, "#2c5f9e"),
        (axes[1], pivotB, "Class B",           nB_str, "#a63603"),
    ]:
        segs = list(pivot.index)[::-1]
        y = np.arange(len(segs))
        width = 0.38

        helix_vals = np.array([
            pivot.loc[s, "helix-dominant"] if s in pivot.index else 0.0
            for s in segs
        ])
        coil_vals = np.array([
            pivot.loc[s, "coil-dominant"] if s in pivot.index else 0.0
            for s in segs
        ])

        # Nombre brut de contacts pour annotation
        n_contacts_by_seg = classA.groupby("segment").size() if label.startswith("Class A") else classB.groupby("segment").size()

        ax.barh(y + width / 2, helix_vals, height=width,
                color=dssp_colors["helix-dominant"], label="Peptide hélicoïdal\n(helix_fraction > 0.5)",
                edgecolor="white", linewidth=0.5)
        ax.barh(y - width / 2, coil_vals, height=width,
                color=dssp_colors["coil-dominant"], label="Peptide en coil\n(helix_fraction ≤ 0.5)",
                edgecolor="white", linewidth=0.5)

        # Annotation des valeurs
        for yi, (h, c, seg) in enumerate(zip(helix_vals, coil_vals, segs)):
            n_tot = n_contacts_by_seg.get(seg, 0)
            if h > 5:
                ax.text(h / 2, yi + width / 2, f"{h:.0f}%",
                        ha="center", va="center", fontsize=8, color="white", fontweight="bold")
            if c > 5:
                ax.text(c / 2, yi - width / 2, f"{c:.0f}%",
                        ha="center", va="center", fontsize=8, color="white", fontweight="bold")
            ax.text(102, yi, f"n={n_tot}", ha="left", va="center", fontsize=8, color="#555555")

        ax.set_yticks(y)
        ax.set_yticklabels(segs, fontsize=11)
        ax.set_xlim(0, 115)
        ax.set_xlabel("% des contacts vers ce segment", fontsize=11)
        ax.set_title(f"{label}\nn={n_str} structures",
                     fontsize=12, fontweight="bold", color=cmap_color)
        ax.axvline(50, color="grey", linewidth=0.7, linestyle="--", alpha=0.5)
        ax.legend(loc="lower right", fontsize=9, frameon=True)

    fig.suptitle(
        "Structure secondaire du peptide × segment GPCR\n"
        "Fraction des contacts vers chaque segment issus de peptides hélicoïdaux vs. en coil",
        fontsize=13, y=1.01
    )
    fig.tight_layout()

    out = outdir / "dssp_segment_profile.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[DONE] {out}")


# ─────────────────────────────────────────────────────────────
# ANALYSE 3 — Positions GPCRdb les plus contactées
# ─────────────────────────────────────────────────────────────

def plot_gpcrdb_hotspots(df: pd.DataFrame, class_label: str,
                          outdir: Path, top_n: int = 20):
    """
    Top N positions GPCRdb les plus contactées.
    Pour chaque position, la couleur reflète la classe biophysique
    dominante du résidu peptidique qui la contacte.
    Chiffre annoté = nombre total de contacts.
    """
    valid = df.dropna(subset=["gpcrdb_pos"]).copy()
    valid = valid[valid["gpcrdb_pos"] != ""]

    if valid.empty:
        print(f"[WARN] Aucune position GPCRdb pour {class_label}")
        return

    # Comptage total
    top_pos = (
        valid.groupby(["gpcrdb_pos", "segment"])
             .size()
             .reset_index(name="n_total")
             .sort_values("n_total", ascending=False)
             .head(top_n)
    )

    # Biophysique dominante pour chaque position
    dominant_biophys = (
        valid.groupby(["gpcrdb_pos", "peptide_class"])
             .size()
             .reset_index(name="n")
    )
    dominant_biophys = (
        dominant_biophys.sort_values("n", ascending=False)
                        .drop_duplicates("gpcrdb_pos")
                        [["gpcrdb_pos", "peptide_class"]]
                        .rename(columns={"peptide_class": "dominant_class"})
    )

    top_pos = top_pos.merge(dominant_biophys, on="gpcrdb_pos", how="left")
    top_pos["color"] = top_pos["dominant_class"].map(BIOPHYS_COLORS).fillna("#7f7f7f")

    # Tri : segment puis GPCRdb position
    top_pos = top_pos.sort_values(["segment", "n_total"], ascending=[True, False])

    fig, ax = plt.subplots(figsize=(11, max(6, len(top_pos) * 0.4)))

    y = np.arange(len(top_pos))
    labels = [
        f"{row['gpcrdb_pos']}  [{row['segment']}]"
        for _, row in top_pos.iterrows()
    ]

    bars = ax.barh(y, top_pos["n_total"].values,
                   color=top_pos["color"].values,
                   edgecolor="white", linewidth=0.5, height=0.75)

    for yi, (bar, row) in enumerate(zip(bars, top_pos.itertuples())):
        n = row.n_total
        ax.text(n + 0.3, yi, str(n), va="center", fontsize=9)
        if n >= 4:
            ax.text(n / 2, yi, row.dominant_class,
                    ha="center", va="center", fontsize=8,
                    color="white", fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Nombre de contacts résidu–résidu", fontsize=11)
    ax.set_title(
        f"Top {top_n} positions GPCRdb les plus contactées — {class_label}\n"
        f"Couleur = classe biophysique dominante du résidu peptidique",
        fontsize=12, fontweight="bold"
    )

    legend_patches = [
        mpatches.Patch(color=BIOPHYS_COLORS[c], label=c)
        for c in BIOPHYS_ORDER if c in top_pos["dominant_class"].values
    ]
    ax.legend(handles=legend_patches, loc="lower right",
              fontsize=9, frameon=True, title="Classe biophysique")

    fig.tight_layout()
    out = outdir / f"gpcrdb_hotspots_{class_label.replace(' ', '_')}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[DONE] {out}")


# ─────────────────────────────────────────────────────────────
# ANALYSE 4 — Stratification intra-Classe A
# ─────────────────────────────────────────────────────────────

def plot_classA_stratification(classA_coil: pd.DataFrame,
                                classA_helix: pd.DataFrame,
                                iqv: pd.DataFrame,
                                outdir: Path):
    """
    Stratification biologique de la Classe A en trois groupes :
    - Coil-dominant (helix_fraction ≤ 0.5) : peptides courts désordonnés
    - Helix-dominant (helix_fraction > 0.5) : peptides allostériques structurés
    - 9IQV microprotéine knottin (65 aa, référence)
    Profil biophysique par segment pour chaque groupe.
    """
    groups = [
        (classA_coil,  "Coil-dominant\n(helix ≤ 0.5)", "#74a892"),
        (classA_helix, "Helix-dominant\n(helix > 0.5)", "#c7522a"),
        (iqv,          "9IQV microprotéine\n(knottin, 65 aa)", "#7b2d8b"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    for ax, (sub, label, color) in zip(axes, groups):
        if sub.empty:
            ax.set_visible(False)
            continue

        pivot = compute_segment_biophys(sub)
        segs = list(pivot.index)[::-1]
        y = np.arange(len(segs))
        left = np.zeros(len(segs))

        for cls in BIOPHYS_ORDER:
            vals = np.array([pivot.loc[s, cls] if s in pivot.index else 0.0
                             for s in segs])
            ax.barh(y, vals, left=left,
                    color=BIOPHYS_COLORS[cls], label=cls,
                    height=0.75, edgecolor="white", linewidth=0.5)
            for yi, (v, l) in enumerate(zip(vals, left)):
                if v >= 10:
                    ax.text(l + v / 2, yi, f"{v:.0f}%",
                            ha="center", va="center", fontsize=8,
                            color="white", fontweight="bold")
            left += vals

        n_str = sub["pdb_id"].nunique()
        n_c   = len(sub)
        ax.set_yticks(y)
        ax.set_yticklabels(segs, fontsize=10)
        ax.set_xlim(0, 105)
        ax.set_xlabel("% des contacts", fontsize=10)
        ax.set_title(f"{label}\nn={n_str} str. · {n_c} contacts",
                     fontsize=11, fontweight="bold", color=color)
        ax.axvline(50, color="grey", linewidth=0.7, linestyle="--", alpha=0.5)

    legend_patches = [
        mpatches.Patch(color=BIOPHYS_COLORS[c], label=c) for c in BIOPHYS_ORDER
    ]
    fig.legend(handles=legend_patches, loc="lower center",
               ncol=len(BIOPHYS_ORDER), fontsize=10, frameon=False,
               title="Classe biophysique", title_fontsize=10,
               bbox_to_anchor=(0.5, -0.04))

    fig.suptitle(
        "Stratification intra-Classe A : structure secondaire du peptide\n"
        "Coil-dominant · Helix-dominant · 9IQV microprotéine (référence)",
        fontsize=13, y=1.01
    )
    fig.tight_layout()

    out = outdir / "classA_stratification.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[DONE] {out}")


# ─────────────────────────────────────────────────────────────
# TABLE DE RÉFÉRENCE — pour comparaison microprotéines futures
# ─────────────────────────────────────────────────────────────

def export_reference_table(classA: pd.DataFrame, classB: pd.DataFrame,
                            iqv: pd.DataFrame, outdir: Path):
    """
    Table de référence pour la future comparaison avec les microprotéines.
    Chaque ligne = (gpcr_class, group, segment, peptide_class, n_contacts, pct)
    """
    rows = []
    groups = [
        (classA[classA["peptide_ss_type"] == "coil-dominant"],  "Class A", "coil-dominant"),
        (classA[classA["peptide_ss_type"] == "helix-dominant"], "Class A", "helix-dominant"),
        (classB,                                                "Class B", "all"),
        (iqv,                                                   "9IQV",    "microprotein"),
    ]

    for sub, gpcr_cls, group in groups:
        if sub.empty:
            continue
        n_structures = sub["pdb_id"].nunique()
        tot_contacts = len(sub)

        # Par segment × classe biophysique
        grp = (
            sub.groupby(["segment", "peptide_class"])
               .size()
               .reset_index(name="n_contacts")
        )
        seg_tot = sub.groupby("segment").size().rename("seg_total")
        grp = grp.merge(seg_tot, on="segment")

        for _, row in grp.iterrows():
            rows.append({
                "gpcr_class":     gpcr_cls,
                "peptide_group":  group,
                "n_structures":   n_structures,
                "segment":        row["segment"],
                "peptide_class":  row["peptide_class"],
                "n_contacts":     row["n_contacts"],
                "pct_in_segment": round(100.0 * row["n_contacts"] / row["seg_total"], 1),
                "seg_total":      row["seg_total"],
                "tot_contacts":   tot_contacts,
            })

    # Positions GPCRdb les plus contactées par classe
    gpcrdb_rows = []
    for sub, gpcr_cls, group in groups:
        if sub.empty:
            continue
        valid = sub.dropna(subset=["gpcrdb_pos"])
        valid = valid[valid["gpcrdb_pos"] != ""]
        top = (
            valid.groupby(["gpcrdb_pos", "segment", "peptide_class"])
                 .size()
                 .reset_index(name="n_contacts")
                 .sort_values("n_contacts", ascending=False)
        )
        top["gpcr_class"]    = gpcr_cls
        top["peptide_group"] = group
        gpcrdb_rows.append(top)

    ref = pd.DataFrame(rows)
    ref_gpcrdb = pd.concat(gpcrdb_rows, ignore_index=True) if gpcrdb_rows else pd.DataFrame()

    out_ref = outdir / "receptor_reference_table.tsv"
    ref.to_csv(out_ref, sep="\t", index=False)
    print(f"[DONE] {out_ref}")

    if not ref_gpcrdb.empty:
        out_gpcrdb = outdir / "gpcrdb_positions_reference_table.tsv"
        ref_gpcrdb.to_csv(out_gpcrdb, sep="\t", index=False)
        print(f"[DONE] {out_gpcrdb}")


# ─────────────────────────────────────────────────────────────
# RAPPORT TEXTE — résumé statistique
# ─────────────────────────────────────────────────────────────

def write_summary(classA: pd.DataFrame, classB: pd.DataFrame,
                  iqv: pd.DataFrame, outdir: Path):
    lines = []
    lines.append("=== RÉSUMÉ ANALYSE POSITIONNELLE CONTACTS PEPTIDE–GPCR ===\n")

    for sub, label in [(classA, "Class A (ex-9IQV)"), (classB, "Class B"), (iqv, "9IQV microprotéine")]:
        if sub.empty:
            continue
        lines.append(f"\n--- {label} ---")
        lines.append(f"  Structures : {sub['pdb_id'].nunique()}")
        lines.append(f"  Contacts   : {len(sub)}")
        lines.append(f"  Segments   : {', '.join(sub['segment'].value_counts().index[:5].tolist())}")
        top_bp = sub["peptide_class"].value_counts(normalize=True).head(3)
        lines.append(f"  Top classes biophysiques : " +
                     ", ".join(f"{k} ({v*100:.0f}%)" for k, v in top_bp.items()))

        dssp_split = sub.groupby("pdb_id")["helix_fraction"].first()
        n_helix = (dssp_split > 0.5).sum()
        n_coil  = (dssp_split <= 0.5).sum()
        lines.append(f"  Peptides hélicoïdaux (>0.5) : {n_helix} · coil : {n_coil}")

        lens = sub.groupby("pdb_id")["peptide_length"].first()
        lines.append(f"  Longueurs : {int(lens.min())}–{int(lens.max())} aa")

    out = outdir / "analysis_summary.txt"
    out.write_text("\n".join(lines))
    print(f"[DONE] {out}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Chargement des données...")
    df = load_and_prepare(
        args.contacts_tsv, args.sequences_tsv,
        args.pocket_tsv, args.dssp_tsv
    )

    classA, classB, classA_coil, classA_helix, iqv = split_groups(df)

    n_by_class = df.groupby("class_simple")["pdb_id"].nunique().to_dict()
    print(f"[INFO] {len(df)} contacts totaux")
    print(f"[INFO] Structures par classe : {n_by_class}")
    print(f"[INFO] 9IQV : {len(iqv)} contacts")
    print(f"[INFO] Class A coil-dominant  : {classA_coil['pdb_id'].nunique()} str. · {len(classA_coil)} contacts")
    print(f"[INFO] Class A helix-dominant : {classA_helix['pdb_id'].nunique()} str. · {len(classA_helix)} contacts")

    # Analyse 1 : profil biophysique receptor-centrique
    print("\n[Analyse 1] Profil biophysique receptor-centrique...")
    plot_contact_profile(classA, classB, outdir)

    # Analyse 2 : DSSP × segment
    print("\n[Analyse 2] Structure secondaire × segment...")
    plot_dssp_segment_profile(classA, classB, outdir)

    # Analyse 3 : positions GPCRdb
    print("\n[Analyse 3] Positions GPCRdb hotspots...")
    plot_gpcrdb_hotspots(classA, "Class_A", outdir, top_n=20)
    plot_gpcrdb_hotspots(classB, "Class_B", outdir, top_n=15)
    if not iqv.empty:
        plot_gpcrdb_hotspots(iqv, "9IQV_microprotein", outdir, top_n=15)

    # Analyse 4 : stratification Classe A
    print("\n[Analyse 4] Stratification intra-Classe A...")
    plot_classA_stratification(classA_coil, classA_helix, iqv, outdir)

    # Tables de référence
    print("\n[Export] Tables de référence...")
    export_reference_table(classA, classB, iqv, outdir)

    # Résumé texte
    write_summary(classA, classB, iqv, outdir)

    print("\n[DONE] Analyse positionnelle terminée.")


if __name__ == "__main__":
    main()
