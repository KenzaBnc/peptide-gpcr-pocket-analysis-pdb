#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
peptide_spatial_variability.py

Analyse la variabilité spatiale des peptides après alignement des récepteurs.

Volet 1 — Pose globale des peptides
  - Centre de masse de chaque peptide (extrait des CIF après alignement)
  - Matrice de distances entre centres de masse
  - PCA 2D des poses peptidiques
  → figures : heatmap distances + scatter PCA

Volet 2 — Variabilité spatiale des résidus peptidiques
  - Centre de la poche consensus extrait depuis la structure de référence
    (dans le même repère que les coordonnées alignées)
  - Pour chaque résidu peptidique Cα :
      * depth_to_pocket_center = distance au centre de poche
      * spatial_deviation      = distance au centroïde global de tous les Cα peptidiques
        alignés (toutes structures confondues)
  → figures : scatter (depth vs spatial deviation) + boxplot 3 zones

Entrées :
  --pocket_tsv     : TSV pocket avec target_chain, peptide_chain, target_resnum, gpcr_class
  --consensus_tsv  : TSV consensus (gpcrdb_pos, ...)
  --pdb_dir        : dossier contenant les {pdb_id}.cif
  --class_label    : "Class A" ou "Class B"
  --reference_pdb  : PDB id de référence pour l'alignement
  --outdir         : dossier de sortie

Dépendances :
  pip install pandas numpy matplotlib seaborn scipy biopython scikit-learn
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

from Bio.PDB import MMCIFParser, Superimposer
from Bio.PDB.PDBExceptions import PDBConstructionWarning

warnings.filterwarnings("ignore", category=PDBConstructionWarning)


# ============================================================
# Constants
# ============================================================

DEPTH_BINS = [0, 10, 20, np.inf]
DEPTH_LABELS = ["Inner pocket\n(< 10 Å)", "Intermediate\n(10–20 Å)", "Outer\n(> 20 Å)"]
DEPTH_COLORS = ["#e31a1c", "#ff7f00", "#1f78b4"]

PEPTIDE_PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9a6324", "#fffac8", "#800000", "#aaffc3",
    "#808000", "#ffd8b1", "#000075", "#a9a9a9", "#666666",
]


# ============================================================
# CLI
# ============================================================

def parse_args():
    ap = argparse.ArgumentParser(
        description="Peptide spatial variability analysis (volet 1 + volet 2)"
    )
    ap.add_argument("--pocket_tsv", required=True,
                    help="TSV pocket with target_chain, peptide_chain, target_resnum, gpcr_class")
    ap.add_argument("--consensus_tsv", required=True,
                    help="Consensus TSV with gpcrdb_pos column")
    ap.add_argument("--pdb_dir", required=True,
                    help="Directory containing {pdb_id}.cif files")
    ap.add_argument("--class_label", required=True,
                    help='e.g. "Class A"')
    ap.add_argument("--reference_pdb", required=True,
                    help="Reference PDB id for receptor alignment")
    ap.add_argument("--outdir", required=True)
    return ap.parse_args()


# ============================================================
# Helpers
# ============================================================

def norm_pdb(x) -> str:
    return str(x).strip().lower()


def simplify_class(s: str) -> str:
    s = str(s)
    if "Class A" in s or "class a" in s.lower():
        return "Class A"
    if "Class B" in s or "class b" in s.lower():
        return "Class B"
    return "Other"


def get_ca_atoms(chain):
    """Return list of (resseq, CA_atom) for a chain, sorted by resseq."""
    cas = []
    for res in chain.get_residues():
        if res.get_id()[0] != " ":
            continue
        if "CA" in res:
            cas.append((res.get_id()[1], res["CA"]))
    return sorted(cas, key=lambda x: x[0])


def atoms_to_coords(atom_list) -> np.ndarray:
    return np.array([a.get_vector().get_array() for a in atom_list], dtype=float)


def find_gpcrdb_column(df: pd.DataFrame) -> str:
    for col in ["gpcrdb_pos", "gpcrdb", "gpcrdb_display_generic_number"]:
        if col in df.columns:
            return col
    raise SystemExit("[ERROR] No gpcrdb_pos/gpcrdb/gpcrdb_display_generic_number column found.")


def extract_chain(model, chain_id: str):
    for ch in model.get_chains():
        if ch.id == chain_id:
            return ch
    return None


# ============================================================
# Step 1 — Load metadata
# ============================================================

def load_metadata(pocket_tsv: str, class_label: str) -> pd.DataFrame:
    pocket = pd.read_csv(pocket_tsv, sep="\t", dtype=str)
    pocket["pdb_id_norm"] = pocket["pdb_id"].map(norm_pdb)

    required = {"target_chain", "peptide_chain", "target_resnum"}
    missing = required - set(pocket.columns)
    if missing:
        raise SystemExit(f"[ERROR] Missing columns in pocket TSV: {sorted(missing)}")

    pocket["target_chain"] = pocket["target_chain"].astype(str).str.strip()
    pocket["peptide_chain"] = pocket["peptide_chain"].astype(str).str.strip()
    pocket["target_resnum"] = pd.to_numeric(pocket["target_resnum"], errors="coerce")

    if "gpcr_class" in pocket.columns:
        pocket["class_simple"] = pocket["gpcr_class"].map(simplify_class)
        pocket = pocket[pocket["class_simple"] == class_label].copy()

    if pocket.empty:
        raise SystemExit(f"[ERROR] No rows found for class {class_label} in pocket TSV")

    return pocket


# ============================================================
# Step 2 — Load structures and align receptors
# ============================================================

def load_structures_and_align(
    struct_info: pd.DataFrame,
    pdb_dir: str,
    reference_pdb: str,
):
    parser = MMCIFParser(QUIET=True)
    structures = {}
    receptor_cas = {}

    ref_id = norm_pdb(reference_pdb)

    print(f"[INFO] Loading {len(struct_info)} structures...")

    for _, row in struct_info.iterrows():
        pdb_id = row["pdb_id"]
        tchain = row["target_chain"]

        cif_path = Path(pdb_dir) / f"{pdb_id}.cif"
        if not cif_path.exists():
            print(f"[WARN] CIF not found: {cif_path}, skipping")
            continue

        try:
            struct = parser.get_structure(pdb_id, str(cif_path))
        except Exception as e:
            print(f"[WARN] Could not parse {cif_path}: {e}, skipping")
            continue

        model = struct[0]
        chain = extract_chain(model, tchain)
        if chain is None:
            print(f"[WARN] Receptor chain {tchain} not found in {pdb_id}, skipping")
            continue

        cas = get_ca_atoms(chain)
        if len(cas) < 10:
            print(f"[WARN] Too few receptor CA atoms in {pdb_id}, skipping")
            continue

        structures[pdb_id] = struct
        receptor_cas[pdb_id] = cas

    if ref_id not in structures:
        raise SystemExit(f"[ERROR] Reference structure {ref_id} could not be loaded")

    ref_cas_full = receptor_cas[ref_id]

    print(f"[INFO] Aligning structures onto reference {ref_id}...")

    for pdb_id, struct in structures.items():
        if pdb_id == ref_id:
            continue

        mov_cas_full = receptor_cas[pdb_id]

        # Better than naive "first n atoms": match on common residue numbers
        ref_map = {resseq: atom for resseq, atom in ref_cas_full}
        mov_map = {resseq: atom for resseq, atom in mov_cas_full}
        common_resseq = sorted(set(ref_map) & set(mov_map))

        if len(common_resseq) < 10:
            print(f"[WARN] Too few common receptor CA residues for {pdb_id} ({len(common_resseq)}), skipping alignment")
            continue

        ref_atoms = [ref_map[r] for r in common_resseq]
        mov_atoms = [mov_map[r] for r in common_resseq]

        try:
            sup = Superimposer()
            sup.set_atoms(ref_atoms, mov_atoms)
            sup.apply(struct[0].get_atoms())
        except Exception as e:
            print(f"[WARN] Alignment failed for {pdb_id}: {e}")

    return structures


# ============================================================
# Step 3 — Pocket center in aligned reference frame
# ============================================================

def extract_consensus_pocket_center_from_reference(
    pocket_df: pd.DataFrame,
    consensus_tsv: str,
    structures: dict,
    reference_pdb: str,
) -> np.ndarray:
    ref_id = norm_pdb(reference_pdb)
    ref_rows = pocket_df[pocket_df["pdb_id_norm"] == ref_id].copy()
    if ref_rows.empty:
        raise SystemExit(f"[ERROR] No pocket rows found for reference {ref_id}")

    gp_col = find_gpcrdb_column(ref_rows)
    ref_rows["gpcrdb_pos_norm"] = ref_rows[gp_col].astype(str).str.strip()

    consensus = pd.read_csv(consensus_tsv, sep="\t", dtype=str)
    if "gpcrdb_pos" not in consensus.columns:
        raise SystemExit("[ERROR] gpcrdb_pos column missing from consensus TSV")
    consensus_positions = set(consensus["gpcrdb_pos"].dropna().astype(str).str.strip())

    ref_cons = ref_rows[ref_rows["gpcrdb_pos_norm"].isin(consensus_positions)].copy()
    ref_cons = ref_cons.dropna(subset=["target_resnum"])

    if ref_cons.empty:
        raise SystemExit("[ERROR] No consensus pocket residues matched on reference structure")

    # Retrieve coordinates directly from aligned reference structure
    struct = structures[ref_id]
    model = struct[0]
    ref_chain_id = ref_cons["target_chain"].astype(str).str.strip().iloc[0]
    ref_chain = extract_chain(model, ref_chain_id)
    if ref_chain is None:
        raise SystemExit(f"[ERROR] Reference chain {ref_chain_id} not found in reference structure")

    target_resnums = sorted(set(int(x) for x in ref_cons["target_resnum"].dropna().astype(int).tolist()))

    coords = []
    missing = []
    for res in ref_chain.get_residues():
        if res.get_id()[0] != " ":
            continue
        resseq = res.get_id()[1]
        if resseq in target_resnums and "CA" in res:
            coords.append(res["CA"].get_vector().get_array())

    found_resnums = set()
    for res in ref_chain.get_residues():
        if res.get_id()[0] != " ":
            continue
        if "CA" in res:
            found_resnums.add(res.get_id()[1])

    for r in target_resnums:
        if r not in found_resnums:
            missing.append(r)

    if not coords:
        raise SystemExit("[ERROR] No CA coordinates found for consensus pocket residues on reference structure")

    center = np.mean(np.array(coords, dtype=float), axis=0)
    print(f"[INFO] Consensus pocket center from aligned reference ({len(coords)} residues): "
          f"({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
    if missing:
        print(f"[WARN] Missing reference pocket residue CA atoms for resnums: {missing}")

    return center


# ============================================================
# Step 4 — Extract peptide CA after alignment
# ============================================================

def extract_peptide_coordinates(
    struct_info: pd.DataFrame,
    structures: dict,
):
    peptide_ca = {}
    peptide_df_rows = []

    for _, row in struct_info.iterrows():
        pdb_id = row["pdb_id"]
        pchain = row["peptide_chain"]

        if pdb_id not in structures:
            continue

        model = structures[pdb_id][0]
        chain = extract_chain(model, pchain)
        if chain is None:
            print(f"[WARN] Peptide chain {pchain} not found in {pdb_id}")
            continue

        cas = get_ca_atoms(chain)
        if not cas:
            continue

        coords = atoms_to_coords([a for _, a in cas])
        peptide_ca[pdb_id] = coords

        for i, (resseq, atom) in enumerate(cas):
            xyz = atom.get_vector().get_array()
            peptide_df_rows.append({
                "pdb_id": pdb_id,
                "resseq": int(resseq),
                "pos_in_peptide": i + 1,
                "x": float(xyz[0]),
                "y": float(xyz[1]),
                "z": float(xyz[2]),
            })

    peptide_df = pd.DataFrame(peptide_df_rows)
    print(f"[INFO] Extracted peptide Cα for {len(peptide_ca)} structures ({len(peptide_df)} residues total)")

    return peptide_ca, peptide_df


# ============================================================
# VOLET 1 — Global peptide pose
# ============================================================

def compute_peptide_centers(peptide_ca: dict) -> pd.DataFrame:
    rows = []
    for pdb_id, coords in peptide_ca.items():
        cm = coords.mean(axis=0)
        rows.append({
            "pdb_id": pdb_id,
            "cx": cm[0],
            "cy": cm[1],
            "cz": cm[2],
            "n_res": len(coords),
        })
    return pd.DataFrame(rows)


def plot_volet1(peptide_ca: dict, outdir: Path, class_label: str):
    centers_df = compute_peptide_centers(peptide_ca)
    if centers_df.empty or len(centers_df) < 2:
        print("[WARN] Not enough structures for Volet 1 plots")
        return

    pdb_ids = centers_df["pdb_id"].tolist()
    coords = centers_df[["cx", "cy", "cz"]].values

    # Heatmap
    dist_mat = cdist(coords, coords, metric="euclidean")
    dist_df = pd.DataFrame(dist_mat, index=pdb_ids, columns=pdb_ids)

    fig, ax = plt.subplots(figsize=(max(6, len(pdb_ids) * 0.55),
                                    max(5, len(pdb_ids) * 0.5)))
    sns.heatmap(
        dist_df, annot=True, fmt=".1f", cmap="YlOrRd",
        linewidths=0.5, ax=ax, cbar_kws={"label": "Distance (Å)"},
        annot_kws={"size": 7},
    )
    ax.set_title(
        f"{class_label} — Peptide center-of-mass distances after receptor alignment",
        fontsize=12, pad=12,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    fig.tight_layout()
    out = outdir / f"{class_label.replace(' ', '_')}.volet1_cm_distances.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out}")

    dist_df.to_csv(
        outdir / f"{class_label.replace(' ', '_')}.volet1_cm_distances.tsv",
        sep="\t", float_format="%.3f",
    )

    # PCA
    if len(coords) < 3:
        print("[WARN] Not enough structures for PCA (need ≥ 3)")
        return

    pca = PCA(n_components=2)
    proj = pca.fit_transform(coords)
    ev = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(7, 6))
    for i, (pid, row) in enumerate(zip(pdb_ids, proj)):
        color = PEPTIDE_PALETTE[i % len(PEPTIDE_PALETTE)]
        n_res = int(centers_df.loc[centers_df["pdb_id"] == pid, "n_res"].iloc[0])
        ax.scatter(row[0], row[1], s=80 + n_res * 3, color=color,
                   edgecolors="black", linewidths=0.6, zorder=3)
        ax.annotate(pid.upper(), (row[0], row[1]),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=7, color="black")

    ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}% variance)", fontsize=10)
    ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}% variance)", fontsize=10)
    ax.set_title(
        f"{class_label} — PCA of peptide centers of mass\n"
        f"(after receptor alignment — point size ∝ peptide length)",
        fontsize=11,
    )
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.axvline(0, color="gray", lw=0.5, ls="--")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = outdir / f"{class_label.replace(' ', '_')}.volet1_pca_poses.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out}")


# ============================================================
# VOLET 2 — Depth and spatial deviation
# ============================================================

def compute_depth_and_spatial_deviation(
    peptide_df: pd.DataFrame,
    pocket_center: np.ndarray,
) -> pd.DataFrame:
    """
    For each peptide residue Cα:
      - depth_to_pocket_center = distance to pocket center
      - spatial_deviation      = distance to global centroid of all peptide Cα
    """
    if peptide_df.empty:
        return peptide_df.copy()

    out = peptide_df.copy()
    coords = out[["x", "y", "z"]].values.astype(float)

    out["depth_to_pocket_center"] = np.linalg.norm(coords - pocket_center, axis=1)

    global_centroid = coords.mean(axis=0)
    out["spatial_deviation"] = np.linalg.norm(coords - global_centroid, axis=1)

    out["depth_zone"] = pd.cut(
        out["depth_to_pocket_center"],
        bins=DEPTH_BINS,
        labels=DEPTH_LABELS,
        right=True,
    )

    return out


def plot_volet2(peptide_df: pd.DataFrame, outdir: Path, class_label: str):
    if peptide_df.empty:
        print("[WARN] No peptide residue data for Volet 2")
        return

    pdb_ids = sorted(peptide_df["pdb_id"].unique())
    color_map = {pid: PEPTIDE_PALETTE[i % len(PEPTIDE_PALETTE)]
                 for i, pid in enumerate(pdb_ids)}

    # Scatter
    fig, ax = plt.subplots(figsize=(9, 6))
    for pid in pdb_ids:
        sub = peptide_df[peptide_df["pdb_id"] == pid]
        ax.scatter(
            sub["depth_to_pocket_center"], sub["spatial_deviation"],
            c=color_map[pid], s=35, alpha=0.75,
            edgecolors="none", label=pid.upper(),
        )

    ax.axvline(10, color="gray", lw=1, ls="--", alpha=0.6)
    ax.axvline(20, color="gray", lw=1, ls="--", alpha=0.6)

    ylim_top = max(float(peptide_df["spatial_deviation"].max()) * 1.05, 1.0)
    ax.text(5, ylim_top * 0.97, "Inner\npocket",
            ha="center", va="top", fontsize=8, color="#e31a1c", alpha=0.8)
    ax.text(15, ylim_top * 0.97, "Intermediate",
            ha="center", va="top", fontsize=8, color="#ff7f00", alpha=0.8)
    ax.text(25, ylim_top * 0.97, "Outer",
            ha="center", va="top", fontsize=8, color="#1f78b4", alpha=0.8)

    ax.set_xlabel("Depth from consensus pocket center (Å)", fontsize=11)
    ax.set_ylabel("Spatial deviation (Å)", fontsize=11)
    ax.set_title(
        f"{class_label} — Peptide residue spatial variability vs pocket depth\n"
        f"(Cα, after receptor alignment)",
        fontsize=12,
    )
    ax.legend(
        title="Structure", fontsize=7, title_fontsize=8,
        loc="upper left", bbox_to_anchor=(1.01, 1), frameon=True,
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = outdir / f"{class_label.replace(' ', '_')}.volet2_scatter_depth_spatial_deviation.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out}")

    # Boxplot
    zone_data = [
        peptide_df.loc[peptide_df["depth_zone"] == zone, "spatial_deviation"].dropna().values
        for zone in DEPTH_LABELS
    ]
    zone_counts = [len(d) for d in zone_data]

    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot(
        zone_data, patch_artist=True, notch=False,
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker="o", markersize=3, alpha=0.4),
    )
    for patch, color in zip(bp["boxes"], DEPTH_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(
        [f"{lbl}\n(n={n})" for lbl, n in zip(DEPTH_LABELS, zone_counts)],
        fontsize=9,
    )
    ax.set_ylabel("Spatial deviation (Å)", fontsize=11)
    ax.set_title(
        f"{class_label} — Spatial deviation of peptide residues by pocket depth zone\n"
        f"(Cα, after receptor alignment)",
        fontsize=11,
    )
    ax.grid(True, axis="y", alpha=0.3)

    for i, d in enumerate(zone_data):
        if len(d) > 0:
            med = np.median(d)
            ax.text(i + 1, med + 0.3, f"{med:.1f} Å",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")

    fig.tight_layout()
    out = outdir / f"{class_label.replace(' ', '_')}.volet2_boxplot_depth_zones.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out}")

    out_tsv = outdir / f"{class_label.replace(' ', '_')}.volet2_residue_depth_spatial_deviation.tsv"
    peptide_df.to_csv(out_tsv, sep="\t", index=False, float_format="%.4f")
    print(f"[SAVED] {out_tsv}")

    print("\n[SUMMARY] Spatial deviation by depth zone:")
    for zone, d in zip(DEPTH_LABELS, zone_data):
        if len(d) > 0:
            print(f"  {zone.replace(chr(10), ' '):30s}  "
                  f"n={len(d):4d}  median={np.median(d):.2f} Å  "
                  f"mean={np.mean(d):.2f} Å  std={np.std(d):.2f} Å")
        else:
            print(f"  {zone.replace(chr(10), ' '):30s}  n=0")


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Peptide spatial variability — {args.class_label}")
    print(f"{'='*60}\n")

    # Metadata
    print("[STEP 1] Loading metadata...")
    pocket_df = load_metadata(args.pocket_tsv, args.class_label)
    struct_info = (
        pocket_df[["pdb_id_norm", "target_chain", "peptide_chain"]]
        .drop_duplicates()
        .rename(columns={"pdb_id_norm": "pdb_id"})
        .sort_values(["pdb_id", "target_chain", "peptide_chain"])
    )

    # Load and align
    print("[STEP 2] Loading structures and aligning receptors...")
    structures = load_structures_and_align(
        struct_info=struct_info,
        pdb_dir=args.pdb_dir,
        reference_pdb=args.reference_pdb,
    )

    # Pocket center in aligned reference frame
    print("\n[STEP 3] Computing consensus pocket center from aligned reference...")
    pocket_center = extract_consensus_pocket_center_from_reference(
        pocket_df=pocket_df,
        consensus_tsv=args.consensus_tsv,
        structures=structures,
        reference_pdb=args.reference_pdb,
    )

    # Peptide coordinates
    print("\n[STEP 4] Extracting peptide Cα coordinates...")
    peptide_ca, peptide_df = extract_peptide_coordinates(
        struct_info=struct_info,
        structures=structures,
    )

    if not peptide_ca:
        raise SystemExit("[ERROR] No peptide data could be extracted. Check CIF paths and chain IDs.")

    # Volet 1
    print("\n[STEP 5] Volet 1 — Global peptide pose analysis...")
    plot_volet1(peptide_ca, outdir, args.class_label)

    # Volet 2
    print("\n[STEP 6] Volet 2 — Depth vs spatial deviation...")
    peptide_df = compute_depth_and_spatial_deviation(peptide_df, pocket_center)
    plot_volet2(peptide_df, outdir, args.class_label)

    print(f"\n[DONE] All outputs written to: {outdir}\n")


if __name__ == "__main__":
    main()
