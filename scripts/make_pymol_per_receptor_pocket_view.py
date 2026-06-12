#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_pymol_per_receptor_pocket_view.py

Génère un fichier PyMOL (.pml) pour un récepteur de référence montrant :
  - Le récepteur en gris cartoon
  - Tous les peptides de la classe superposés en transparent (sticks + cartoon)
  - Les positions consensus de la poche colorées par propriété biophysique majoritaire

Exclusions par défaut : 9iqv (muscarinic toxin, 65 aa), 9mni (66 aa)

Usage :
  python3 make_pymol_per_receptor_pocket_view.py \\
    --by_residue run_out/biophys_annotations/pocket_biophys_by_residue.with_gpcrdb_segments.with_target_resnum.tsv \\
    --consensus_tsv out/consensus_validable/consensus_Class_A_thr50.validable.tsv \\
    --cif_dir cif_cache/ \\
    --class_label "Class A" \\
    --reference_pdb 9bkk \\
    --outdir out/pymol_pocket_views/

  # Class B :
  python3 make_pymol_per_receptor_pocket_view.py \\
    --by_residue run_out/biophys_annotations/pocket_biophys_by_residue.with_gpcrdb_segments.with_target_resnum.tsv \\
    --consensus_tsv out/consensus_validable/consensus_Class_B_thr50.validable.tsv \\
    --cif_dir cif_cache/ \\
    --class_label "Class B" \\
    --reference_pdb 9bue \\
    --outdir out/pymol_pocket_views/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# ─── Couleurs biophysiques (cohérentes avec le reste du projet) ──────────────
BIOPHYS_PYMOL_COLORS: dict[str, str] = {
    "hydrophobic":     "black",
    "aromatic":        "orange",
    "positive":        "blue",
    "negative":        "red",
    "polar_uncharged": "violet",
    "structural":      "green",
    "other":           "gray70",
    "NA":              "gray70",
}

# Ordre de priorité pour le vote majoritaire (en cas d'égalité)
BIOPHYS_PRIORITY = [
    "aromatic", "positive", "negative", "polar_uncharged", "hydrophobic", "other"
]
BIOPHYS_FLAG_COLS = {
    "aromatic":        "is_aromatic",
    "positive":        "is_pos",
    "negative":        "is_neg",
    "polar_uncharged": "is_polar",
    "hydrophobic":     "is_hydrophobic",
    "other":           "is_other",
}

# PDBs exclus par défaut (peptides trop grands)
DEFAULT_EXCLUDE = {"9iqv", "9mni"}


# ─── Helpers ────────────────────────────────────────────────────────────────

def norm_pdb(x: str) -> str:
    return str(x).strip().lower()


def make_resi_sel(resis: list[int]) -> str:
    resis = sorted(set(r for r in resis if r is not None))
    if not resis:
        return "resi 0"
    return "resi " + "+".join(str(r) for r in resis)


def filter_class(df: pd.DataFrame, class_label: str) -> pd.DataFrame:
    # Use full "Class A" / "Class B" to avoid 'a' matching within "Class B1 (Secretin)" etc.
    if class_label == "Class A":
        mask = df["gpcr_class"].astype(str).str.contains("Class A", case=False, na=False)
    elif class_label == "Class B":
        mask = df["gpcr_class"].astype(str).str.contains("Class B", case=False, na=False)
    else:
        mask = df["gpcr_class"].astype(str).str.contains(class_label, case=False, na=False)
    return df[mask].copy()


def compute_consensus_biophys(by_res: pd.DataFrame, class_label: str) -> dict[str, str]:
    """
    Vote majoritaire de la classe biophysique par position GPCRdb pour la classe donnée.
    Retourne un dict {gpcrdb_pos: biophys_class}.
    """
    df = filter_class(by_res, class_label).copy()

    for col in BIOPHYS_FLAG_COLS.values():
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    grouped = df.groupby("gpcrdb")[list(BIOPHYS_FLAG_COLS.values())].sum()

    result: dict[str, str] = {}
    for gpcrdb_pos, counts in grouped.iterrows():
        best_cls = "other"
        best_n = -1
        for cls in BIOPHYS_PRIORITY:
            col = BIOPHYS_FLAG_COLS[cls]
            n = int(counts[col])
            if n > best_n:
                best_n = n
                best_cls = cls
        result[str(gpcrdb_pos).strip()] = best_cls

    return result


# ─── PyMOL helpers ───────────────────────────────────────────────────────────

def pml_header(bg: str = "white") -> list[str]:
    return [
        "reinitialize",
        f"bg_color {bg}",
        "set ray_opaque_background, 1",
        "viewport 2400, 1800",
        "set antialias, 2",
        "set cartoon_fancy_helices, 1",
        "set cartoon_sampling, 14",
        "set stick_radius, 0.18",
        "set depth_cue, 0",
        "set two_sided_lighting, on",
        "set ray_trace_mode, 1",
        "set label_size, 18",
        "set label_color, black",
        "set label_outline_color, white",
        "",
    ]


def png_cmd(path: Path, dpi: int = 300) -> str:
    return f"png {path.resolve().as_posix()}, dpi={dpi}, ray=1"


# ─── Main ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--by_residue",    required=True,
                    help="pocket_biophys_by_residue.with_gpcrdb_segments.with_target_resnum.tsv")
    ap.add_argument("--consensus_tsv", required=True,
                    help="consensus_Class_X_thr50.validable.tsv")
    ap.add_argument("--cif_dir",       required=True,
                    help="Dossier contenant les fichiers {pdb_id}.cif")
    ap.add_argument("--class_label",   required=True,
                    help='Ex: "Class A" or "Class B"')
    ap.add_argument("--reference_pdb", required=True,
                    help="PDB de référence pour l'alignement, ex: 9bkk")
    ap.add_argument("--outdir",        required=True)
    ap.add_argument("--exclude_pdbs",  nargs="*", default=None,
                    help="PDB IDs à exclure (défaut: 9iqv 9mni)")
    ap.add_argument("--peptide_transparency", type=float, default=0.65,
                    help="Transparence des peptides (0=opaque, 1=invisible, défaut=0.65)")
    ap.add_argument("--show_surface",  action="store_true",
                    help="Affiche la surface des résidus consensus")
    ap.add_argument("--surface_transparency", type=float, default=0.35)
    ap.add_argument("--label_consensus", action="store_true",
                    help="Annote les résidus consensus avec leur position GPCRdb")
    ap.add_argument("--png",           default=None,
                    help="Si fourni, exporte une image PNG via ray")
    ap.add_argument("--png_dpi",       type=int, default=300)
    ap.add_argument("--bg",            default="white", choices=["white", "black"])
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    exclude_pdbs: set[str] = {norm_pdb(x) for x in (
        args.exclude_pdbs if args.exclude_pdbs is not None else DEFAULT_EXCLUDE
    )}

    # ── Chargement des données ──────────────────────────────────────────────
    by_res = pd.read_csv(args.by_residue, sep="\t", dtype=str)
    consensus = pd.read_csv(args.consensus_tsv, sep="\t", dtype=str)

    by_res["pdb_id"] = by_res["pdb_id"].map(norm_pdb)

    # Filtre classe + exclusions
    by_res_cls = filter_class(by_res, args.class_label)
    by_res_cls = by_res_cls[~by_res_cls["pdb_id"].isin(exclude_pdbs)].copy()

    if by_res_cls.empty:
        raise SystemExit(f"[ERROR] Aucune donnée pour classe {args.class_label} après filtrage")

    # Positions consensus
    consensus_positions: set[str] = set(
        consensus["gpcrdb_pos"].astype(str).str.strip().tolist()
    )

    # Biophysique majoritaire par position GPCRdb
    biophys_map = compute_consensus_biophys(by_res, args.class_label)

    # ── Récepteur de référence ──────────────────────────────────────────────
    ref_pdb = norm_pdb(args.reference_pdb)
    ref_rows = by_res_cls[by_res_cls["pdb_id"] == ref_pdb].copy()

    if ref_rows.empty:
        raise SystemExit(f"[ERROR] {ref_pdb} absent de la classe {args.class_label}")

    ref_chain     = str(ref_rows["target_chain"].iloc[0]).strip()
    ref_pep_chain = str(ref_rows["peptide_chain"].iloc[0]).strip()

    # Positions consensus sur la référence (gpcrdb → target_resnum)
    ref_rows["gpcrdb_norm"]      = ref_rows["gpcrdb"].astype(str).str.strip()
    ref_rows["target_resnum_int"] = pd.to_numeric(
        ref_rows["target_resnum"], errors="coerce"
    )
    ref_rows = ref_rows.dropna(subset=["target_resnum_int"])
    ref_rows["target_resnum_int"] = ref_rows["target_resnum_int"].astype(int)

    ref_cons = ref_rows[ref_rows["gpcrdb_norm"].isin(consensus_positions)].copy()
    ref_cons = (
        ref_cons
        .drop_duplicates(subset=["target_resnum_int", "gpcrdb_norm"])
        .sort_values("target_resnum_int")
    )

    if ref_cons.empty:
        raise SystemExit(
            f"[ERROR] Aucune position consensus mappée sur {ref_pdb}.\n"
            f"  Positions consensus : {sorted(consensus_positions)[:10]}...\n"
            f"  Positions GPCRdb sur {ref_pdb} : "
            f"{sorted(ref_rows['gpcrdb_norm'].unique())[:10]}..."
        )

    print(f"[INFO] {len(ref_cons)} positions consensus mappées sur {ref_pdb}")

    # ── Structures à charger ────────────────────────────────────────────────
    struct_info = (
        by_res_cls[["pdb_id", "target_chain", "peptide_chain"]]
        .drop_duplicates()
        .sort_values("pdb_id")
    )

    # ── Construction du PML ─────────────────────────────────────────────────
    pml: list[str] = []
    pml += pml_header(args.bg)

    receptor_objects: list[str] = []
    peptide_objects:  list[str] = []

    for _, row in struct_info.iterrows():
        pdb_id = str(row["pdb_id"])
        tchain = str(row["target_chain"]).strip()
        pchain = str(row["peptide_chain"]).strip()

        cif_path = Path(args.cif_dir).resolve() / f"{pdb_id}.cif"
        if not cif_path.exists():
            print(f"[WARN] CIF manquant : {cif_path}, ignoré")
            continue

        rec = f"rec_{pdb_id}"
        pep = f"pep_{pdb_id}"
        obj = f"obj_{pdb_id}"

        receptor_objects.append(rec)
        peptide_objects.append(pep)

        pml.append(f'load "{cif_path.as_posix()}", {obj}')
        pml.append(f"select {rec}, {obj} and chain {tchain} and polymer.protein")
        pml.append(f"select {pep}, {obj} and chain {pchain} and polymer.protein")

    if not receptor_objects:
        raise SystemExit("[ERROR] Aucun fichier CIF trouvé")

    pml += ["hide everything", ""]

    # ── Alignement ──────────────────────────────────────────────────────────
    ref_rec = f"rec_{ref_pdb}"
    pml.append(f"# Alignement de tous les récepteurs sur {ref_pdb}")
    for rec in receptor_objects:
        if rec != ref_rec:
            pml.append(f"align {rec}, {ref_rec}")
    pml.append("")

    # ── Sélections consensus ────────────────────────────────────────────────
    cons_resis = ref_cons["target_resnum_int"].tolist()
    pml.append(f"select ref_consensus, {ref_rec} and ({make_resi_sel(cons_resis)})")
    pml.append("")

    # ── Récepteur de référence en gris ──────────────────────────────────────
    pml.append(f"# Récepteur {ref_pdb} en gris")
    pml.append(f"show cartoon, {ref_rec}")
    pml.append(f"color gray60, {ref_rec}")
    pml.append("")

    # ── Peptides transparents ────────────────────────────────────────────────
    pml.append("# Peptides superposés en transparent")
    all_pep_sel = " or ".join(peptide_objects)
    pml.append(f"select all_peptides, {all_pep_sel}")
    pml.append("show sticks, all_peptides")
    pml.append("show cartoon, all_peptides")
    pml.append("color wheat, all_peptides")
    pml.append(f"set stick_transparency, {args.peptide_transparency:.2f}, all_peptides")
    pml.append(f"set cartoon_transparency, {args.peptide_transparency:.2f}, all_peptides")
    pml.append(f"set stick_radius, 0.22, all_peptides")
    pml.append("")

    # ── Résidus consensus colorés (par-dessus le gris du récepteur) ─────────
    pml.append("# Résidus consensus colorés par propriété biophysique")
    pml.append("show sticks, ref_consensus")
    pml.append("show spheres, ref_consensus")
    pml.append("set sphere_scale, 0.30, ref_consensus")
    pml.append("set stick_radius, 0.28, ref_consensus")

    for _, r in ref_cons.iterrows():
        resi     = int(r["target_resnum_int"])
        gp       = str(r["gpcrdb_norm"])
        biophys  = biophys_map.get(gp, "other")
        color    = BIOPHYS_PYMOL_COLORS.get(biophys, "gray70")
        sel_name = f"cons_{resi}"
        pml.append(f"select {sel_name}, {ref_rec} and resi {resi}")
        pml.append(f"color {color}, {sel_name}")
        if args.label_consensus:
            label = gp.replace('"', "'")
            pml.append(f'label ({sel_name} and name CA), "{label}"')

    pml.append("")

    # ── Surface optionnelle ──────────────────────────────────────────────────
    if args.show_surface:
        pml.append("# Surface des résidus consensus")
        pml.append("show surface, ref_consensus")
        pml.append(f"set transparency, {args.surface_transparency:.2f}, ref_consensus")
        pml.append("")

    # ── Vue centrée sur la poche ─────────────────────────────────────────────
    pml.append("# Orientation centrée sur la poche")
    pml.append("orient all_peptides")
    pml.append("zoom ref_consensus, 10")
    pml.append("turn y, 10")
    pml.append("turn x, -5")
    pml.append("rebuild")
    pml.append("refresh")
    pml.append("")

    # ── Export PNG optionnel ─────────────────────────────────────────────────
    if args.png:
        png_path = Path(args.png).resolve()
        pml.append(png_cmd(png_path, args.png_dpi))
        pml.append("")

    # ── Écriture ────────────────────────────────────────────────────────────
    class_tag = args.class_label.replace(" ", "_")
    out_pml   = outdir / f"pocket_view_{class_tag}_{ref_pdb}.pml"
    out_pml.write_text("\n".join(pml) + "\n")

    print(f"[DONE] {out_pml}")
    print()
    print("Lancer avec :")
    print(f"  LIBGL_ALWAYS_SOFTWARE=1 pymol {out_pml}")
    print()
    print("Légende des couleurs :")
    for cls, color in BIOPHYS_PYMOL_COLORS.items():
        if cls not in ("NA",):
            print(f"  {color:12s} → {cls}")


if __name__ == "__main__":
    main()
