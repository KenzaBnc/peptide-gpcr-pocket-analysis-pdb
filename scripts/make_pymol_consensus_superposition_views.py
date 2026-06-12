#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_pymol_consensus_superposition_views.py

Génère un script PyMOL pour visualiser, pour une classe donnée :

1. reference + all peptides
2. reference + all peptides + consensus residues
3. consensus residues only on reference

Corrections anti-image-noire intégrées :
- ray=1 dans png
- viewport fixe
- ray_opaque_background=1
- refresh avant chaque export
- chemins PNG absolus
- hide/show explicite avant chaque figure

Fonctionnalités :
- charge toutes les structures .cif d'une classe
- aligne tous les récepteurs sur une structure de référence
- affiche le récepteur de référence de manière lisible
- affiche tous les peptides ensemble
- colore les résidus consensus sur la référence selon la classe biophysique majoritaire
- annote les résidus consensus avec leur position GPCRdb
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


BIOPHYS_PYMOL_COLORS = {
    "hydrophobic": "black",
    "aromatic": "orange",
    "positive": "blue",
    "negative": "red",
    "polar_uncharged": "violet",
    "structural": "green",
    "other": "gray70",
    "NA": "gray70",
}

PEPTIDE_PALETTE = [
    "tv_orange", "marine", "forest", "magenta", "purple", "cyan",
    "salmon", "teal", "olive", "hotpink", "deepblue", "lime",
    "yelloworange", "raspberry", "slate", "bluewhite"
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--by_residue", required=True)
    ap.add_argument("--consensus_tsv", required=True)
    ap.add_argument("--pdb_dir", required=True, help="Directory containing {pdb_id}.cif files")
    ap.add_argument("--class_label", required=True, help='Ex: "Class A" or "Class B"')
    ap.add_argument("--reference_pdb", required=True, help="Reference PDB id for alignment, ex: 9bkk")
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--fig1_png", default=None, help="reference + all peptides")
    ap.add_argument("--fig2_png", default=None, help="reference + all peptides + consensus residues")
    ap.add_argument("--fig3_png", default=None, help="consensus residues only on reference")

    ap.add_argument("--png_dpi", type=int, default=300)
    ap.add_argument("--bg", default="white", choices=["white", "black"])
    ap.add_argument("--show_surface", action="store_true")
    ap.add_argument("--surface_transparency", type=float, default=0.35)

    ap.add_argument(
        "--label_consensus",
        action="store_true",
        help="Annoter les résidus consensus sur la référence"
    )
    ap.add_argument(
        "--label_mode",
        default="gpcrdb",
        choices=["gpcrdb", "top_aa", "both"],
        help="Texte des labels consensus"
    )

    return ap.parse_args()


def safe_str(x):
    if x is None:
        return "NA"
    s = str(x).strip()
    return s if s and s.upper() != "NA" else "NA"


def safe_int(x):
    try:
        if x is None:
            return None
        s = str(x).strip()
        if not s or s.upper() == "NA":
            return None
        return int(float(s))
    except Exception:
        return None


def norm_pdb(x: str) -> str:
    return safe_str(x).lower()


def normalize_major_biophys(x: str) -> str:
    s = safe_str(x)
    if s == "other":
        return "polar_uncharged"
    return s


def make_resi_selection(resis: list[int]) -> str:
    resis = sorted(set(r for r in resis if isinstance(r, int)))
    if not resis:
        return "resi 0"
    return "resi " + "+".join(str(r) for r in resis)


def filter_class(df: pd.DataFrame, class_label: str) -> pd.DataFrame:
    if "gpcr_class" not in df.columns:
        raise SystemExit("[ERROR] column gpcr_class not found in by_residue TSV")

    if class_label == "Class A":
        mask = df["gpcr_class"].astype(str).str.contains("Class A", case=False, na=False)
    elif class_label == "Class B":
        mask = df["gpcr_class"].astype(str).str.contains("Class B", case=False, na=False)
    else:
        mask = df["gpcr_class"].astype(str).str.contains(class_label, case=False, na=False)

    return df[mask].copy()


def build_consensus_label(row, mode="gpcrdb"):
    gp = safe_str(row.get("gpcrdb_pos"))
    aa = safe_str(row.get("top_aa"))

    if mode == "gpcrdb":
        return gp
    if mode == "top_aa":
        return aa
    if mode == "both":
        if gp != "NA" and aa != "NA":
            return f"{gp} {aa}"
        return gp if gp != "NA" else aa
    return gp


def choose_gpcrdb_column(df: pd.DataFrame) -> str:
    for cand in ["gpcrdb_pos", "gpcrdb", "gpcrdb_display_generic_number"]:
        if cand in df.columns:
            return cand
    raise SystemExit("[ERROR] No gpcrdb_pos/gpcrdb/gpcrdb_display_generic_number column in by_residue")


def png_cmd(path_str: str, dpi: int) -> str:
    return f"png {Path(path_str).resolve().as_posix()}, dpi={dpi}, ray=1"


def add_common_render_settings(pml: list[str], bg: str):
    pml.append("reinitialize")
    pml.append(f"bg_color {bg}")
    pml.append("set ray_opaque_background, 1")
    pml.append("viewport 2400,1800")
    pml.append("set antialias, 2")
    pml.append("set cartoon_fancy_helices, 1")
    pml.append("set cartoon_sampling, 14")
    pml.append("set stick_radius, 0.18")
    pml.append("set sphere_scale, 0.25")
    pml.append("set depth_cue, 0")
    pml.append("set two_sided_lighting, on")
    pml.append("set ray_trace_mode, 1")
    pml.append("set label_size, 16")
    pml.append("set label_color, black")
    pml.append("set label_outline_color, white")
    pml.append("set dash_gap, 0.35")
    pml.append("set dash_radius, 0.08")
    pml.append("")


def figure_preamble(pml: list[str]):
    pml.append("hide everything")
    pml.append("disable all")
    pml.append("enable all")
    pml.append("rebuild")
    pml.append("refresh")


def main():
    args = parse_args()

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    by_residue = pd.read_csv(args.by_residue, sep="\t", dtype=str)
    consensus = pd.read_csv(args.consensus_tsv, sep="\t", dtype=str)

    required_by_res = {"pdb_id", "target_chain", "peptide_chain", "target_resnum"}
    missing = sorted(required_by_res - set(by_residue.columns))
    if missing:
        raise SystemExit(f"[ERROR] Missing columns in by_residue: {missing}")

    if "gpcrdb_pos" not in consensus.columns:
        raise SystemExit("[ERROR] Missing gpcrdb_pos in consensus_tsv")

    by_residue["pdb_id"] = by_residue["pdb_id"].map(norm_pdb)
    by_residue["target_chain"] = by_residue["target_chain"].astype(str).str.strip()
    by_residue["peptide_chain"] = by_residue["peptide_chain"].astype(str).str.strip()
    by_residue["target_resnum_int"] = by_residue["target_resnum"].map(safe_int)

    by_residue = by_residue.dropna(subset=["target_resnum_int"]).copy()
    by_residue = filter_class(by_residue, args.class_label)

    if by_residue.empty:
        raise SystemExit(f"[ERROR] No rows found for class {args.class_label}")

    reference_pdb = norm_pdb(args.reference_pdb)
    if reference_pdb not in set(by_residue["pdb_id"]):
        raise SystemExit(f"[ERROR] reference_pdb {reference_pdb} not found in filtered by_residue table")

    struct_info = (
        by_residue[["pdb_id", "target_chain", "peptide_chain"]]
        .drop_duplicates()
        .sort_values(["pdb_id", "target_chain", "peptide_chain"])
    )

    ref_rows = by_residue[by_residue["pdb_id"] == reference_pdb].copy()
    gp_col = choose_gpcrdb_column(ref_rows)

    ref_rows["gpcrdb_pos_norm"] = ref_rows[gp_col].astype(str).str.strip()
    consensus["gpcrdb_pos_norm"] = consensus["gpcrdb_pos"].astype(str).str.strip()
    if "major_biophys_class" not in consensus.columns:
        consensus["major_biophys_class"] = "NA"
    consensus["major_biophys_class"] = consensus["major_biophys_class"].map(normalize_major_biophys)
    if "top_aa" not in consensus.columns:
        consensus["top_aa"] = "X"

    ref_cons = ref_rows.merge(
        consensus[["gpcrdb_pos_norm", "gpcrdb_pos", "top_aa", "major_biophys_class", "freq_structures"]],
        on="gpcrdb_pos_norm",
        how="inner"
    ).copy()

    if ref_cons.empty:
        raise SystemExit("[ERROR] No consensus positions matched onto the reference structure")

    ref_cons = (
        ref_cons.sort_values(["target_resnum_int", "gpcrdb_pos"])
        .drop_duplicates(subset=["target_resnum_int", "gpcrdb_pos"])
    )

    ref_chain = (
        struct_info.loc[struct_info["pdb_id"] == reference_pdb, "target_chain"]
        .iloc[0]
    )
    ref_pep_chain = (
        struct_info.loc[struct_info["pdb_id"] == reference_pdb, "peptide_chain"]
        .iloc[0]
    )

    consensus_resis = ref_cons["target_resnum_int"].tolist()
    consensus_resi_sel = make_resi_selection(consensus_resis)

    all_consensus_pos = set(consensus["gpcrdb_pos_norm"].dropna())
    ref_consensus_pos = set(ref_cons["gpcrdb_pos_norm"].dropna())
    missing_consensus_pos = sorted(all_consensus_pos - ref_consensus_pos)

    class_tag = args.class_label.replace(" ", "_")

    fig1_png = str((Path(args.fig1_png).resolve()) if args.fig1_png else (outdir / f"{class_tag}.reference_plus_peptides.png").resolve())
    fig2_png = str((Path(args.fig2_png).resolve()) if args.fig2_png else (outdir / f"{class_tag}.reference_peptides_consensus.png").resolve())
    fig3_png = str((Path(args.fig3_png).resolve()) if args.fig3_png else (outdir / f"{class_tag}.reference_consensus_only.png").resolve())
    out_pml = (outdir / f"consensus_superposition_{class_tag}.pml").resolve()

    pml = []
    add_common_render_settings(pml, args.bg)

    if missing_consensus_pos:
        pml.append(f'print "Missing consensus positions on reference {reference_pdb}: {", ".join(missing_consensus_pos)}"')
        pml.append("")

    loaded_objects = []
    peptide_objects = []
    receptor_objects = []

    for _, row in struct_info.iterrows():
        pdb_id = row["pdb_id"]
        tchain = row["target_chain"]
        pchain = row["peptide_chain"]

        structure_path = Path(args.pdb_dir).resolve() / f"{pdb_id}.cif"
        if not structure_path.exists():
            print(f"[WARN] Missing CIF file: {structure_path}, skipping")
            continue

        obj = f"obj_{pdb_id}_{tchain}_{pchain}"
        rec = f"rec_{pdb_id}_{tchain}_{pchain}"
        pep = f"pep_{pdb_id}_{tchain}_{pchain}"

        loaded_objects.append(obj)
        receptor_objects.append(rec)
        peptide_objects.append(pep)

        pml.append(f'load "{structure_path.as_posix()}", {obj}')
        pml.append(f"select {rec}, {obj} and chain {tchain} and polymer.protein")
        pml.append(f"select {pep}, {obj} and chain {pchain} and polymer.protein")

    if not loaded_objects:
        raise SystemExit("[ERROR] No CIF files found locally")

    pml.append("hide everything")
    pml.append("")

    ref_rec = f"rec_{reference_pdb}_{ref_chain}_{ref_pep_chain}"
    if ref_rec not in receptor_objects:
        raise SystemExit("[ERROR] Reference receptor object not loaded")

    pml.append(f"# Alignment onto reference {reference_pdb}")
    for rec in receptor_objects:
        if rec == ref_rec:
            continue
        pml.append(f"align {rec}, {ref_rec}")
    pml.append("")

    pml.append("# Define consensus selection on reference")
    pml.append(f"select ref_consensus, {ref_rec} and ({consensus_resi_sel})")
    for _, r in ref_cons.iterrows():
        resi = int(r["target_resnum_int"])
        major = normalize_major_biophys(r.get("major_biophys_class"))
        color = BIOPHYS_PYMOL_COLORS.get(major, "gray70")
        pml.append(f"select cons_{resi}, {ref_rec} and resi {resi}")
        pml.append(f"color {color}, cons_{resi}")
        if args.label_consensus:
            label = build_consensus_label(r, mode=args.label_mode).replace('"', "'")
            pml.append(f'label (cons_{resi} and name CA), "{label}"')
    pml.append("")

    pml.append("select all_peptides, " + " or ".join(peptide_objects))
    pml.append("")

    # FIGURE 1
    pml.append("# Figure 1: reference + all peptides")
    figure_preamble(pml)

    pml.append(f"show cartoon, {ref_rec}")
    pml.append(f"color gray70, {ref_rec}")
    for i, pep in enumerate(peptide_objects):
        color = PEPTIDE_PALETTE[i % len(PEPTIDE_PALETTE)]
        pml.append(f"show sticks, {pep}")
        pml.append(f"color {color}, {pep}")
        pml.append(f"set stick_radius, 0.28, {pep}")

    pml.append(f"orient {ref_rec}")
    pml.append(f"zoom {ref_rec} or all_peptides, 14")
    pml.append("turn y, 10")
    pml.append("turn x, -5")
    pml.append("refresh")
    pml.append(png_cmd(fig1_png, args.png_dpi))
    pml.append("")

    # FIGURE 2
    pml.append("# Figure 2: reference + all peptides + consensus residues")
    figure_preamble(pml)

    pml.append(f"show cartoon, {ref_rec}")
    pml.append(f"color gray70, {ref_rec}")
    for i, pep in enumerate(peptide_objects):
        color = PEPTIDE_PALETTE[i % len(PEPTIDE_PALETTE)]
        pml.append(f"show sticks, {pep}")
        pml.append(f"color {color}, {pep}")
        pml.append(f"set stick_radius, 0.28, {pep}")

    pml.append("show sticks, ref_consensus")
    pml.append("show spheres, ref_consensus")
    pml.append("set sphere_scale, 0.32, ref_consensus")
    pml.append("set stick_radius, 0.30, ref_consensus")
    for _, r in ref_cons.iterrows():
        resi = int(r["target_resnum_int"])
        major = normalize_major_biophys(r.get("major_biophys_class"))
        color = BIOPHYS_PYMOL_COLORS.get(major, "gray70")
        pml.append(f"color {color}, cons_{resi}")

    if args.show_surface:
        pml.append("show surface, ref_consensus")
        pml.append(f"set transparency, {args.surface_transparency}, ref_consensus")

    pml.append(f"orient {ref_rec}")
    pml.append(f"zoom {ref_rec} or all_peptides or ref_consensus, 10")
    pml.append("turn y, 10")
    pml.append("turn x, -5")
    pml.append("refresh")
    pml.append(png_cmd(fig2_png, args.png_dpi))
    pml.append("")

    # FIGURE 3
    pml.append("# Figure 3: consensus residues only on reference")
    figure_preamble(pml)

    pml.append(f"show cartoon, {ref_rec}")
    pml.append(f"color gray70, {ref_rec}")
    pml.append("show sticks, ref_consensus")
    pml.append("show spheres, ref_consensus")
    pml.append("set sphere_scale, 0.32, ref_consensus")
    pml.append("set stick_radius, 0.30, ref_consensus")
    for _, r in ref_cons.iterrows():
        resi = int(r["target_resnum_int"])
        major = normalize_major_biophys(r.get("major_biophys_class"))
        color = BIOPHYS_PYMOL_COLORS.get(major, "gray70")
        pml.append(f"color {color}, cons_{resi}")

    pml.append(f"orient {ref_rec}")
    pml.append(f"zoom {ref_rec} or ref_consensus, 11")
    pml.append("turn y, 10")
    pml.append("turn x, -5")
    pml.append("refresh")
    pml.append(png_cmd(fig3_png, args.png_dpi))
    pml.append("")

    out_pml.write_text("\n".join(pml))

    print("[DONE] Wrote PyMOL script:", out_pml)
    print("[DONE] Expected renders:")
    print("  -", fig1_png)
    print("  -", fig2_png)
    print("  -", fig3_png)
    if missing_consensus_pos:
        print("[WARN] Missing consensus positions on reference:", ", ".join(missing_consensus_pos))
    print()
    print("Run:")
    print(f"  LIBGL_ALWAYS_SOFTWARE=1 pymol -c {out_pml.as_posix()}")


if __name__ == "__main__":
    main()
