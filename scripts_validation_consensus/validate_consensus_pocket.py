#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
validate_consensus_pocket.py

Validations robustes de la poche consensus (par classe) à partir de :
  1) Leave-one-out (jackknife) : retirer une structure à la fois et recalculer le consensus
     -> mesure d'overlap avec le consensus "référence" fourni (fichiers consensus_*_thrXX.validable.tsv)

  2) Structural mapping (segments GPCRdb) :
     -> pour chaque position consensus, segment(s) observé(s) (TM/ECL/ICL)
     -> résumé des segments (comptage)

Entrées:
  --contacts_tsv   : pocket_biophys_by_residue.with_gpcrdb_segments.with_target_resnum.tsv
     Colonnes attendues (tolérance via renaming):
       - pdb_id
       - gpcr_class (ex: "Class A (Rhodopsin)")
       - gpcrdb_pos OU gpcrdb_display_generic_number OU gpcrdb
       - gpcrdb_segment (ou gpcrdb_segments selon tes tables)

  --consensus_dir  : dossier contenant consensus_Class_A_thr50.validable.tsv etc.
     Colonnes attendues:
       - gpcrdb_pos OU gpcrdb_display_generic_number OU gpcrdb

Usage exemple:
  python3 scripts_validation_consensus/validate_consensus_pocket.py \
    --contacts_tsv run_out/biophys_annotations/pocket_biophys_by_residue.with_gpcrdb_segments.with_target_resnum.tsv \
    --consensus_dir out/consensus_validable_strict \
    --classes "Class A,Class B" \
    --threshold 0.50 \
    --outdir out/consensus_validation

Sorties:
  outdir/
    Class_A/
      mapping_segments.tsv
      segments_summary.tsv
      leave_one_out.tsv
      leave_one_out.summary.txt
    Class_B/
      ...

Notes:
- Le recalcul du consensus leave-one-out utilise la même définition simple:
    position consensus si elle apparaît dans >= threshold des structures de la classe.
- On compte "apparaît" = position présente au moins 1 fois dans une structure.
"""

import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd


# -----------------------
# Helpers
# -----------------------

def norm_pdb(x: str) -> str:
    return str(x).strip().upper()

def simplify_gpcrdb_pos(pos: str):
    """Normalise les positions GPCRdb pour matcher entre tables."""
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
        a = int(m.group(1)); b = int(m.group(2)); c = int(m.group(3))
        # loop-like
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

def simplify_class(s):
    s = str(s)
    if "Class A" in s:
        return "Class A"
    if "Class B" in s:
        return "Class B"
    return "NA"


def pick_gpcrdb_col(df: pd.DataFrame) -> str:
    for cand in ["gpcrdb_pos", "gpcrdb_display_generic_number", "gpcrdb"]:
        if cand in df.columns:
            return cand
    raise ValueError("Aucune colonne GPCRdb trouvée: gpcrdb_pos / gpcrdb_display_generic_number / gpcrdb")

def pick_segment_col(df: pd.DataFrame) -> str:
    for cand in ["gpcrdb_segment", "segment", "gpcrdb_segments"]:
        if cand in df.columns:
            return cand
    raise ValueError("Aucune colonne segment trouvée: gpcrdb_segment / segment / gpcrdb_segments")


# -----------------------
# Loaders
# -----------------------

def load_contacts(contacts_tsv: str) -> pd.DataFrame:
    df = pd.read_csv(contacts_tsv, sep="\t", dtype=str)

    if "pdb_id" not in df.columns:
        raise ValueError(f"[contacts_tsv] colonne pdb_id manquante. Colonnes: {df.columns.tolist()}")
    if "gpcr_class" not in df.columns:
        raise ValueError(f"[contacts_tsv] colonne gpcr_class manquante. Colonnes: {df.columns.tolist()}")

    gpcrdb_col = pick_gpcrdb_col(df)
    seg_col = pick_segment_col(df)

    df["pdb_id"] = df["pdb_id"].map(norm_pdb)
    df["class_simple"] = df["gpcr_class"].map(simplify_class)

    df["gpcrdb_pos"] = df[gpcrdb_col].map(simplify_gpcrdb_pos)
    df[seg_col] = df[seg_col].astype(str).str.strip()

    df = df.dropna(subset=["gpcrdb_pos"]).copy()
    df = df.rename(columns={seg_col: "gpcrdb_segment"})

    return df[["pdb_id", "class_simple", "gpcrdb_pos", "gpcrdb_segment"]].copy()


def load_consensus_positions(consensus_dir: str, class_label: str, thr: float) -> list[str]:
    safe = class_label.replace(" ", "_")
    thr_int = int(round(thr * 100))

    p_valid = Path(consensus_dir) / f"consensus_{safe}_thr{thr_int}.validable.tsv"
    p_pos   = Path(consensus_dir) / f"consensus_{safe}_thr{thr_int}.positions.tsv"

    if p_valid.exists():
        path = p_valid
    elif p_pos.exists():
        path = p_pos
    else:
        raise FileNotFoundError(f"Consensus introuvable: {p_valid.name} ou {p_pos.name} dans {consensus_dir}")

    cdf = pd.read_csv(path, sep="\t", dtype=str)
    gpcrdb_col = pick_gpcrdb_col(cdf)

    cdf["gpcrdb_pos"] = cdf[gpcrdb_col].map(simplify_gpcrdb_pos)
    cdf = cdf.dropna(subset=["gpcrdb_pos"]).drop_duplicates(subset=["gpcrdb_pos"]).copy()

    return sorted(cdf["gpcrdb_pos"].tolist(), key=gpcrdb_sort_key)


# -----------------------
# Core computations
# -----------------------

def compute_consensus_from_contacts(sub: pd.DataFrame, threshold: float) -> list[str]:
    """
    sub contient: pdb_id, gpcrdb_pos, ...
    Une position est "présente dans une structure" si elle apparaît au moins 1 fois dans (pdb_id).
    Consensus = positions présentes dans >= threshold fraction des structures.
    """
    n_struct = sub["pdb_id"].nunique()
    if n_struct == 0:
        return []

    # présence binaire par structure
    pres = (
        sub.drop_duplicates(subset=["pdb_id", "gpcrdb_pos"])
           .groupby("gpcrdb_pos")["pdb_id"]
           .nunique()
    )
    freq = pres / float(n_struct)
    cons = freq[freq >= threshold].index.tolist()
    return sorted(cons, key=gpcrdb_sort_key)


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def leave_one_out_validation(sub: pd.DataFrame, ref_consensus: list[str], threshold: float) -> pd.DataFrame:
    """
    Pour chaque pdb_id, on retire la structure, on recalcule consensus,
    puis on mesure:
      - overlap_ref = |cons_LOO ∩ ref| / |ref|
      - jaccard_ref = J(cons_LOO, ref)
      - n_cons_LOO
    """
    ref_set = set(ref_consensus)
    structs = sorted(sub["pdb_id"].unique().tolist())
    rows = []

    for pdb in structs:
        loo = sub[sub["pdb_id"] != pdb].copy()
        cons_loo = compute_consensus_from_contacts(loo, threshold=threshold)
        cons_set = set(cons_loo)

        overlap_ref = (len(cons_set & ref_set) / len(ref_set)) if len(ref_set) > 0 else np.nan
        jac = jaccard(cons_set, ref_set)

        rows.append({
            "removed_pdb_id": pdb,
            "n_structures_remaining": loo["pdb_id"].nunique(),
            "n_consensus_LOO": len(cons_loo),
            "overlap_with_ref": overlap_ref,
            "jaccard_with_ref": jac,
        })

    return pd.DataFrame(rows)


def structural_mapping(consensus_positions: list[str], sub: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pour chaque position consensus, récupérer le(s) segment(s) observé(s) dans les structures.
    - mapping_segments.tsv: gpcrdb_pos -> segments uniques + counts
    - segments_summary.tsv: distribution globale des segments sur les positions consensus
    """
    if not consensus_positions:
        return pd.DataFrame(), pd.DataFrame()

    csub = sub[sub["gpcrdb_pos"].isin(consensus_positions)].copy()

    mapping = (
        csub.groupby(["gpcrdb_pos", "gpcrdb_segment"])["pdb_id"]
            .nunique()
            .reset_index(name="n_structures")
    )

    # segments list per position
    agg = (
        mapping.sort_values(["gpcrdb_pos", "n_structures"], ascending=[True, False])
               .groupby("gpcrdb_pos")
               .agg(
                   segments=("gpcrdb_segment", lambda x: ",".join(x.astype(str).tolist())),
                   n_structures_by_segment=("n_structures", lambda x: ",".join(x.astype(str).tolist())),
               )
               .reset_index()
    )

    agg["gpcrdb_pos"] = pd.Categorical(agg["gpcrdb_pos"], categories=consensus_positions, ordered=True)
    agg = agg.sort_values("gpcrdb_pos").reset_index(drop=True)

    seg_summary = (
        mapping.groupby("gpcrdb_segment")["gpcrdb_pos"]
               .nunique()
               .reset_index(name="n_positions")
               .sort_values("n_positions", ascending=False)
               .reset_index(drop=True)
    )

    return agg, seg_summary


# -----------------------
# CLI
# -----------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contacts_tsv", required=True)
    ap.add_argument("--consensus_dir", required=True)
    ap.add_argument("--classes", default="Class A,Class B", help='ex: "Class A,Class B"')
    ap.add_argument("--threshold", type=float, default=0.50)
    ap.add_argument("--outdir", required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_contacts(args.contacts_tsv)

    wanted = [c.strip() for c in args.classes.split(",") if c.strip()]
    for cl in wanted:
        cl_dir = outdir / cl.replace(" ", "_")
        cl_dir.mkdir(parents=True, exist_ok=True)

        sub = df[df["class_simple"] == cl].copy()
        if sub.empty:
            print(f"[WARN] aucune donnée pour {cl}")
            continue

        # --- reference consensus from file
        ref_cons = load_consensus_positions(args.consensus_dir, cl, args.threshold)
        ref_set = set(ref_cons)

        # --- leave-one-out
        loo_df = leave_one_out_validation(sub, ref_consensus=ref_cons, threshold=args.threshold)
        loo_path = cl_dir / "leave_one_out.tsv"
        loo_df.to_csv(loo_path, sep="\t", index=False)

        # summary text
        mean_overlap = loo_df["overlap_with_ref"].mean()
        min_overlap = loo_df["overlap_with_ref"].min()
        mean_jac = loo_df["jaccard_with_ref"].mean()
        min_jac = loo_df["jaccard_with_ref"].min()

        summary_txt = (
            f"Class: {cl}\n"
            f"threshold: {args.threshold:.2f}\n"
            f"n_structures: {sub['pdb_id'].nunique()}\n"
            f"ref_consensus_npos: {len(ref_cons)}\n"
            f"mean overlap_with_ref: {mean_overlap:.3f}\n"
            f"min  overlap_with_ref: {min_overlap:.3f}\n"
            f"mean jaccard_with_ref: {mean_jac:.3f}\n"
            f"min  jaccard_with_ref: {min_jac:.3f}\n"
        )
        (cl_dir / "leave_one_out.summary.txt").write_text(summary_txt)

        # --- structural mapping
        mapping_df, segsum_df = structural_mapping(ref_cons, sub)
        mapping_df.to_csv(cl_dir / "mapping_segments.tsv", sep="\t", index=False)
        segsum_df.to_csv(cl_dir / "segments_summary.tsv", sep="\t", index=False)

        print(f"[DONE] {cl}")
        print(f"  - ref consensus positions: {len(ref_cons)}")
        print(f"  - leave-one-out: {loo_path}")
        print(f"  - mapping: {cl_dir / 'mapping_segments.tsv'}")
        print(f"  - segments summary: {cl_dir / 'segments_summary.tsv'}")

    print("[ALL DONE] outdir:", outdir)


if __name__ == "__main__":
    main()
