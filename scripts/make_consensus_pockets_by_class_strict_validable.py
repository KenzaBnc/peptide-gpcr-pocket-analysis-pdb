#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Recompute consensus pockets per GPCR class with "strict-validable" denominator.

Consensus definition:
- strict : position counted if source == 'both'
- lenient: position counted if source in {'both','gpcrdb_only'}

Denominator (per class) is mode-consistent:
- strict : only structures that have >=1 'both' row (within class subset)
- lenient: only structures that have >=1 ('both' or 'gpcrdb_only') row (within class subset)

Inputs:
- cmp_tsv : out/gpcrdb_vs_gemmi.tsv
- biophys_by_pocket : run_out/biophys_annotations/pocket_biophys_by_pocket.tsv (for class labels)

Outputs (per class):
- outdir/consensus_<Class>_thrXX.validable.tsv
- outdir/consensus_<Class>_thrXX.validable.meta.tsv

Les catégories d’interaction GPCRdb sont conservées telles quelles :
  hydrophobic / aromatic / polar / vdw / ionic

usage :
python3 scripts/make_consensus_pockets_by_class_strict_validable.py \
  --cmp_tsv out/gpcrdb_vs_gemmi.tsv \
  --biophys_by_pocket run_out/biophys_annotations/pocket_biophys_by_pocket.tsv \
  --outdir out/consensus_validable \
  --threshold 0.50 \
  --classes "Class A,Class B" \
  --mode strict
"""

import argparse
import re
from pathlib import Path
import pandas as pd


def norm_pdb(x: str) -> str:
    return str(x).strip().upper()


def simplify_class_label(x: str) -> str:
    """
    Stable labels:
      "Class A (Rhodopsin)" -> "Class A"
      "Class B1 (Secretin)" -> "Class B"
    """
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.startswith("Class A"):
        return "Class A"
    if s.startswith("Class B"):
        return "Class B"
    return s


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmp_tsv", required=True)
    ap.add_argument("--biophys_by_pocket", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--threshold", type=float, default=0.50)
    ap.add_argument(
        "--classes",
        default="Class A,Class B",
        help='Comma-separated list, e.g. "Class A,Class B"'
    )
    ap.add_argument("--mode", choices=["strict", "lenient"], default="strict")
    ap.add_argument(
        "--exclude_pdbs",
        default="",
        help="Comma-separated PDB IDs to exclude from all classes (e.g. '9MNI')"
    )
    return ap.parse_args()


def load_cmp(cmp_tsv: str) -> pd.DataFrame:
    df = pd.read_csv(cmp_tsv, sep="\t", dtype=str)
    df["pdb_id"] = df["pdb_id"].apply(norm_pdb)
    df["source"] = df["source"].fillna("").astype(str)
    df["gpcrdb_pos"] = df["gpcrdb_pos"].fillna("").astype(str)

    if "segment_gemmi" in df.columns:
        df["segment_gemmi"] = df["segment_gemmi"].fillna("").astype(str)
    if "segment_gpcrdb" in df.columns:
        df["segment_gpcrdb"] = df["segment_gpcrdb"].fillna("").astype(str)
    if "gpcrdb_interaction_types" in df.columns:
        df["gpcrdb_interaction_types"] = df["gpcrdb_interaction_types"].fillna("").astype(str)

    df = df[df["gpcrdb_pos"] != ""].copy()
    return df


def load_class_map(biophys_by_pocket_tsv: str) -> pd.DataFrame:
    df = pd.read_csv(biophys_by_pocket_tsv, sep="\t", dtype=str)
    df["pdb_id"] = df["pdb_id"].apply(norm_pdb)
    df["gpcr_class_simplified"] = df["gpcr_class"].apply(simplify_class_label)
    return df[["pdb_id", "gpcr_class", "gpcr_class_simplified"]].drop_duplicates()


def agg_types(series: pd.Series) -> str:
    """Agrège les types d’interaction GPCRdb sans simplification."""
    seen = set()
    for v in series.dropna().astype(str):
        v = v.strip()
        if not v:
            continue
        for t in re.split(r"[;,]\s*", v):
            t = t.strip().lower()
            if t:
                seen.add(t)
    return "; ".join(sorted(seen)) if seen else ""


def compute_consensus_for_class(
    cmp_df: pd.DataFrame,
    class_pdbs_all: list,
    class_label: str,
    threshold: float,
    mode: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      consensus_df: one row per consensus GPCRdb position
      meta_df: summary stats
    """
    n_total = len(class_pdbs_all)
    if n_total == 0:
        consensus_df = pd.DataFrame()
        meta_df = pd.DataFrame([{
            "class_label": class_label,
            "mode": mode,
            "threshold": threshold,
            "n_structures_total_in_class": 0,
            "n_structures_validable_in_mode": 0,
            "consensus_positions": 0
        }])
        return consensus_df, meta_df

    # Subset of cmp rows for this class' pdbs
    sub_all = cmp_df[cmp_df["pdb_id"].isin(class_pdbs_all)].copy()

    # Define "validable in this mode" pdbs (denominator consistent with mode)
    if mode == "strict":
        denom_sources = {"both"}
        keep_sources = {"both"}
    else:
        denom_sources = {"both", "gpcrdb_only"}
        keep_sources = {"both", "gpcrdb_only"}

    pdbs_valid_mode = sorted(
        sub_all.loc[sub_all["source"].isin(denom_sources), "pdb_id"].unique().tolist()
    )
    n_valid = len(pdbs_valid_mode)

    if n_valid == 0:
        consensus_df = pd.DataFrame(columns=[
            "gpcrdb_pos", "n_structures_with_pos", "n_structures_validable_in_mode",
            "freq_structures", "threshold", "segment_gemmi", "segment_gpcrdb",
            "gpcrdb_interaction_types"
        ])
        meta_df = pd.DataFrame([{
            "class_label": class_label,
            "mode": mode,
            "threshold": threshold,
            "n_structures_total_in_class": n_total,
            "n_structures_validable_in_mode": 0,
            "consensus_positions": 0,
            "validable_pdbs": ""
        }])
        return consensus_df, meta_df

    # Now only work within validable pdbs (mode-specific)
    sub = sub_all[sub_all["pdb_id"].isin(pdbs_valid_mode)].copy()

    # Count frequency of positions using keep_sources
    sub_pos = sub[sub["source"].isin(keep_sources)][["pdb_id", "gpcrdb_pos"]].drop_duplicates()
    counts = sub_pos.groupby("gpcrdb_pos").size().reset_index(name="n_structures_with_pos")
    counts["n_structures_validable_in_mode"] = n_valid
    counts["freq_structures"] = counts["n_structures_with_pos"] / float(n_valid)
    counts["threshold"] = threshold

    consensus = counts[counts["freq_structures"] >= threshold].copy()

    # -------------------------
    # SEGMENT MAP (mode of segment per gpcrdb_pos)
    # -------------------------
    seg_cols = [c for c in ["segment_gemmi", "segment_gpcrdb"] if c in sub.columns]
    if seg_cols and not consensus.empty:
        seg_map = (
            sub[["gpcrdb_pos"] + seg_cols]
            .dropna(subset=["gpcrdb_pos"])
            .replace("", pd.NA)
            .dropna(how="all", subset=seg_cols)
            .groupby("gpcrdb_pos")[seg_cols]
            .agg(lambda x: x.mode().iloc[0] if len(x.dropna()) > 0 else pd.NA)
            .reset_index()
        )
        consensus = consensus.merge(seg_map, on="gpcrdb_pos", how="left")

    # -------------------------
    # INTERACTION TYPES (GPCRdb-derived rows only)
    # -------------------------
    if "gpcrdb_interaction_types" in sub.columns and not consensus.empty:
        mask = (
            sub["source"].isin(["both", "gpcrdb_only"]) &
            sub["gpcrdb_interaction_types"].astype(str).str.strip().ne("")
        )
        it = sub.loc[mask, ["gpcrdb_pos", "gpcrdb_interaction_types"]].dropna()

        if not it.empty:
            it_map = (
                it.groupby("gpcrdb_pos")["gpcrdb_interaction_types"]
                .apply(agg_types)
                .reset_index()
            )

            if "gpcrdb_interaction_types" in consensus.columns:
                consensus = consensus.drop(columns=["gpcrdb_interaction_types"])

            consensus = consensus.merge(it_map, on="gpcrdb_pos", how="left")

    # Stable sorting
    if not consensus.empty:
        consensus = consensus.sort_values(
            ["freq_structures", "gpcrdb_pos"],
            ascending=[False, True]
        ).reset_index(drop=True)

    meta_df = pd.DataFrame([{
        "class_label": class_label,
        "mode": mode,
        "threshold": threshold,
        "n_structures_total_in_class": n_total,
        "n_structures_validable_in_mode": n_valid,
        "consensus_positions": int(consensus.shape[0]),
        "validable_pdbs": ",".join(pdbs_valid_mode),
    }])

    return consensus, meta_df


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cmp_df = load_cmp(args.cmp_tsv)
    class_map = load_class_map(args.biophys_by_pocket)

    excluded = {norm_pdb(p) for p in args.exclude_pdbs.split(",") if p.strip()}
    if excluded:
        print(f"[INFO] Excluding PDB(s): {', '.join(sorted(excluded))}")
        class_map = class_map[~class_map["pdb_id"].isin(excluded)].copy()
        cmp_df    = cmp_df[~cmp_df["pdb_id"].isin(excluded)].copy()

    # attach class info to cmp_df
    cmp_df = cmp_df.merge(class_map, on="pdb_id", how="left")

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]

    for cl in classes:
        pdbs_in_class = sorted(
            class_map.loc[class_map["gpcr_class_simplified"] == cl, "pdb_id"].unique().tolist()
        )

        consensus_df, meta_df = compute_consensus_for_class(
            cmp_df=cmp_df,
            class_pdbs_all=pdbs_in_class,
            class_label=cl,
            threshold=args.threshold,
            mode=args.mode
        )

        thr = int(round(args.threshold * 100))
        prefix = outdir / f"consensus_{cl.replace(' ', '_')}_thr{thr}.validable"

        consensus_df.to_csv(str(prefix) + ".tsv", sep="\t", index=False)
        meta_df.to_csv(str(prefix) + ".meta.tsv", sep="\t", index=False)

        print(
            f"[DONE] {cl}: total={len(pdbs_in_class)} | "
            f"validable(mode)={int(meta_df.loc[0, 'n_structures_validable_in_mode'])} | "
            f"consensus_positions={int(meta_df.loc[0, 'consensus_positions'])} | "
            f"out={prefix}.tsv"
        )


if __name__ == "__main__":
    main()
