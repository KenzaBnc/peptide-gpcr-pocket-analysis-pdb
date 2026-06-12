#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
radar_by_class_polygons_with_structural_labels.py

Radar plot par classe GPCR:
- Axes = positions GPCRdb (communes)
- 1 polygone par structure (pdb_id + target_chain)
- Labels axes = "GPCRdb (≈chain:seqnum_mode)" pour ajouter une info structurale
- Export TSV:
  - axis_labels_<Class>.tsv : mapping axe -> seqnum_mode
  - values_by_structure_<Class>.tsv : matrice KD (ou autre) par structure

Input principal:
  --target_annot_tsv run_out/biophys_annotations/pocket_biophys_by_residue.with_gpcrdb_segments.tsv

Usage:
python3 scripts/radar_by_class_polygons_with_structural_labels.py \
  --target_annot_tsv run_out/biophys_annotations/pocket_biophys_by_residue.with_gpcrdb_segments.tsv \
  --outdir out/peptide_biophys/radar_class \
  --classes "Class A,Class B" \
  --value_col kd \
  --axes_mode consensus \
  --consensus_dir out/consensus_validable_strict \
  --threshold 0.50 \
  --max_structures 30
"""

import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
        a = int(m.group(1)); b = int(m.group(2)); c = int(m.group(3))
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

    df = pd.read_csv(path, sep="\t", dtype=str)
    if "gpcrdb_pos" not in df.columns:
        raise ValueError(f"[{path.name}] colonne gpcrdb_pos manquante. Colonnes: {df.columns.tolist()}")

    df["gpcrdb_pos"] = df["gpcrdb_pos"].map(simplify_gpcrdb_pos)
    df = df.dropna(subset=["gpcrdb_pos"]).drop_duplicates(subset=["gpcrdb_pos"])
    return sorted(df["gpcrdb_pos"].tolist(), key=gpcrdb_sort_key)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_annot_tsv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--classes", default="Class A,Class B")
    ap.add_argument("--value_col", default="kd", help="ex: kd")
    ap.add_argument("--max_structures", type=int, default=30)

    # Axes
    ap.add_argument("--axes_mode", choices=["all", "consensus"], default="consensus")
    ap.add_argument("--consensus_dir", default=None)
    ap.add_argument("--threshold", type=float, default=0.50)
    return ap.parse_args()


def pick_gpcrdb_col(df: pd.DataFrame) -> str:
    for cand in ["gpcrdb_pos", "gpcrdb_display_generic_number", "gpcrdb"]:
        if cand in df.columns:
            return cand
    raise ValueError("Aucune colonne GPCRdb trouvée: gpcrdb_pos / gpcrdb_display_generic_number / gpcrdb")


def pick_struct_pos_col(df: pd.DataFrame) -> str:
    # priorité: sequence_number (souvent fourni), sinon pocket_resi, sinon target_resnum si un jour présent
    for cand in ["sequence_number", "pocket_resi", "target_resnum"]:
        if cand in df.columns:
            return cand
    return None


def mode_or_na(series: pd.Series):
    s = series.dropna()
    if s.empty:
        return "NA"
    try:
        m = s.mode()
        if len(m) > 0:
            return str(m.iloc[0])
    except Exception:
        pass
    return str(s.iloc[0])


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.target_annot_tsv, sep="\t", dtype=str)
    df["pdb_id"] = df["pdb_id"].map(norm_pdb)

    if "gpcr_class" not in df.columns:
        raise ValueError("Colonne gpcr_class manquante dans target_annot_tsv.")

    gpcrdb_col = pick_gpcrdb_col(df)
    df["gpcrdb_pos"] = df[gpcrdb_col].map(simplify_gpcrdb_pos)

    struct_col = pick_struct_pos_col(df)
    if struct_col is None:
        df["struct_pos"] = pd.NA
    else:
        df["struct_pos"] = pd.to_numeric(df[struct_col], errors="coerce")

    if "target_chain" not in df.columns:
        df["target_chain"] = "?"
    df["target_chain"] = df["target_chain"].astype(str).str.strip()

    if args.value_col not in df.columns:
        raise ValueError(f"value_col='{args.value_col}' absent. Colonnes dispo: {df.columns.tolist()}")
    df["value"] = pd.to_numeric(df[args.value_col], errors="coerce")

    df["class_simple"] = df["gpcr_class"].map(simplify_class)

    wanted = [c.strip() for c in args.classes.split(",") if c.strip()]

    for cl in wanted:
        sub = df[(df["class_simple"] == cl) & df["gpcrdb_pos"].notna()].copy()
        if sub.empty:
            print(f"[WARN] pas de données pour {cl}")
            continue

        # ----- choisir axes
        if args.axes_mode == "consensus":
            if not args.consensus_dir:
                raise ValueError("--consensus_dir requis si --axes_mode consensus")
            axes = load_consensus_positions(args.consensus_dir, cl, args.threshold)
        else:
            axes = sorted(sub["gpcrdb_pos"].dropna().unique().tolist(), key=gpcrdb_sort_key)

        if not axes:
            print(f"[WARN] axes vides pour {cl}")
            continue

        # ----- construire un label structural “représentatif” par axe: mode(struct_pos) + mode(chain)
        axis_map = (
            sub[sub["gpcrdb_pos"].isin(axes)]
            .groupby("gpcrdb_pos", as_index=False)
            .agg({
                "struct_pos": mode_or_na,
                "target_chain": mode_or_na,
            })
        )
        axis_map = axis_map.set_index("gpcrdb_pos").reindex(axes)

        axis_labels = []
        for p in axes:
            ch = axis_map.loc[p, "target_chain"]
            sp = axis_map.loc[p, "struct_pos"]
            if sp == "NA" or pd.isna(sp):
                axis_labels.append(f"{p} (NA)")
            else:
                axis_labels.append(f"{p} (≈{ch}:{sp})")

        # export axis labels
        cl_dir = outdir / cl.replace(" ", "_")
        cl_dir.mkdir(parents=True, exist_ok=True)
        axis_out = pd.DataFrame({"gpcrdb_pos": axes, "axis_label": axis_labels,
                                 "chain_mode": axis_map["target_chain"].tolist(),
                                 "struct_pos_mode": axis_map["struct_pos"].tolist()})
        axis_out.to_csv(cl_dir / f"axis_labels_{cl.replace(' ','_')}.tsv", sep="\t", index=False)

        # ----- matrice values par structure (pdb_id + chain)
        sub["structure_id"] = sub["pdb_id"].astype(str) + "_" + sub["target_chain"].astype(str)

        # 1 valeur par (structure_id, gpcrdb_pos) -> mode/mean
        v = (
            sub[sub["gpcrdb_pos"].isin(axes)]
            .dropna(subset=["value"])
            .groupby(["structure_id", "gpcrdb_pos"], as_index=False)["value"]
            .mean()
        )

        mat = v.pivot(index="structure_id", columns="gpcrdb_pos", values="value").reindex(columns=axes)

        # limiter nb structures pour lisibilité
        mat = mat.sort_index()
        if mat.shape[0] > args.max_structures:
            mat = mat.head(args.max_structures)

        mat.to_csv(cl_dir / f"values_by_structure_{cl.replace(' ','_')}.tsv", sep="\t")

        # ----- plot radar
        N = len(axes)
        angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
        angles += angles[:1]  # close

        fig = plt.figure(figsize=(11, 11))
        ax = plt.subplot(111, polar=True)

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(axis_labels, fontsize=10)

        # une courbe par structure
        for sid, row in mat.iterrows():
            vals = row.values.astype(float)
            # gérer NaN: on les met à 0 (ou tu peux choisir de skip)
            vals = np.nan_to_num(vals, nan=0.0)
            vals = vals.tolist() + [vals.tolist()[0]]
            ax.plot(angles, vals, linewidth=2, alpha=0.9, label=sid)
            ax.fill(angles, vals, alpha=0.06)

        ax.set_title(f"{cl} — {args.value_col} across axes (1 polygon per structure)", pad=25, fontsize=16)

        # légende à droite (peut devenir longue)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.02), fontsize=9, framealpha=0.95)

        plt.tight_layout()
        out_png = cl_dir / f"radar_{args.value_col}_{cl.replace(' ','_')}.png"
        plt.savefig(out_png, dpi=200, bbox_inches="tight")
        plt.close(fig)

        print(f"[DONE] {cl}: axes={len(axes)} structures_plotted={mat.shape[0]} out={out_png}")


if __name__ == "__main__":
    main()
