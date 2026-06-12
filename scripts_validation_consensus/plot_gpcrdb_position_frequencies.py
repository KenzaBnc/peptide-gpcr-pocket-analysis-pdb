#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_gpcrdb_position_frequencies.py

But:
- Afficher la fréquence des positions GPCRdb à travers les structures
- Annoter chaque position avec son segment GPCRdb
- Mettre en évidence:
    - le seuil de consensus (0.5)
    - les positions ultra-robustes (1.0)

Usage:
python3 scripts_validation_consensus/plot_gpcrdb_position_frequencies.py \
  --contacts_tsv run_out/biophys_annotations/pocket_biophys_by_residue.with_gpcrdb_segments.with_target_resnum.tsv \
  --consensus_tsv out/consensus_validable_strict/consensus_Class_A_thr50.validable.tsv \
  --class_label "Class A" \
  --outdir out/consensus_validation/Class_A/figures
"""

import argparse
from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt


def norm_pdb(x: str) -> str:
    return str(x).strip().upper()


def simplify_gpcrdb_pos(pos: str):
    if pos is None or pd.isna(pos):
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

    return None


def gpcrdb_sort_key(pos: str):
    try:
        a, b = pos.split("x")
        return (int(a), int(b))
    except Exception:
        return (999999, 999999)


def simplify_class_label(x: str):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.startswith("Class A"):
        return "Class A"
    if s.startswith("Class B"):
        return "Class B"
    return s


def pick_gpcrdb_col(df: pd.DataFrame):
    for cand in ["gpcrdb_pos", "gpcrdb_display_generic_number", "gpcrdb"]:
        if cand in df.columns:
            return cand
    raise ValueError("Aucune colonne GPCRdb trouvée.")


def load_contacts(contacts_tsv: str, class_label: str) -> pd.DataFrame:
    df = pd.read_csv(contacts_tsv, sep="\t", dtype=str)

    if "pdb_id" not in df.columns:
        raise ValueError("Colonne pdb_id manquante")
    if "gpcr_class" not in df.columns:
        raise ValueError("Colonne gpcr_class manquante")

    gpcrdb_col = pick_gpcrdb_col(df)

    if "gpcrdb_segment" not in df.columns:
        raise ValueError("Colonne gpcrdb_segment manquante")

    df["pdb_id"] = df["pdb_id"].map(norm_pdb)
    df["class_simple"] = df["gpcr_class"].map(simplify_class_label)
    df["gpcrdb_pos"] = df[gpcrdb_col].map(simplify_gpcrdb_pos)
    df["gpcrdb_segment"] = df["gpcrdb_segment"].astype(str).str.strip()

    df = df[(df["class_simple"] == class_label) & df["gpcrdb_pos"].notna()].copy()
    return df


def load_consensus(consensus_tsv: str) -> list:
    df = pd.read_csv(consensus_tsv, sep="\t", dtype=str)
    gpcrdb_col = pick_gpcrdb_col(df)
    df["gpcrdb_pos"] = df[gpcrdb_col].map(simplify_gpcrdb_pos)
    df = df.dropna(subset=["gpcrdb_pos"]).drop_duplicates(subset=["gpcrdb_pos"])
    return sorted(df["gpcrdb_pos"].tolist(), key=gpcrdb_sort_key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contacts_tsv", required=True)
    ap.add_argument("--consensus_tsv", required=True)
    ap.add_argument("--class_label", required=True, help='Ex: "Class A" or "Class B"')
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    contacts = load_contacts(args.contacts_tsv, args.class_label)
    consensus = load_consensus(args.consensus_tsv)

    n_structures = contacts["pdb_id"].nunique()

    # fréquence par position = nb structures contenant la position / nb structures total
    pres = (
        contacts[contacts["gpcrdb_pos"].isin(consensus)]
        .drop_duplicates(subset=["pdb_id", "gpcrdb_pos"])
        .groupby("gpcrdb_pos")["pdb_id"]
        .nunique()
        .reset_index(name="n_structures")
    )
    pres["frequency"] = pres["n_structures"] / n_structures

    # segment majoritaire par position
    segmap = (
        contacts[contacts["gpcrdb_pos"].isin(consensus)]
        .groupby("gpcrdb_pos")["gpcrdb_segment"]
        .agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0])
        .reset_index()
    )

    freq_df = pres.merge(segmap, on="gpcrdb_pos", how="left")

    # s’assurer qu’on garde l’ordre consensus
    freq_df["gpcrdb_pos"] = pd.Categorical(freq_df["gpcrdb_pos"], categories=consensus, ordered=True)
    freq_df = freq_df.sort_values("gpcrdb_pos").reset_index(drop=True)

    # labels position + segment
    freq_df["xlab"] = freq_df.apply(
        lambda r: f"{r['gpcrdb_pos']}\n{r['gpcrdb_segment']}", axis=1
    )

    # export TSV
    out_tsv = outdir / f"gpcrdb_position_frequencies_{args.class_label.replace(' ','_')}.tsv"
    freq_df.to_csv(out_tsv, sep="\t", index=False)

    # plot
    plt.figure(figsize=(max(10, len(freq_df) * 0.8), 6))
    bars = plt.bar(freq_df["xlab"], freq_df["frequency"], color="#4C78A8", alpha=0.9)

    plt.axhline(1.0, linestyle="--", color="#1f77b4", linewidth=2, label="Ultra-robust core (100%)")
    plt.axhline(0.5, linestyle=":", color="#2ca02c", linewidth=2, label="Consensus threshold (50%)")

    plt.ylabel("Frequency across structures", fontsize=12)
    plt.xlabel("GPCRdb position (segment)", fontsize=12)
    plt.title(f"{args.class_label} — Frequency of consensus GPCRdb positions", fontsize=14)

    plt.xticks(rotation=60, ha="right")
    plt.ylim(0, 1.08)
    plt.legend(frameon=False)
    plt.tight_layout()

    out_png = outdir / f"gpcrdb_position_frequencies_{args.class_label.replace(' ','_')}.png"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

    print("[DONE]")
    print(" - Figure:", out_png)
    print(" - Table :", out_tsv)


if __name__ == "__main__":
    main()
