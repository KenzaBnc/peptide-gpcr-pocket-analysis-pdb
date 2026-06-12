#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import gemmi
import subprocess
import tempfile


def parse_args():
    ap = argparse.ArgumentParser(
        description="Calcule les features de structure secondaire des peptides via mkdssp."
    )
    ap.add_argument("--input_tsv", required=True,
                    help="TSV peptide_contacts.peptide_sequences.tsv")
    ap.add_argument("--pdb_dir", required=True,
                    help="Dossier cache mmCIF (ex: cif_cache)")
    ap.add_argument("--out_tsv", required=True,
                    help="Fichier TSV de sortie (peptide_structure_features.tsv)")
    return ap.parse_args()


# ======================
# DSSP extraction
# ======================

def compute_dssp_map(cif_path: Path):
    dssp_map = {}

    with tempfile.NamedTemporaryFile(suffix=".dssp", mode="w+", delete=True) as tmp:
        cmd = ["mkdssp", "--output-format", "dssp", str(cif_path), tmp.name]
        subprocess.run(cmd, check=True)

        with open(tmp.name, "r", encoding="utf-8", errors="ignore") as f:
            in_data = False
            for line in f:
                if line.startswith("  #  RESIDUE"):
                    in_data = True
                    continue
                if not in_data:
                    continue
                if len(line) < 17:
                    continue

                resseq_raw = line[5:10].strip()
                chain_id = line[11].strip()
                ss = line[16].strip()

                if not resseq_raw or not chain_id:
                    continue

                try:
                    resseq = int(resseq_raw)
                except ValueError:
                    continue

                if ss == "":
                    ss = " "

                dssp_map[(chain_id, resseq)] = ss

    return dssp_map

# ======================
# Extract residues
# ======================

def extract_peptide_residues(structure, chain_id):
    residues = []

    for model in structure:
        for chain in model:
            if str(chain.name).strip() != str(chain_id).strip():
                continue
            for res in chain:
                if str(res.name).strip() == "HOH":
                    continue
                residues.append(res)

    return residues


# ======================
# Features
# ======================

def compute_end_to_end(residues):
    if len(residues) < 2:
        return np.nan

    try:
        a = residues[0].find_atom("CA", "\0")
        b = residues[-1].find_atom("CA", "\0")

        if a is None or b is None:
            return np.nan

        return a.pos.dist(b.pos)
    except Exception:
        return np.nan


def compute_ss_features(residues, dssp_map, chain_id):
    helix = 0
    coil = 0
    total = 0

    for res in residues:
        try:
            resseq = int(res.seqid.num)
        except Exception:
            continue

        ss = dssp_map.get((str(chain_id).strip(), resseq), " ")

        total += 1

        if ss in ["H", "G", "I"]:
            helix += 1
        else:
            # coil inclut P, T, S, blank, etc.
            coil += 1

    if total == 0:
        return np.nan, np.nan

    return helix / total, coil / total


# ======================
# MAIN
# ======================

def main():
    args = parse_args()
    input_tsv = Path(args.input_tsv)
    struct_dir = Path(args.pdb_dir)
    out_tsv = Path(args.out_tsv)

    df = pd.read_csv(input_tsv, sep="\t", dtype=str)

    df["pdb_id"] = df["pdb_id"].str.upper().str.strip()
    df["peptide_chain"] = df["peptide_chain"].str.strip()

    results = []

    for _, row in df.iterrows():
        pdb_id = row["pdb_id"]
        chain = row["peptide_chain"]

        cif_path = struct_dir / f"{pdb_id}.cif"
        if not cif_path.exists():
            cif_path = struct_dir / f"{pdb_id.lower()}.cif"

        if not cif_path.exists():
            print(f"[WARN] missing CIF: {pdb_id}")
            continue

        try:
            structure = gemmi.read_structure(str(cif_path))
        except Exception:
            print(f"[ERROR] read CIF: {pdb_id}")
            continue

        residues = extract_peptide_residues(structure, chain)

        if len(residues) == 0:
            print(f"[WARN] peptide vide: {pdb_id} chain {chain}")
            continue

        try:
            dssp_map = compute_dssp_map(cif_path)
        except Exception:
            print(f"[WARN] DSSP fail: {pdb_id}")
            dssp_map = {}

        helix, coil = compute_ss_features(residues, dssp_map, chain)
        dist = compute_end_to_end(residues)

        results.append({
            "pdb_id": pdb_id,
            "peptide_chain": chain,
            "helix_fraction": helix,
            "coil_fraction": coil,
            "end_to_end_distance": dist
        })

    out = pd.DataFrame(results)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_tsv, sep="\t", index=False)

    print(f"[DONE] {out_tsv}")
    print(f"[INFO] {len(out)} peptides résumés")


if __name__ == "__main__":
    main()
