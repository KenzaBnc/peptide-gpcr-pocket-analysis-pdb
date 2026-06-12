#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
annotate_pocket_biophys_gpcrdb_with_class.py

Fusion de 2 scripts en 1 :
1) Annotation biophysique des résidus de poche + hydrophobicité (Kyte–Doolittle) + GPCRdb numbering
2) Ajout direct de la classe GPCRdb (gpcr_class) depuis gpcr_70_evidence_table.tsv

- On charge une seule fois le mapping PDB -> GPCRdb class
- On écrit directement les sorties finales (avec gpcr_class) dans un répertoire dédié

INPUTS
- --input_tsv : peptide_ligands_gpcr.pockets.gpcrdb.tsv
- --evidence  : gpcr_70_evidence_table.tsv (colonnes: PDB__id, GPCRdb__class)

OUTPUTS (dans --outdir_biophys)
- pocket_biophys_by_residue.tsv  (avec colonne gpcr_class)
- pocket_biophys_by_pocket.tsv   (avec colonne gpcr_class)

USAGE
python3 scripts/annotate_pocket_biophys_gpcrdb_with_class.py \
  --input_tsv run_out/peptide_ligands_gpcr.pockets.gpcrdb.tsv \
  --evidence gpcr_70_evidence_table.tsv \
  --outdir_biophys run_out/biophys_annotations
"""

import csv
import argparse
import math
import os
import sys
from collections import Counter

# ===============================
# 1) Groupes biophysiques
# ===============================
# Schéma personnalisé 6 classes (Lehninger 7e éd. comme référence générale).
# C → hydrophobe (KD = +2.5, valeur positive comme AVILM).
# G/P → other (contraintes conformationnelles particulières : absence de chaîne
#   latérale pour G, cycle pyrrolidine pour P ; KD < 0 pour les deux).
AA_AROMATIC = set("FYW")
AA_POSITIVE = set("KRH")
AA_NEGATIVE = set("DE")
AA_POLAR_UNCHARGED = set("STNQ")
AA_HYDROPHOBIC_ALIPHATIC = set("AVILMC")
AA_OTHER = set("GP")

# ===============================
# 2) Kyte–Doolittle
# ===============================
KD = {
    "A": 1.8,  "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8,  "K": -3.9, "M": 1.9,  "F": 2.8,  "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

# ===============================
# 3) 3-letter → 1-letter
# ===============================
AA3_TO_1 = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C",
    "GLN":"Q","GLU":"E","GLY":"G","HIS":"H","ILE":"I",
    "LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P",
    "SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V"
}

# ===============================
# Helpers
# ===============================

def norm_pdb(x: str):
    """Normalize PDB id: strip, handle NA, lowercase."""
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.upper() == "NA":
        return None
    return s.lower()

def mean_ignore_nan(vals):
    vals = [v for v in vals if not math.isnan(v)]
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)

def classify_aa(aa1: str) -> dict:
    return {
        "aa": aa1,
        "is_aromatic": int(aa1 in AA_AROMATIC),
        "is_pos": int(aa1 in AA_POSITIVE),
        "is_neg": int(aa1 in AA_NEGATIVE),
        "is_polar": int(aa1 in AA_POLAR_UNCHARGED),
        "is_hydrophobic": int(aa1 in AA_HYDROPHOBIC_ALIPHATIC),
        "is_other": int(aa1 in AA_OTHER),
        "kd": KD.get(aa1, float("nan")),
    }

def parse_pocket_residues(pocket_residues: str):
    """
    Input:  F:82(GLN),F:84(THR),...
    Output: list of dicts preserving order:
      [{"chain":"F","resi":82,"aa3":"GLN","aa":"Q","raw":"F:82(GLN)"} , ...]
    """
    if not pocket_residues or pocket_residues == "NA":
        return []

    out = []
    for item in pocket_residues.split(","):
        item = item.strip()
        try:
            left, aa3_part = item.split("(")
            aa3 = aa3_part.replace(")", "").strip()
            chain, pos = left.split(":")
            chain = chain.strip()
            pos = int(pos.strip())
        except Exception:
            continue

        aa1 = AA3_TO_1.get(aa3, None)
        if aa1 is None:
            continue

        out.append({
            "chain": chain,
            "resi": pos,
            "aa3": aa3,
            "aa": aa1,
            "raw": item
        })
    return out

def parse_gpcrdb_numbers(pocket_gpcrdb_numbers: str):
    """Input: 2x60,2x62,... OR NA. Output: list preserving order."""
    if not pocket_gpcrdb_numbers or pocket_gpcrdb_numbers == "NA":
        return []
    return [x.strip() for x in pocket_gpcrdb_numbers.split(",") if x.strip()]

def parse_unmapped(unmapped_pocket_residues: str):
    """
    Input: same format as pocket_residues OR NA
    Output: set of raw tokens (e.g. {"F:82(GLN)", ...}) for quick membership.
    """
    if not unmapped_pocket_residues or unmapped_pocket_residues == "NA":
        return set()
    return set([x.strip() for x in unmapped_pocket_residues.split(",") if x.strip()])

def load_gpcr_class_mapping(evidence_tsv: str) -> dict:
    """
    Evidence file must contain columns: PDB__id, GPCRdb__class
    Returns mapping {pdb_id_norm: gpcr_class_str}
    Keeps the FIRST non-empty class per PDB.
    """
    mapping = {}
    with open(evidence_tsv, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"PDB__id", "GPCRdb__class"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"[ERROR] Evidence file missing columns: {sorted(missing)}. "
                f"Found: {reader.fieldnames}"
            )

        for row in reader:
            pdb = norm_pdb(row.get("PDB__id"))
            gpcr_class = (row.get("GPCRdb__class") or "").strip()
            if pdb is None:
                continue
            if gpcr_class == "" or gpcr_class.upper() in ("NA", "NAN", "NONE"):
                continue
            if pdb not in mapping:
                mapping[pdb] = gpcr_class
    return mapping

# ===============================
# Main
# ===============================

def main(input_tsv: str, evidence_tsv: str, outdir_biophys: str,
         out_residue_name: str, out_summary_name: str):

    os.makedirs(outdir_biophys, exist_ok=True)
    out_residue_tsv = os.path.join(outdir_biophys, out_residue_name)
    out_summary_tsv = os.path.join(outdir_biophys, out_summary_name)

    mapping = load_gpcr_class_mapping(evidence_tsv)
    if not mapping:
        print("[ERROR] Mapping evidence PDB__id -> GPCRdb__class is empty.", file=sys.stderr)
        sys.exit(1)

    residue_rows = []
    pocket_summary = []

    n_total_rows = 0
    n_skipped_empty = 0

    with open(input_tsv, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")

        required_cols = {
            "pdb_id", "target_chain", "peptide_chain",
            "n_pocket_residues", "pocket_residues", "pocket_gpcrdb_numbers",
            "n_mapped", "n_unmapped", "unmapped_pocket_residues"
        }
        missing = required_cols - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"[ERROR] Input pockets file missing columns: {sorted(missing)}. "
                f"Found: {reader.fieldnames}"
            )

        for row in reader:
            n_total_rows += 1

            pdb_id_raw = row["pdb_id"]
            pdb_id = (pdb_id_raw or "").strip()
            pdb_norm = norm_pdb(pdb_id)

            target_chain = row["target_chain"]
            peptide_chain = row["peptide_chain"]

            pocket_residues = row["pocket_residues"]
            pocket_gpcrdb_numbers = row["pocket_gpcrdb_numbers"]
            n_mapped = int(row.get("n_mapped", "0") or 0)
            n_unmapped = int(row.get("n_unmapped", "0") or 0)
            unmapped_pocket_residues = row.get("unmapped_pocket_residues", "NA")
            n_pocket_residues = row.get("n_pocket_residues", "0")

            # skip empty pockets
            if pocket_residues == "NA" or str(n_pocket_residues).strip() in ("0", "", "NA"):
                n_skipped_empty += 1
                continue

            gpcr_class = mapping.get(pdb_norm, "NA")

            residues_list = parse_pocket_residues(pocket_residues)
            gpcrdb_list = parse_gpcrdb_numbers(pocket_gpcrdb_numbers)
            unmapped_set = parse_unmapped(unmapped_pocket_residues)

            # Strategy:
            # - If n_unmapped == 0: we assume 1:1 order alignment residues_list <-> gpcrdb_list
            # - Else: we assign gpcrdb numbers only to residues not in unmapped_set, in order.
            gpcrdb_iter_idx = 0
            per_pocket_props = []

            for r in residues_list:
                is_unmapped = int(r["raw"] in unmapped_set) if n_unmapped > 0 else 0

                if is_unmapped:
                    gpcrdb = "NA"
                else:
                    if gpcrdb_iter_idx < len(gpcrdb_list):
                        gpcrdb = gpcrdb_list[gpcrdb_iter_idx]
                        gpcrdb_iter_idx += 1
                    else:
                        gpcrdb = "NA"

                props = classify_aa(r["aa"])
                out_row = {
                    "pdb_id": pdb_id,
                    "gpcr_class": gpcr_class,
                    "target_chain": target_chain,
                    "peptide_chain": peptide_chain,
                    "pocket_chain": r["chain"],
                    "pocket_resi": r["resi"],
                    "aa3": r["aa3"],
                    "aa": r["aa"],
                    "gpcrdb": gpcrdb,
                    "is_unmapped_gpcrdb": is_unmapped,
                    "is_aromatic": props["is_aromatic"],
                    "is_pos": props["is_pos"],
                    "is_neg": props["is_neg"],
                    "is_polar": props["is_polar"],
                    "is_hydrophobic": props["is_hydrophobic"],
                    "is_other": props["is_other"],
                    "kd": props["kd"],
                }
                residue_rows.append(out_row)
                per_pocket_props.append(out_row)

            # Summary per pocket
            if per_pocket_props:
                kds = [x["kd"] for x in per_pocket_props]
                n = len(per_pocket_props)
                mean_kd_val = mean_ignore_nan(kds)
                summary = {
                    "pdb_id": pdb_id,
                    "gpcr_class": gpcr_class,
                    "target_chain": target_chain,
                    "peptide_chain": peptide_chain,
                    "n_residues": n,
                    "n_gpcrdb_mapped": sum(1 for x in per_pocket_props if x["gpcrdb"] != "NA"),
                    "mean_kd": round(mean_kd_val, 3) if not math.isnan(mean_kd_val) else "NA",
                    "frac_aromatic": round(sum(x["is_aromatic"] for x in per_pocket_props)/n, 4),
                    "frac_pos": round(sum(x["is_pos"] for x in per_pocket_props)/n, 4),
                    "frac_neg": round(sum(x["is_neg"] for x in per_pocket_props)/n, 4),
                    "frac_polar": round(sum(x["is_polar"] for x in per_pocket_props)/n, 4),
                    "frac_hydrophobic": round(sum(x["is_hydrophobic"] for x in per_pocket_props)/n, 4),
                }

                gpcrdb_positions = [x["gpcrdb"] for x in per_pocket_props if x["gpcrdb"] != "NA"]
                c = Counter(gpcrdb_positions)
                summary["top_gpcrdb_positions"] = ",".join([f"{k}:{v}" for k, v in c.most_common(10)]) if c else "NA"

                pocket_summary.append(summary)

    # Export residue-level
    residue_fields = [
        "pdb_id", "gpcr_class",
        "target_chain", "peptide_chain",
        "pocket_chain", "pocket_resi", "aa3", "aa",
        "gpcrdb", "is_unmapped_gpcrdb",
        "is_aromatic", "is_pos", "is_neg", "is_polar", "is_hydrophobic", "is_other",
        "kd"
    ]
    with open(out_residue_tsv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=residue_fields, delimiter="\t")
        w.writeheader()
        w.writerows(residue_rows)

    # Export pocket summary
    summary_fields = [
        "pdb_id", "gpcr_class",
        "target_chain", "peptide_chain",
        "n_residues", "n_gpcrdb_mapped",
        "mean_kd", "frac_aromatic", "frac_pos", "frac_neg", "frac_polar", "frac_hydrophobic",
        "top_gpcrdb_positions"
    ]
    with open(out_summary_tsv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields, delimiter="\t")
        w.writeheader()
        w.writerows(pocket_summary)

    # Quick summary
    unmapped_class = sum(1 for x in pocket_summary if x.get("gpcr_class") in ("NA", "", None))
    class_counts = Counter([x.get("gpcr_class", "NA") or "NA" for x in pocket_summary])

    print("[DONE] Biophys + GPCR class annotation written in:", outdir_biophys)
    print("  -", out_residue_tsv, f"(rows={len(residue_rows)})")
    print("  -", out_summary_tsv, f"(pockets={len(pocket_summary)})")
    print(f"[INFO] Input rows: {n_total_rows} | skipped empty pockets: {n_skipped_empty}")
    print(f"[INFO] Pockets with gpcr_class=NA: {unmapped_class}")
    print("[INFO] gpcr_class counts (by-pocket):")
    for k, v in class_counts.most_common():
        print(f"  {k}: {v}")

# ===============================
# CLI
# ===============================

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Annotate GPCR peptide binding pockets with biophys + KD + GPCRdb numbers + GPCRdb class (single script)."
    )
    ap.add_argument("--input_tsv", required=True, help="peptide_ligands_gpcr.pockets.gpcrdb.tsv")
    ap.add_argument("--evidence", required=True, help="gpcr_70_evidence_table.tsv (needs PDB__id, GPCRdb__class)")
    ap.add_argument("--outdir_biophys", required=True, help="Directory where biophys annotation TSV outputs will be written")

    ap.add_argument("--out_residue_name", default="pocket_biophys_by_residue.tsv",
                    help="Output filename (residue-level) inside outdir_biophys")
    ap.add_argument("--out_summary_name", default="pocket_biophys_by_pocket.tsv",
                    help="Output filename (pocket-level) inside outdir_biophys")

    args = ap.parse_args()
    main(
        input_tsv=args.input_tsv,
        evidence_tsv=args.evidence,
        outdir_biophys=args.outdir_biophys,
        out_residue_name=args.out_residue_name,
        out_summary_name=args.out_summary_name,
    )
