#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
annotate_pockets_with_gpcrdb.py

Annoter peptide_ligands_gpcr.pockets.tsv avec les numéros génériques GPCRdb
en utilisant gpcrdb_numbering.mapping.tsv (produit par gpcrdb_numbering_from_target_pdbs.py).

Input pockets.tsv:
  colonnes attendues: pdb_id, target_chain, peptide_chain, n_pocket_residues, pocket_residues
  pocket_residues format: "R:123(ASP),R:124(GLU),R:125A(THR),..."

Input mapping.tsv:
  colonnes attendues:
    pdb_id, target_chain, chain, resnum, icode, gpcrdb_generic_number

Output:
  pockets.gpcrdb.tsv avec colonnes ajoutées:
    pocket_gpcrdb_numbers, n_mapped, n_unmapped, unmapped_pocket_residues
"""

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, Tuple, List


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pockets_tsv", required=True, help="peptide_ligands_gpcr.pockets.tsv")
    ap.add_argument("--mapping_tsv", required=True, help="gpcrdb_numbering.mapping.tsv")
    ap.add_argument("--out_tsv", required=True, help="pockets annoté GPCRdb")
    return ap.parse_args()


def read_tsv(path: Path):
    with Path(path).open("r", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        if not r.fieldnames:
            raise SystemExit(f"TSV vide/sans header: {path}")
        return list(r), list(r.fieldnames)


def write_tsv(path: Path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def norm_icode(x: str) -> str:
    x = (x or "").strip()
    if x in ("NA", ".", "?", "\x00"):
        return ""
    return x


def load_mapping(mapping_tsv: Path) -> Dict[Tuple[str, str, str, int, str], str]:
    """
    key:
      (pdb_id, target_chain, chain, resnum, icode) -> gpcrdb_generic_number
    """
    rows, _ = read_tsv(mapping_tsv)
    mp: Dict[Tuple[str, str, str, int, str], str] = {}

    for r in rows:
        pdb_id = (r.get("pdb_id", "") or "").strip().lower()
        target_chain = (r.get("target_chain", "") or "").strip()
        chain = (r.get("chain", "") or "").strip()
        resnum = int(r.get("resnum", "0") or 0)
        icode = norm_icode(r.get("icode", ""))
        gnum = (r.get("gpcrdb_generic_number", "") or "").strip()

        if not pdb_id or not target_chain or not chain or resnum <= 0 or not gnum:
            continue
        mp[(pdb_id, target_chain, chain, resnum, icode)] = gnum

    return mp


# Exemple tokens:
#   R:123(ASP)
#   R:123A(THR)
RES_TOKEN_RE = re.compile(
    r"^(?P<chain>[^:]+):(?P<resnum>\d+)(?P<icode>[A-Za-z]?)\((?P<resname>[^)]+)\)$"
)


def parse_res_token(tok: str):
    tok = (tok or "").strip()
    if not tok or tok == "NA":
        return None
    m = RES_TOKEN_RE.match(tok)
    if not m:
        return None
    chain = m.group("chain").strip()
    resnum = int(m.group("resnum"))
    icode = (m.group("icode") or "").strip()
    return chain, resnum, icode


def split_pocket_residues(pocket_residues: str) -> List[str]:
    s = (pocket_residues or "").strip()
    if not s or s == "NA":
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def main():
    args = parse_args()

    pockets_rows, pockets_fields = read_tsv(Path(args.pockets_tsv))
    if not pockets_rows:
        raise SystemExit("pockets_tsv vide")

    mp = load_mapping(Path(args.mapping_tsv))
    print(f"[INFO] loaded mapping entries: {len(mp)}")

    # Ajouter colonnes
    new_cols = [
        "pocket_gpcrdb_numbers",
        "n_mapped",
        "n_unmapped",
        "unmapped_pocket_residues",
    ]
    for c in new_cols:
        if c not in pockets_fields:
            pockets_fields.append(c)

    n_rows = 0
    n_total_res = 0
    n_total_mapped = 0

    for r in pockets_rows:
        pdb_id = (r.get("pdb_id", "") or "").strip().lower()
        target_chain = (r.get("target_chain", "") or "").strip()

        toks = split_pocket_residues(r.get("pocket_residues", ""))
        gpcrdb_nums = []
        unmapped = []

        for tok in toks:
            parsed = parse_res_token(tok)
            if parsed is None:
                unmapped.append(tok)
                continue
            chain, resnum, icode = parsed
            key = (pdb_id, target_chain, chain, resnum, norm_icode(icode))
            gnum = mp.get(key, "NA")
            if gnum == "NA":
                unmapped.append(tok)
            else:
                gpcrdb_nums.append(gnum)

        r["pocket_gpcrdb_numbers"] = ",".join(gpcrdb_nums) if gpcrdb_nums else "NA"
        r["n_mapped"] = str(len(gpcrdb_nums))
        r["n_unmapped"] = str(len(unmapped))
        r["unmapped_pocket_residues"] = ",".join(unmapped) if unmapped else "NA"

        n_rows += 1
        n_total_res += len(toks)
        n_total_mapped += len(gpcrdb_nums)

    write_tsv(Path(args.out_tsv), pockets_rows, pockets_fields)
    print(f"[DONE] wrote {args.out_tsv}")
    print(f"[STATS] rows={n_rows} total_res={n_total_res} mapped={n_total_mapped} unmapped={n_total_res - n_total_mapped}")


if __name__ == "__main__":
    main()
