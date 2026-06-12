#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
annotate_contacts_with_gpcrdb.py

Ajoute gpcrdb_number à chaque ligne de peptide_ligands_gpcr.contacts.tsv
en se basant sur gpcrdb_numbering.mapping.tsv.

Inputs:
- contacts.tsv
  colonnes attendues (min): pdb_id, target_chain, target_resnum, target_icode
  et idéalement: target_res_chain (présent dans ton pipeline v3.3)
- mapping.tsv
  colonnes attendues:
    pdb_id, target_chain, chain, resnum, icode, gpcrdb_generic_number

Output:
- contacts.gpcrdb.tsv avec colonne gpcrdb_number

usage : 
python3 annotate_contacts_with_gpcrdb.py \
  --contacts_tsv peptide_ligands_gpcr.contacts.tsv \
  --mapping_tsv gpcrdb_numbering/gpcrdb_numbering.mapping.tsv \
  --out_tsv peptide_ligands_gpcr.contacts.gpcrdb.tsv \
  --write_stats
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, Tuple
from collections import defaultdict


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contacts_tsv", required=True)
    ap.add_argument("--mapping_tsv", required=True)
    ap.add_argument("--out_tsv", required=True)
    ap.add_argument("--write_stats", action="store_true", help="Ecrit un petit TSV stats à côté")
    return ap.parse_args()


def norm_icode(x: str) -> str:
    x = (x or "").strip()
    if x in ("NA", ".", "?", "\x00"):
        return ""
    return x


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


def main():
    args = parse_args()

    contacts_tsv = Path(args.contacts_tsv)
    mapping_tsv = Path(args.mapping_tsv)
    out_tsv = Path(args.out_tsv)

    contact_rows, contact_fields = read_tsv(contacts_tsv)
    if not contact_rows:
        raise SystemExit("contacts_tsv vide")

    mp = load_mapping(mapping_tsv)
    print(f"[INFO] loaded mapping entries: {len(mp)}")

    # Ajout colonne
    if "gpcrdb_number" not in contact_fields:
        contact_fields.append("gpcrdb_number")

    n_hit = 0
    n_miss = 0
    miss_by_pdb = defaultdict(int)
    hit_by_pdb = defaultdict(int)

    for r in contact_rows:
        pdb_id = (r.get("pdb_id", "") or "").strip().lower()
        target_chain = (r.get("target_chain", "") or "").strip()

        # IMPORTANT: dans ton contacts.tsv tu as target_res_chain
        chain = (r.get("target_res_chain", "") or "").strip() or target_chain

        resnum = int(r.get("target_resnum", "0") or 0)
        icode = norm_icode(r.get("target_icode", ""))

        # lookup principal
        gnum = mp.get((pdb_id, target_chain, chain, resnum, icode))

        # fallback utile: si l’icode crée un miss inutile, essayer icode=""
        if not gnum and icode != "":
            gnum = mp.get((pdb_id, target_chain, chain, resnum, ""))

        if not gnum:
            gnum = "NA"

        r["gpcrdb_number"] = gnum

        if gnum != "NA":
            n_hit += 1
            hit_by_pdb[pdb_id] += 1
        else:
            n_miss += 1
            miss_by_pdb[pdb_id] += 1

    write_tsv(out_tsv, contact_rows, contact_fields)
    print(f"[DONE] wrote {out_tsv}")
    print(f"[STATS] gpcrdb hits={n_hit} miss={n_miss} total={len(contact_rows)}")

    if args.write_stats:
        stats_path = out_tsv.with_suffix(".stats.tsv")
        rows = []
        all_pdb = sorted(set(list(hit_by_pdb.keys()) + list(miss_by_pdb.keys())))
        for pid in all_pdb:
            rows.append({
                "pdb_id": pid,
                "hits": str(hit_by_pdb.get(pid, 0)),
                "miss": str(miss_by_pdb.get(pid, 0)),
            })
        write_tsv(stats_path, rows, ["pdb_id", "hits", "miss"])
        print(f"[DONE] wrote {stats_path}")


if __name__ == "__main__":
    main()
