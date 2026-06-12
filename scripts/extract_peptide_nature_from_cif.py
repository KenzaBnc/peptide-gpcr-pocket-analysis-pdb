#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
extract_peptide_nature_from_cif.py

Extrait, depuis les fichiers mmCIF locaux, les informations de nature
des peptides/entités polymériques pour les chaînes peptidiques d'intérêt.

Entrées :
  --inventory_tsv : TSV avec au moins pdb_id et peptide_chain
  --pdb_dir       : dossier contenant les {pdb_id}.cif
  --out_tsv       : fichier de sortie

Exemple :
  python3 extract_peptide_nature_from_cif.py \
    --inventory_tsv out/peptide_inventory.tsv \
    --pdb_dir cif_cache \
    --out_tsv out/peptide_nature_from_cif.tsv
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
from Bio.PDB.MMCIF2Dict import MMCIF2Dict


THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "SEC": "U", "PYL": "O",
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory_tsv", required=True)
    ap.add_argument("--pdb_dir", required=True)
    ap.add_argument("--out_tsv", required=True)
    return ap.parse_args()


def norm_pdb(x) -> str:
    return str(x).strip().lower()


def ensure_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def safe_get_list(d, key):
    return ensure_list(d.get(key, []))


def clean_seq(seq: str) -> str:
    seq = str(seq).replace("\n", "").replace(" ", "").replace(";", "")
    seq = seq.replace("(", "").replace(")", "")
    return seq.strip()


def build_entity_table(cifd):
    ids = safe_get_list(cifd, "_entity.id")
    types = safe_get_list(cifd, "_entity.type")
    descs = safe_get_list(cifd, "_entity.pdbx_description")

    n = max(len(ids), len(types), len(descs), 0)
    rows = []
    for i in range(n):
        rows.append({
            "entity_id": ids[i] if i < len(ids) else "",
            "entity_type": types[i] if i < len(types) else "",
            "entity_description": descs[i] if i < len(descs) else "",
        })
    return pd.DataFrame(rows)


def build_struct_asym_table(cifd):
    asym_ids = safe_get_list(cifd, "_struct_asym.id")
    entity_ids = safe_get_list(cifd, "_struct_asym.entity_id")

    n = max(len(asym_ids), len(entity_ids), 0)
    rows = []
    for i in range(n):
        rows.append({
            "label_asym_id": asym_ids[i] if i < len(asym_ids) else "",
            "entity_id": entity_ids[i] if i < len(entity_ids) else "",
        })
    return pd.DataFrame(rows)


def build_entity_poly_table(cifd):
    entity_ids = safe_get_list(cifd, "_entity_poly.entity_id")
    poly_types = safe_get_list(cifd, "_entity_poly.type")
    seq_raw = safe_get_list(cifd, "_entity_poly.pdbx_seq_one_letter_code")
    seq_can = safe_get_list(cifd, "_entity_poly.pdbx_seq_one_letter_code_can")

    n = max(len(entity_ids), len(poly_types), len(seq_raw), len(seq_can), 0)
    rows = []
    for i in range(n):
        rows.append({
            "entity_id": entity_ids[i] if i < len(entity_ids) else "",
            "polymer_type": poly_types[i] if i < len(poly_types) else "",
            "seq_one_letter_raw": seq_raw[i] if i < len(seq_raw) else "",
            "seq_one_letter_can": seq_can[i] if i < len(seq_can) else "",
        })
    return pd.DataFrame(rows)


def build_chain_entity_map_from_atom_site(cifd):
    """
    Build mappings:
      auth_asym_id -> entity_id
      label_asym_id -> entity_id
    using atom_site rows.
    """
    auth_asym = safe_get_list(cifd, "_atom_site.auth_asym_id")
    label_asym = safe_get_list(cifd, "_atom_site.label_asym_id")
    label_entity = safe_get_list(cifd, "_atom_site.label_entity_id")

    n = max(len(auth_asym), len(label_asym), len(label_entity), 0)

    auth_map = {}
    label_map = {}

    for i in range(n):
        a = auth_asym[i] if i < len(auth_asym) else ""
        l = label_asym[i] if i < len(label_asym) else ""
        e = label_entity[i] if i < len(label_entity) else ""
        if a and e and a not in auth_map:
            auth_map[a] = e
        if l and e and l not in label_map:
            label_map[l] = e

    return auth_map, label_map


def extract_seq_from_atom_site(cifd, chain_id: str) -> str:
    group_pdb = safe_get_list(cifd, "_atom_site.group_PDB")
    label_asym_id = safe_get_list(cifd, "_atom_site.label_asym_id")
    auth_asym_id = safe_get_list(cifd, "_atom_site.auth_asym_id")
    comp_id = safe_get_list(cifd, "_atom_site.label_comp_id")
    seq_id = safe_get_list(cifd, "_atom_site.label_seq_id")

    n = max(len(group_pdb), len(label_asym_id), len(auth_asym_id), len(comp_id), len(seq_id), 0)

    seen = []
    used_ids = set()

    for i in range(n):
        g = group_pdb[i] if i < len(group_pdb) else ""
        la = label_asym_id[i] if i < len(label_asym_id) else ""
        aa = auth_asym_id[i] if i < len(auth_asym_id) else ""
        comp = comp_id[i] if i < len(comp_id) else ""
        sid = seq_id[i] if i < len(seq_id) else ""

        if g != "ATOM":
            continue
        if not (la == chain_id or aa == chain_id):
            continue
        if sid in ("", ".", "?"):
            continue

        key = (sid, comp)
        if key in used_ids:
            continue
        used_ids.add(key)
        try:
            sid_int = int(sid)
        except Exception:
            continue
        seen.append((sid_int, comp))

    seen = sorted(seen, key=lambda x: x[0])
    return "".join(THREE_TO_ONE.get(comp.upper(), "X") for _, comp in seen)


def first_nonempty(*vals):
    for v in vals:
        if str(v).strip():
            return str(v)
    return ""


def main():
    args = parse_args()

    inv = pd.read_csv(args.inventory_tsv, sep="\t", dtype=str).fillna("")
    inv["pdb_id"] = inv["pdb_id"].map(norm_pdb)
    inv["peptide_chain"] = inv["peptide_chain"].astype(str).str.strip()

    rows = []

    for _, r in inv[["pdb_id", "peptide_chain"]].drop_duplicates().iterrows():
        pdb_id = r["pdb_id"]
        pep_chain = r["peptide_chain"]

        cif_path = Path(args.pdb_dir) / f"{pdb_id}.cif"

        result = {
            "pdb_id": pdb_id,
            "peptide_chain": pep_chain,
            "entity_id": "",
            "entity_description": "",
            "entity_type": "",
            "polymer_type": "",
            "peptide_seq_from_cif": "",
            "mapping_source": "",
            "cif_found": 0,
        }

        if not cif_path.exists():
            rows.append(result)
            continue

        result["cif_found"] = 1

        try:
            cifd = MMCIF2Dict(str(cif_path))
        except Exception as e:
            result["entity_description"] = f"PARSE_ERROR: {e}"
            rows.append(result)
            continue

        entity_df = build_entity_table(cifd)
        asym_df = build_struct_asym_table(cifd)
        poly_df = build_entity_poly_table(cifd)
        auth_map, label_map = build_chain_entity_map_from_atom_site(cifd)

        entity_id = ""

        # 1) primary: peptide_chain is auth_asym_id — look it up in atom_site
        if pep_chain in auth_map:
            entity_id = auth_map[pep_chain]
            result["mapping_source"] = "atom_site.auth_asym_id"

        # 2) fallback via _atom_site.label_asym_id
        if not entity_id and pep_chain in label_map:
            entity_id = label_map[pep_chain]
            result["mapping_source"] = "atom_site.label_asym_id"

        # 3) last resort: _struct_asym.label_asym_id (may mismatch auth chain IDs)
        if not entity_id:
            hit = asym_df[asym_df["label_asym_id"] == pep_chain]
            if not hit.empty:
                entity_id = str(hit["entity_id"].iloc[0])
                result["mapping_source"] = "struct_asym.label_asym_id"

        result["entity_id"] = entity_id

        if entity_id:
            ent_row = entity_df[entity_df["entity_id"] == entity_id]
            if not ent_row.empty:
                result["entity_description"] = str(ent_row["entity_description"].iloc[0])
                result["entity_type"] = str(ent_row["entity_type"].iloc[0])

            poly_row = poly_df[poly_df["entity_id"] == entity_id]
            if not poly_row.empty:
                result["polymer_type"] = str(poly_row["polymer_type"].iloc[0])
                seq_can = clean_seq(poly_row["seq_one_letter_can"].iloc[0])
                seq_raw = clean_seq(poly_row["seq_one_letter_raw"].iloc[0])
                result["peptide_seq_from_cif"] = first_nonempty(seq_can, seq_raw)

        # final fallback on atom_site-derived sequence
        if not result["peptide_seq_from_cif"]:
            result["peptide_seq_from_cif"] = extract_seq_from_atom_site(cifd, pep_chain)

        rows.append(result)

    out = pd.DataFrame(rows).sort_values(["pdb_id", "peptide_chain"])
    out.to_csv(args.out_tsv, sep="\t", index=False)

    print(out.to_string(index=False))
    print(f"\n[DONE] wrote {args.out_tsv}")


if __name__ == "__main__":
    main()
