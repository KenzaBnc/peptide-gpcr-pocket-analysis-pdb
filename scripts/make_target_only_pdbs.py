#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_target_only_pdbs.py  (altloc-safe)

Crée des PDB "target-only" (uniquement la chaîne GPCR cible) depuis le cache mmCIF.

Entrée TSV: colonnes minimales
  - pdb_id
  - target_chain

Sortie:
  <out_dir>/<pdb_id>.pdb

Notes altloc:
- mmCIF encode souvent "pas d'altloc" comme "." ou "?"
- On les considère comme équivalents à "" (donc on les garde)
- Fallback auto: si filtrage altloc vide la chaîne, on ré-essaie sans filtrage

Usage:
python3 make_target_only_pdbs.py \
  --in_tsv peptide_ligands_gpcr.tsv \
  --cif_cache cif_cache \
  --out_dir target_pdbs \
  --keep_altloc A \
  --overwrite
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

import gemmi


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_tsv", required=True, help="TSV contenant pdb_id + target_chain")
    ap.add_argument("--cif_cache", required=True, help="Dossier contenant <pdb_id>.cif")
    ap.add_argument("--out_dir", required=True, help="Dossier de sortie target-only PDB")
    ap.add_argument("--overwrite", action="store_true", help="Écrase les PDB existants")
    ap.add_argument("--keep_altloc", default="A", help="Altloc à garder (ex: A). Vide => pas de filtre.")
    return ap.parse_args()


def read_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        if not r.fieldnames:
            raise SystemExit(f"TSV vide/sans header: {path}")
        return [{k: (v if v is not None else "") for k, v in row.items()} for row in r]


def collect_pairs(rows: List[Dict[str, str]]) -> List[Tuple[str, str]]:
    seen: Set[Tuple[str, str]] = set()
    out: List[Tuple[str, str]] = []
    for r in rows:
        pdb_id = (r.get("pdb_id", "") or "").strip().lower()
        chain = (r.get("target_chain", "") or "").strip()
        if not pdb_id or not chain or chain == "NA":
            continue
        key = (pdb_id, chain)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _new_model_one() -> gemmi.Model:
    try:
        return gemmi.Model("1")
    except TypeError:
        return gemmi.Model(1)


def _norm_altloc(altloc_value) -> str:
    """
    Normalise altloc (gemmi.Atom.altloc).
    mmCIF peut utiliser '.', '?' pour "pas d'altloc".
    Gemmi peut aussi renvoyer '\x00' selon les builds.
    """
    alt = (altloc_value or "")
    # altloc peut être un char (pas forcément str classique)
    try:
        alt = str(alt)
    except Exception:
        alt = ""
    alt = alt.strip()
    if alt in ("", ".", "?", "\x00"):
        return ""
    return alt


def extract_target_chain_to_new_structure(
    structure: gemmi.Structure,
    target_chain: str,
    keep_altloc: str = "A",
    allow_fallback_no_altloc_filter: bool = True,
) -> gemmi.Structure:
    if len(structure) == 0:
        raise RuntimeError("Structure vide (0 models)")

    model0 = structure[0]
    ch = model0.find_chain(target_chain)
    if ch is None:
        raise RuntimeError(f"Chaîne '{target_chain}' introuvable dans le mmCIF")

    keep_altloc = (keep_altloc or "").strip()

    def _build(filter_altloc: bool) -> gemmi.Structure:
        out = gemmi.Structure()
        out.name = structure.name
        out.cell = structure.cell

        new_model = _new_model_one()
        new_chain = gemmi.Chain(target_chain)

        for res in ch:
            new_res = gemmi.Residue()
            new_res.name = res.name
            new_res.seqid = gemmi.SeqId(res.seqid.num, res.seqid.icode)

            for at in res:
                alt = _norm_altloc(at.altloc)

                if filter_altloc and keep_altloc:
                    # on garde: pas d'altloc ("") OU altloc == keep_altloc
                    if alt not in ("", keep_altloc):
                        continue

                new_at = gemmi.Atom()
                new_at.name = at.name
                new_at.pos = gemmi.Position(at.pos.x, at.pos.y, at.pos.z)
                new_at.occ = at.occ
                new_at.b_iso = at.b_iso
                new_at.altloc = at.altloc
                new_at.element = at.element
                new_at.charge = at.charge
                new_res.add_atom(new_at)

            if len(new_res) > 0:
                new_chain.add_residue(new_res)

        # chaîne vide ?
        if sum(1 for _ in new_chain) == 0:
            raise RuntimeError("target_chain_empty_after_copy")

        new_model.add_chain(new_chain)
        out.add_model(new_model)

        try:
            out.setup_entities()
        except Exception:
            pass
        try:
            out.remove_empty_chains()
        except Exception:
            pass

        return out

    # 1) essai avec filtrage altloc si demandé
    if keep_altloc:
        try:
            return _build(filter_altloc=True)
        except RuntimeError as e:
            if str(e) == "target_chain_empty_after_copy" and allow_fallback_no_altloc_filter:
                # 2) fallback: sans filtrage altloc
                return _build(filter_altloc=False)
            raise

    # keep_altloc vide => pas de filtre
    return _build(filter_altloc=False)


def main():
    args = parse_args()
    in_tsv = Path(args.in_tsv)
    cif_cache = Path(args.cif_cache)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_tsv(in_tsv)
    pairs = collect_pairs(rows)
    if not pairs:
        raise SystemExit("Aucun (pdb_id, target_chain) trouvé dans le TSV.")

    print(f"[INFO] n_unique_pairs={len(pairs)} from {in_tsv}")

    n_ok = 0
    n_fail = 0
    n_fallback = 0

    for i, (pdb_id, chain) in enumerate(pairs, start=1):
        cif_path = cif_cache / f"{pdb_id}.cif"
        out_pdb = out_dir / f"{pdb_id}.pdb"

        if out_pdb.exists() and out_pdb.stat().st_size > 0 and not args.overwrite:
            print(f"[{i}/{len(pairs)}] SKIP {pdb_id} chain={chain} (exists)")
            n_ok += 1
            continue

        if not cif_path.exists():
            print(f"[{i}/{len(pairs)}] FAIL {pdb_id} missing {cif_path}")
            n_fail += 1
            continue

        try:
            st = gemmi.read_structure(str(cif_path))

            # On détecte si on a eu besoin du fallback en regardant si la version filtrée échoue
            used_fallback = False
            try:
                target_only = extract_target_chain_to_new_structure(
                    structure=st,
                    target_chain=chain,
                    keep_altloc=args.keep_altloc,
                    allow_fallback_no_altloc_filter=False,  # test strict d'abord
                )
            except RuntimeError as e:
                if str(e) == "target_chain_empty_after_copy":
                    used_fallback = True
                    target_only = extract_target_chain_to_new_structure(
                        structure=st,
                        target_chain=chain,
                        keep_altloc=args.keep_altloc,
                        allow_fallback_no_altloc_filter=True,
                    )
                else:
                    raise

            target_only.write_pdb(str(out_pdb))
            if used_fallback:
                n_fallback += 1
                print(f"[{i}/{len(pairs)}] OK*  {pdb_id} chain={chain} -> {out_pdb} (fallback: no altloc filter)")
            else:
                print(f"[{i}/{len(pairs)}] OK   {pdb_id} chain={chain} -> {out_pdb}")

            n_ok += 1

        except Exception as e:
            print(f"[{i}/{len(pairs)}] FAIL {pdb_id} chain={chain} :: {e}")
            n_fail += 1

    print(f"[DONE] ok={n_ok} fail={n_fail} fallback_used={n_fallback} out_dir={out_dir}")


if __name__ == "__main__":
    main()
