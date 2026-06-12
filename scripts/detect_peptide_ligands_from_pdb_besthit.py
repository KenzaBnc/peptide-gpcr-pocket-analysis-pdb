#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
detect_peptide_ligands_from_pdb_besthit.py  (NeighborSearch-only, robuste)

Objectif
--------
Produire un TSV final contenant UNIQUEMENT des ligands peptidiques réellement liés
au GPCR (chaine cible) : is_bound == "YES" (min distance <= --cutoff)

Cette version est 100% NeighborSearch (Gemmi) :
- Binding YES/NO + min_distance_to_target : NeighborSearch
- Contacts par résidu (target) + poche (liste de résidus target) : NeighborSearch
=> plus de ContactSearch (qui te faisait 0 contacts pour certains PDB)

Outputs
-------
1) --out_tsv:                 peptide_ligands_gpcr.tsv (YES only)
2) --out_residue_contacts:    peptide_ligands_gpcr.contacts.tsv (par résidu target)
3) --out_pockets:             peptide_ligands_gpcr.pockets.tsv (liste résidus pocket)
4) --out_stats:               peptide_ligands_gpcr.stats.tsv (summary)
5) --out_log:                 peptide_ligands_gpcr.log (console + file)

Exemple
-------
python3 detect_peptide_ligands_from_pdb_besthit.py \
  --pdb_besthit_tsv gpcr_70_pdb_besthit.tsv \
  --cif_cache cif_cache \
  --out_tsv run_out/peptide_ligands_gpcr.tsv \
  --out_stats run_out/peptide_ligands_gpcr.stats.tsv \
  --out_residue_contacts run_out/peptide_ligands_gpcr.contacts.tsv \
  --out_pockets run_out/peptide_ligands_gpcr.pockets.tsv \
  --out_log run_out/peptide_ligands_gpcr.log \
  --cutoff 5

Deps
----
- gemmi
- requests
"""

import argparse
import csv
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import requests
import gemmi


# -----------------------------
# Logging (console + file)
# -----------------------------
class SimpleLogger:
    def __init__(self, log_path: Path, also_stdout: bool = True):
        self.log_path = log_path
        self.also_stdout = also_stdout
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.log_path.open("w", encoding="utf-8")

    def close(self):
        try:
            self._fh.close()
        except Exception:
            pass

    def _write(self, level: str, msg: str):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {level} {msg}"
        self._fh.write(line + "\n")
        self._fh.flush()
        if self.also_stdout:
            print(line, flush=True)

    def info(self, msg: str):
        self._write("INFO", msg)

    def warning(self, msg: str):
        self._write("WARN", msg)

    def error(self, msg: str):
        self._write("ERROR", msg)


# -----------------------------
# Args / IO
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb_besthit_tsv", required=True, help="TSV contenant pdb_seqid (ex pdb|7xjl|F)")
    ap.add_argument("--out_tsv", required=True, help="TSV final: uniquement peptides bound (is_bound=YES)")
    ap.add_argument("--out_stats", required=True, help="TSV stats")
    ap.add_argument("--out_residue_contacts", required=True, help="TSV résidus target en contact (poche)")
    ap.add_argument("--out_pockets", required=True, help="TSV poches (liste résidus) par structure/peptide")
    ap.add_argument("--out_log", required=True, help="Fichier log texte (suivi temps réel + sauvegarde)")

    ap.add_argument("--cutoff", type=float, default=5.0, help="Distance cutoff Å (ONE threshold for everything)")
    ap.add_argument("--cif_cache", required=True, help="Dossier cache mmCIF téléchargés")

    ap.add_argument("--min_len", type=int, default=0, help="Longueur min peptide candidat")
    ap.add_argument("--max_len", type=int, default=80, help="Longueur max peptide candidat")

    ap.add_argument(
        "--target_min_len",
        type=int,
        default=150,
        help="Longueur min (en résidus) pour sélectionner le target via fallback structure-based.",
    )
    ap.add_argument("--timeout", type=int, default=60, help="Timeout HTTP (s)")

    ap.add_argument(
        "--exclude_polymer_regex",
        default=r"("
                r"nanobody|single[- ]domain antibody|vhh|sdab|camelid|"
                r"antibody|immunoglobulin|fab|f\(ab\)|scfv|heavy chain|light chain|"
                r"nb\d+|nb35|"
                r"guanine nucleotide[- ]binding|g protein|g\([a-z0-9]+\)|"
                r"subunit alpha|subunit beta|subunit gamma|"
                r"mini[- ]g|mini[- ]gs|engineered g alpha|"
                r"arrestin|beta[- ]arrestin|"
                r"t4 lysozyme|t4l\b|lysozyme|bril\b|apocytochrome|rubredoxin|"
                r"thioredoxin|maltose[- ]binding|mbp|gfp|"
                r"fusion|chaperone"
                r")",
        help="Regex (case-insensitive) pour exclure des polymères non désirés (nanobodies, G proteins, fusions, etc.)",
    )

    ap.add_argument(
        "--include_peptide_regex",
        default="",
        help="Optionnel: regex d'inclusion via description (ex: peptide|neuropeptide|hormone). Vide = pas de filtre.",
    )

    ap.add_argument(
        "--peptide_tm_max",
        type=int,
        default=1,
        help="Seuil max tm_est autorisé pour un ligand peptidique (tm_est <= peptide_tm_max). Recommandé: 1 (0 si strict).",
    )

    return ap.parse_args()


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def read_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        if not r.fieldnames:
            raise SystemExit(f"[ERROR] TSV vide ou sans header: {path}")
        return [{k: (v if v is not None else "") for k, v in row.items()} for row in r]


def pdb_id_from_seqid(pdb_seqid: str) -> str:
    s = (pdb_seqid or "").strip()
    if not s:
        return ""
    if "|" in s:
        parts = s.split("|")
        if len(parts) >= 2:
            return parts[1].strip().lower()
    s = re.sub(r"^pdb:?", "", s, flags=re.IGNORECASE).strip()
    return s.lower()


# -----------------------------
# Download mmCIF (cache)
# -----------------------------
def download_mmcif(pdb_id: str, cache_dir: Path, timeout: int = 60) -> Optional[Path]:
    pdb_id = (pdb_id or "").lower().strip()
    if not pdb_id:
        return None

    out = cache_dir / f"{pdb_id}.cif"
    if out.exists() and out.stat().st_size > 0:
        return out

    url = f"https://files.rcsb.org/download/{pdb_id}.cif"
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code != 200 or not resp.text:
            return None
        txt = resp.text
        if "data_" not in txt[:500] and len(txt) < 1000:
            return None
        out.write_text(txt, encoding="utf-8")
        return out
    except Exception:
        return None


# -----------------------------
# CIF parsing helpers (Gemmi)
# -----------------------------
def get_best_block(doc: gemmi.cif.Document, pdb_id: str = "") -> gemmi.cif.Block:
    pdb_id = (pdb_id or "").strip().lower()
    if pdb_id:
        for b in doc:
            name = (b.name or "").lower()
            if pdb_id in name:
                return b
    return doc[0]


def _tag_index(table: gemmi.cif.Table, wanted: str) -> int:
    w = wanted.strip()
    tags = list(table.tags)
    for i, t in enumerate(tags):
        if t == w:
            return i
    for i, t in enumerate(tags):
        if t.endswith("." + w):
            return i
    for i, t in enumerate(tags):
        if t.split(".")[-1] == w:
            return i
    return -1


def _table_rows_as_dicts(table: gemmi.cif.Table, colnames: List[str]) -> List[Dict[str, str]]:
    idxs = {c: _tag_index(table, c) for c in colnames}
    out = []
    for row in table:
        d = {}
        for c in colnames:
            j = idxs[c]
            d[c] = str(row[j]).strip() if j >= 0 else ""
        out.append(d)
    return out


def parse_entities_from_cif(
    doc: gemmi.cif.Document,
    pdb_id: str = ""
) -> Tuple[List[Dict[str, str]], Dict[str, Dict[str, str]]]:
    block = get_best_block(doc, pdb_id=pdb_id)

    entities: List[Dict[str, str]] = []
    ent_tab = block.find_mmcif_category("_entity.")
    if ent_tab is not None and len(ent_tab) > 0:
        entities = _table_rows_as_dicts(ent_tab, ["id", "type", "pdbx_description", "formula_weight"])

    entity_poly_by_id: Dict[str, Dict[str, str]] = {}
    poly_tab = block.find_mmcif_category("_entity_poly.")
    if poly_tab is not None and len(poly_tab) > 0:
        poly_rows = _table_rows_as_dicts(
            poly_tab,
            ["entity_id", "type", "pdbx_strand_id", "pdbx_seq_one_letter_code_can", "pdbx_seq_one_letter_code"],
        )
        for r in poly_rows:
            eid = (r.get("entity_id") or "").strip()
            if not eid:
                continue
            entity_poly_by_id[eid] = {
                "poly_type": (r.get("type") or "").strip(),
                "strand_id": (r.get("pdbx_strand_id") or "").strip(),
                "seq_can": (r.get("pdbx_seq_one_letter_code_can") or "").strip(),
                "seq_raw": (r.get("pdbx_seq_one_letter_code") or "").strip(),
            }

    return entities, entity_poly_by_id


def split_chains(strand_id: str) -> List[str]:
    s = (strand_id or "").strip()
    if not s:
        return []
    s = s.replace(";", ",")
    parts = [x.strip() for x in s.split(",") if x.strip()]
    if len(parts) == 1 and " " in parts[0]:
        parts = [x.strip() for x in parts[0].split() if x.strip()]
    return parts


def clean_seq(seq: str) -> str:
    if not seq:
        return ""
    s = seq.replace(";", " ").replace("\n", " ").replace("\r", " ")
    s = re.sub(r"[^A-Za-z]", "", s)
    return s.upper()


def seq_length(seq: str) -> int:
    return len(clean_seq(seq))


def is_polypeptide(poly_type: str) -> bool:
    return "polypeptide" in (poly_type or "").lower()


# -----------------------------
# TM estimate (heuristique)
# -----------------------------
HYDRO = set(list("AILMFWVYC"))


def estimate_tm_count(seq: str) -> int:
    s = clean_seq(seq)
    if len(s) < 19:
        return 0
    hits = [0] * len(s)
    for i in range(0, len(s) - 19 + 1):
        win = s[i: i + 19]
        h = sum(1 for aa in win if aa in HYDRO)
        if h >= 12:
            for k in range(i, i + 19):
                hits[k] = 1
    tm = 0
    i = 0
    while i < len(hits):
        if hits[i] == 1:
            j = i
            while j < len(hits) and hits[j] == 1:
                j += 1
            if (j - i) >= 18:
                tm += 1
            i = j
        else:
            i += 1
    return tm


# -----------------------------
# Structure-based chain length (fallback target)
# -----------------------------
def chain_length_from_structure(structure: gemmi.Structure, chain_id: str) -> int:
    if len(structure) == 0:
        return 0
    model = structure[0]
    ch = model.find_chain(chain_id)
    if ch is None:
        return 0
    return sum(1 for _ in ch)


# -----------------------------
# Regex helpers
# -----------------------------
def compile_optional_regex(pat: str) -> Optional[re.Pattern]:
    p = (pat or "").strip()
    if not p:
        return None
    return re.compile(p, flags=re.IGNORECASE)


def normalize_desc(desc: str) -> str:
    d = (desc or "").strip()
    if d in (".", "?", "NA", "N/A", "null", "None"):
        return ""
    return d


def is_opaque_desc(desc: str) -> bool:
    d = normalize_desc(desc)
    if not d:
        return True
    if len(d) < 4:
        return True
    if re.fullmatch(r"[A-Z0-9_ -]{1,10}", d):
        return True
    if re.fullmatch(r"UNK|UNKNOWN|UNCHARACTERIZED|PROTEIN", d, flags=re.IGNORECASE):
        return True
    return False


# -----------------------------
# Peptide decision
# -----------------------------
def peptide_decision(
    pc: Dict[str, object],
    min_len: int,
    max_len: int,
    peptide_tm_max: int,
    exclude_re: Optional[re.Pattern],
    include_re: Optional[re.Pattern],
) -> Tuple[bool, str, bool]:
    if not bool(pc.get("is_polypeptide", False)):
        return (False, "not_polypeptide", False)

    L = int(pc.get("length", 0) or 0)
    tm = int(pc.get("tm_est", 0) or 0)
    desc = normalize_desc(str(pc.get("entity_desc", "") or ""))

    warn_desc = is_opaque_desc(desc)

    if L < min_len:
        return (False, f"len_lt_{min_len}", warn_desc)
    if L > max_len:
        return (False, f"len_gt_{max_len}", warn_desc)

    if exclude_re and desc and exclude_re.search(desc):
        return (False, "excluded_by_description_regex", warn_desc)

    if include_re:
        if not (desc and include_re.search(desc)):
            return (False, "not_matching_include_regex", warn_desc)

    if tm > peptide_tm_max:
        return (False, "has_tm_helices", warn_desc)

    return (True, "", warn_desc)


def build_categories(
    polymer_chains: List[Dict[str, object]],
    min_len: int,
    max_len: int,
    peptide_tm_max: int,
    exclude_re: Optional[re.Pattern],
    include_re: Optional[re.Pattern],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    target_candidates: List[Dict[str, object]] = []
    peptide_candidates: List[Dict[str, object]] = []

    for pc in polymer_chains:
        poly_ok = bool(pc.get("is_polypeptide", False))
        tm = int(pc.get("tm_est", 0) or 0)
        desc = normalize_desc(str(pc.get("entity_desc", "") or ""))
        excluded_by_desc = bool(exclude_re and desc and exclude_re.search(desc))

        # target candidates (STRICT): polypeptide + many TM + not excluded
        if poly_ok and (tm >= 6) and (not excluded_by_desc):
            target_candidates.append(pc)
            continue

        keep, _, warn_desc = peptide_decision(
            pc=pc,
            min_len=min_len,
            max_len=max_len,
            peptide_tm_max=peptide_tm_max,
            exclude_re=exclude_re,
            include_re=include_re,
        )
        pc["_warn_desc"] = warn_desc

        if keep:
            peptide_candidates.append(pc)

    return target_candidates, peptide_candidates


def choose_target_chain_from_candidates(target_candidates: List[Dict[str, object]]) -> Optional[str]:
    if not target_candidates:
        return None

    best = None
    best_tm = -1
    best_L = -1

    for pc in target_candidates:
        tm = int(pc.get("tm_est", 0) or 0)
        L = int(pc.get("length", 0) or 0)
        ch = str(pc.get("chain_id", "") or "")
        if tm >= 6:
            if (tm > best_tm) or (tm == best_tm and L > best_L):
                best_tm, best_L = tm, L
                best = ch

    return best


def choose_target_chain_fallback_structure(
    structure: gemmi.Structure,
    polymer_chains: List[Dict[str, object]],
    exclude_re: Optional[re.Pattern],
    target_min_len: int,
) -> Optional[Tuple[str, int]]:
    best_ch = None
    best_len = -1

    for pc in polymer_chains:
        if not bool(pc.get("is_polypeptide", False)):
            continue
        ch = str(pc.get("chain_id", "") or "").strip()
        if not ch:
            continue

        desc = normalize_desc(str(pc.get("entity_desc", "") or ""))
        if exclude_re and desc and exclude_re.search(desc):
            continue

        Ls = chain_length_from_structure(structure, ch)
        if Ls < target_min_len:
            continue

        if Ls > best_len:
            best_len = Ls
            best_ch = ch

    if best_ch is None:
        return None
    return (best_ch, best_len)


# -----------------------------
# Residue helpers
# -----------------------------
def residue_key(chain_id: str, res: gemmi.Residue) -> Tuple[str, int, str, str]:
    seqid = res.seqid
    resnum = int(seqid.num)
    icode = (seqid.icode or "").strip()
    resname = (res.name or "").strip()
    return (chain_id, resnum, icode, resname)


def format_residue_id(chain: str, resnum: int, icode: str, resname: str) -> str:
    if icode:
        return f"{chain}:{resnum}{icode}({resname})"
    return f"{chain}:{resnum}({resname})"


# -----------------------------
# NeighborSearch Mark -> Residue (FIX for your crash)
# -----------------------------
def mark_to_residue(mark) -> Optional[gemmi.Residue]:
    """
    Robust conversion of gemmi.Mark to gemmi.Residue.

    Depending on Gemmi/Python bindings, Mark may expose:
      - mark.residue (sometimes)
      - mark.chain + mark.residue_idx (common)
      - mark.chain + mark.residue_index (variant)
    """
    if mark is None:
        return None

    if hasattr(mark, "residue"):
        try:
            return mark.residue
        except Exception:
            pass

    if hasattr(mark, "chain") and hasattr(mark, "residue_idx"):
        try:
            ch = mark.chain
            idx = int(mark.residue_idx)
            return ch[idx]
        except Exception:
            pass

    if hasattr(mark, "chain") and hasattr(mark, "residue_index"):
        try:
            ch = mark.chain
            idx = int(mark.residue_index)
            return ch[idx]
        except Exception:
            pass

    return None


# -----------------------------
# NeighborSearch-only binding + contacts
# -----------------------------
def ns_min_distance_and_pairs(
    structure: gemmi.Structure,
    target_chain_id: str,
    peptide_chain_id: str,
    cutoff: float,
    include_h: bool = False,
) -> Tuple[Optional[float], int, str]:
    """
    Returns:
      - min_dist (Å) among atom pairs within cutoff (None if none within cutoff)
      - n_atom_pairs_within_cutoff
      - msg
    """
    if len(structure) == 0:
        return None, 0, "empty_structure"
    model = structure[0]
    ch_t = model.find_chain(target_chain_id)
    ch_p = model.find_chain(peptide_chain_id)
    if ch_t is None or ch_p is None:
        return None, 0, "missing_chain"

    # build NS on target
    ns = gemmi.NeighborSearch(model, structure.cell, cutoff)
    ns.add_chain(ch_t, include_h=include_h)

    best2 = None
    n_pairs = 0

    for res in ch_p:
        for at in res:
            if (not include_h) and at.element.name == "H":
                continue
            for nb in ns.find_atoms(at.pos):
                # nb is gemmi.Mark
                dx = at.pos.x - nb.pos.x
                dy = at.pos.y - nb.pos.y
                dz = at.pos.z - nb.pos.z
                d2 = dx * dx + dy * dy + dz * dz
                n_pairs += 1
                if best2 is None or d2 < best2:
                    best2 = d2

    if best2 is None:
        return None, 0, "no_pairs_within_cutoff"
    return best2 ** 0.5, n_pairs, "ok"


def ns_contacts_per_target_residue(
    pdb_id: str,
    structure: gemmi.Structure,
    target_chain_id: str,
    peptide_chain_id: str,
    cutoff: float,
    include_h: bool = False,
):
    """
    Returns:
      - residue_rows : list[dict] (per target residue)
      - pocket_row   : dict       (pocket summary)
      - extra        : dict       (debug stats)
      - msg          : str        ("ok" or error message)
    """
    try:
        if len(structure) == 0:
            return [], {
                "pdb_id": pdb_id, "target_chain": target_chain_id, "peptide_chain": peptide_chain_id,
                "n_pocket_residues": "0", "pocket_residues": "NA"
            }, {"n_atom_pairs": 0}, "empty_structure"

        model = structure[0]
        ch_t = model.find_chain(target_chain_id)
        ch_p = model.find_chain(peptide_chain_id)
        if ch_t is None or ch_p is None:
            return [], {
                "pdb_id": pdb_id, "target_chain": target_chain_id, "peptide_chain": peptide_chain_id,
                "n_pocket_residues": "0", "pocket_residues": "NA"
            }, {"n_atom_pairs": 0}, "missing_chain"

        # NeighborSearch built on peptide atoms (we query with target atoms)
        ns = gemmi.NeighborSearch(model, structure.cell, cutoff)
        ns.add_chain(ch_p, include_h=include_h)

        # stats per target residue
        # key = (chain, resnum, icode, resname)
        per_res = {}
        n_atom_pairs = 0

        for tres in ch_t:
            for at in tres:
                if (not include_h) and at.element.name == "H":
                    continue
                for mk in ns.find_atoms(at.pos):
                    cra = mk.to_cra(model)     # ✅ crucial
                    # keep only peptide chain (safety)
                    if cra.chain.name != peptide_chain_id:
                        continue
                    pres = cra.residue

                    # target residue id
                    t_resnum = int(tres.seqid.num)
                    t_icode = (tres.seqid.icode or "").strip()
                    t_resname = (tres.name or "").strip()

                    key = (target_chain_id, t_resnum, t_icode, t_resname)
                    # distance
                    dx = at.pos.x - cra.atom.pos.x
                    dy = at.pos.y - cra.atom.pos.y
                    dz = at.pos.z - cra.atom.pos.z
                    d = (dx*dx + dy*dy + dz*dz) ** 0.5

                    if key not in per_res:
                        per_res[key] = {
                            "min_dist": d,
                            "n_atom_pairs": 1,
                            "pep_res_set": set(),   # store peptide residue ids (resnum+icode+name)
                        }
                    else:
                        per_res[key]["min_dist"] = min(per_res[key]["min_dist"], d)
                        per_res[key]["n_atom_pairs"] += 1

                    # peptide residue id (for n_peptide_residues_contacted)
                    p_resnum = int(pres.seqid.num)
                    p_icode = (pres.seqid.icode or "").strip()
                    p_resname = (pres.name or "").strip()
                    per_res[key]["pep_res_set"].add((peptide_chain_id, p_resnum, p_icode, p_resname))

                    n_atom_pairs += 1

        # build output rows
        residue_rows = []
        pocket_res_ids = []

        for (ch, resnum, icode, resname) in sorted(per_res.keys(), key=lambda x: (x[1], x[2], x[3])):
            d = per_res[(ch, resnum, icode, resname)]
            # pocket token format identical to your previous pipeline
            if icode:
                pocket_res_ids.append(f"{ch}:{resnum}{icode}({resname})")
            else:
                pocket_res_ids.append(f"{ch}:{resnum}({resname})")

            residue_rows.append({
                "pdb_id": pdb_id,
                "target_chain": target_chain_id,
                "peptide_chain": peptide_chain_id,
                "target_res_chain": ch,
                "target_resnum": str(resnum),
                "target_icode": icode if icode else "NA",
                "target_resname": resname,
                "min_dist": f"{float(d['min_dist']):.3f}",
                "n_atom_pairs": str(int(d["n_atom_pairs"])),
                "n_peptide_residues_contacted": str(len(d["pep_res_set"])),
            })

        pocket_row = {
            "pdb_id": pdb_id,
            "target_chain": target_chain_id,
            "peptide_chain": peptide_chain_id,
            "n_pocket_residues": str(len(pocket_res_ids)),
            "pocket_residues": ",".join(pocket_res_ids) if pocket_res_ids else "NA",
        }

        extra = {"n_atom_pairs": n_atom_pairs, "n_target_res_in_pocket": len(pocket_res_ids)}
        return residue_rows, pocket_row, extra, "ok"

    except Exception as e:
        return [], {
            "pdb_id": pdb_id, "target_chain": target_chain_id, "peptide_chain": peptide_chain_id,
            "n_pocket_residues": "0", "pocket_residues": "NA"
        }, {"n_atom_pairs": 0}, f"error:{e}"
        
# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()

    cache_dir = Path(args.cif_cache)
    ensure_dir(cache_dir)

    Path(args.out_tsv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_stats).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_residue_contacts).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_pockets).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_log).parent.mkdir(parents=True, exist_ok=True)

    logger = SimpleLogger(Path(args.out_log), also_stdout=True)

    try:
        exclude_re = compile_optional_regex(args.exclude_polymer_regex)
        include_re = compile_optional_regex(args.include_peptide_regex)

        rows = read_tsv(Path(args.pdb_besthit_tsv))
        if not rows:
            logger.error("pdb_besthit_tsv est vide.")
            raise SystemExit(1)

        pdb_ids = []
        seen = set()
        for r in rows:
            pdb_id = pdb_id_from_seqid(r.get("pdb_seqid", ""))
            if pdb_id and pdb_id not in seen:
                seen.add(pdb_id)
                pdb_ids.append(pdb_id)

        if not pdb_ids:
            logger.error("Aucun PDB ID extrait depuis la colonne pdb_seqid.")
            raise SystemExit(1)

        logger.info(
            f"START: n_input_pdb_ids={len(pdb_ids)} | min_len={args.min_len} max_len={args.max_len} "
            f"peptide_tm_max={args.peptide_tm_max} cutoff={args.cutoff}"
        )

        out_fields = [
            "pdb_id",
            "target_chain",
            "target_tm_est",
            "target_length",
            "peptide_entity_id",
            "peptide_chain",
            "peptide_length",
            "peptide_poly_type",
            "peptide_tm_est",
            "min_distance_to_target",
            "ns_n_atom_pairs_within_cutoff",
            "is_bound",
            "peptide_entity_desc",
        ]

        res_fields = [
            "pdb_id",
            "target_chain",
            "peptide_chain",
            "target_res_chain",
            "target_resnum",
            "target_icode",
            "target_resname",
            "min_dist",
            "n_atom_pairs",
            "n_peptide_residues_contacted",
        ]

        pocket_fields = [
            "pdb_id",
            "target_chain",
            "peptide_chain",
            "n_pocket_residues",
            "pocket_residues",
        ]

        results_yes: List[Dict[str, str]] = []
        residue_rows_all: List[Dict[str, str]] = []
        pocket_rows_all: List[Dict[str, str]] = []

        n_download_fail = 0
        n_entity_parse_fail = 0
        n_structure_parse_fail = 0
        n_no_target = 0
        n_no_entity_poly_table = 0

        n_bound_yes = 0
        n_bound_no = 0
        n_bound_na = 0

        for i, pdb_id in enumerate(pdb_ids, start=1):
            logger.info(f"[{i}/{len(pdb_ids)}] PDB={pdb_id} :: download/parse")

            cif_path = download_mmcif(pdb_id, cache_dir, timeout=args.timeout)
            if cif_path is None:
                n_download_fail += 1
                logger.warning(f"PDB={pdb_id} download_mmcif FAILED")
                continue

            try:
                doc = gemmi.cif.read_file(str(cif_path))
                entities, entity_poly = parse_entities_from_cif(doc, pdb_id=pdb_id)
            except Exception as e:
                n_entity_parse_fail += 1
                logger.warning(f"PDB={pdb_id} parse_entities_from_cif FAILED :: {e}")
                continue

            if not entity_poly:
                n_no_entity_poly_table += 1
                logger.warning(f"PDB={pdb_id} entity_poly table missing/empty")

            ent_desc_by_id: Dict[str, str] = {}
            for ent in entities:
                eid = (ent.get("id") or "").strip()
                if not eid:
                    continue
                ent_desc_by_id[eid] = (ent.get("pdbx_description") or "").strip()

            polymer_chains: List[Dict[str, object]] = []
            for ent in entities:
                if (ent.get("type") or "").strip().lower() != "polymer":
                    continue
                eid = (ent.get("id") or "").strip()
                poly = entity_poly.get(eid, {})
                poly_type = (poly.get("poly_type") or "").strip()
                poly_ok = is_polypeptide(poly_type)

                seq = poly.get("seq_can") or poly.get("seq_raw") or ""
                L = seq_length(seq)
                tm_est = estimate_tm_count(seq) if seq else 0

                chains = split_chains(poly.get("strand_id", ""))
                ent_desc = ent_desc_by_id.get(eid, "")

                for ch in chains:
                    polymer_chains.append(
                        {
                            "entity_id": eid,
                            "chain_id": ch,
                            "poly_type": poly_type,
                            "is_polypeptide": poly_ok,
                            "length": L,
                            "tm_est": tm_est,
                            "entity_desc": ent_desc,
                        }
                    )

            try:
                structure = gemmi.read_structure(str(cif_path))
            except Exception as e:
                n_structure_parse_fail += 1
                logger.warning(f"PDB={pdb_id} read_structure FAILED :: {e}")
                continue

            target_candidates, peptide_candidates = build_categories(
                polymer_chains=polymer_chains,
                min_len=args.min_len,
                max_len=args.max_len,
                peptide_tm_max=args.peptide_tm_max,
                exclude_re=exclude_re,
                include_re=include_re,
            )

            target_chain = choose_target_chain_from_candidates(target_candidates)
            target_tm = "NA"
            target_L = "NA"

            if target_chain is None:
                n_no_target += 1
                target_chain = ""
                logger.warning(f"PDB={pdb_id} NO_TARGET_CHAIN (no tm>=6 polypeptide found)")

                fb = choose_target_chain_fallback_structure(
                    structure=structure,
                    polymer_chains=polymer_chains,
                    exclude_re=exclude_re,
                    target_min_len=args.target_min_len,
                )
                if fb is not None:
                    fb_ch, fb_len = fb
                    target_chain = fb_ch
                    target_L = str(fb_len)
                    target_tm = "NA"
                    logger.warning(f"PDB={pdb_id} TARGET_FALLBACK_STRUCTLEN chain={target_chain} len={target_L}")
                else:
                    logger.warning(f"PDB={pdb_id} TARGET_FALLBACK_STRUCTLEN FAILED (no suitable chain >= {args.target_min_len})")

            if target_chain:
                for pc in polymer_chains:
                    if str(pc.get("chain_id", "")) == target_chain:
                        target_tm = str(int(pc.get("tm_est", 0) or 0))
                        if target_L == "NA":
                            target_L = str(int(pc.get("length", 0) or 0))
                        break

                L_struct = chain_length_from_structure(structure, target_chain)
                if L_struct > 0:
                    target_L = str(L_struct)

                logger.info(f"PDB={pdb_id} target_chain={target_chain} target_len={target_L} target_tm={target_tm}")

            logger.info(f"PDB={pdb_id} peptide_candidates={len(peptide_candidates)}")

            for pc in peptide_candidates:
                peptide_chain = str(pc.get("chain_id", "") or "")
                if not peptide_chain:
                    continue
                if target_chain and peptide_chain == target_chain:
                    continue

                L = int(pc.get("length", 0) or 0)
                tm = int(pc.get("tm_est", 0) or 0)

                if bool(pc.get("_warn_desc", False)):
                    logger.warning(
                        f"PDB={pdb_id} DESC_WARNING(kept) chain={peptide_chain} len={L} tm={tm} "
                        f"desc='{normalize_desc(str(pc.get('entity_desc', '') or ''))}'"
                    )

                min_dist = "NA"
                is_bound = "NA"
                ns_pairs = "0"
                ns_msg = "NA"

                if target_chain:
                    d, n_pairs, msg = ns_min_distance_and_pairs(
                        structure=structure,
                        target_chain_id=target_chain,
                        peptide_chain_id=peptide_chain,
                        cutoff=args.cutoff,
                        include_h=False,
                    )
                    ns_msg = msg
                    ns_pairs = str(int(n_pairs))
                    if d is None:
                        min_dist = f">{args.cutoff:.3f}"
                        is_bound = "NO"
                    else:
                        min_dist = f"{d:.3f}"
                        is_bound = "YES" if d <= args.cutoff else "NO"

                if is_bound == "YES":
                    n_bound_yes += 1
                elif is_bound == "NO":
                    n_bound_no += 1
                else:
                    n_bound_na += 1

                logger.info(
                    f"PDB={pdb_id} NS_BIND chain={peptide_chain} msg={ns_msg} "
                    f"pairs_within_cutoff={ns_pairs} min_dist={min_dist} bound={is_bound}"
                )

                if is_bound != "YES":
                    continue

                row_yes = {
                    "pdb_id": pdb_id,
                    "target_chain": target_chain or "NA",
                    "target_tm_est": target_tm,
                    "target_length": target_L,
                    "peptide_entity_id": str(pc.get("entity_id", "NA") or "NA"),
                    "peptide_chain": peptide_chain,
                    "peptide_length": str(L),
                    "peptide_poly_type": str(pc.get("poly_type", "NA") or "NA"),
                    "peptide_tm_est": str(tm),
                    "min_distance_to_target": min_dist,
                    "ns_n_atom_pairs_within_cutoff": ns_pairs,
                    "is_bound": "YES",
                    "peptide_entity_desc": str(pc.get("entity_desc", "NA") or "NA"),
                }
                results_yes.append(row_yes)

                # pocket / contacts via NeighborSearch-only
                rr, pr, extra, msg = ns_contacts_per_target_residue(
                    pdb_id=pdb_id,
                    structure=structure,
                    target_chain_id=target_chain,
                    peptide_chain_id=peptide_chain,
                    cutoff=args.cutoff,
                    include_h=False,
                )
                if msg != "ok":
                    logger.warning(f"PDB={pdb_id} NS_CONTACTS_FAIL chain={peptide_chain} msg={msg}")
                else:
                    # fill pdb_id in rows
                    for x in rr:
                        x["pdb_id"] = pdb_id
                    pr["pdb_id"] = pdb_id

                    residue_rows_all.extend(rr)
                    pocket_rows_all.append(pr)
                    logger.info(
                        f"PDB={pdb_id} NS_POCKET chain={peptide_chain} "
                        f"n_res={pr['n_pocket_residues']} pairs={extra.get('ns_n_atom_pairs_within_cutoff', 'NA')}"
                    )

        out_tsv = Path(args.out_tsv)
        out_stats = Path(args.out_stats)
        out_res = Path(args.out_residue_contacts)
        out_pockets = Path(args.out_pockets)

        logger.info("WRITE: out_tsv (YES only) / out_residue_contacts / out_pockets / out_stats")

        with out_tsv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, delimiter="\t", fieldnames=out_fields)
            w.writeheader()
            for r in results_yes:
                w.writerow(r)

        with out_res.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, delimiter="\t", fieldnames=res_fields)
            w.writeheader()
            for r in residue_rows_all:
                w.writerow(r)

        with out_pockets.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, delimiter="\t", fieldnames=pocket_fields)
            w.writeheader()
            for r in pocket_rows_all:
                w.writerow(r)

        # Stats
        yes = len(results_yes)
        pdb_yes = sorted({r["pdb_id"] for r in results_yes})
        pdb_any = sorted({pid for pid in pdb_ids})

        counts_yes_by_pdb: Dict[str, int] = {}
        for r in results_yes:
            counts_yes_by_pdb[r["pdb_id"]] = counts_yes_by_pdb.get(r["pdb_id"], 0) + 1
        top = sorted(counts_yes_by_pdb.items(), key=lambda x: x[1], reverse=True)[:20]

        with out_stats.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["metric", "value"])
            w.writerow(["n_input_pdb_ids", len(pdb_ids)])
            w.writerow(["n_unique_pdb_processed_input", len(pdb_any)])
            w.writerow(["n_pdb_with_bound_peptide_yes", len(pdb_yes)])
            w.writerow(["n_rows_written_out_tsv_yes_only", yes])

            w.writerow(["cutoff_A", args.cutoff])
            w.writerow(["peptide_tm_max", args.peptide_tm_max])
            w.writerow(["exclude_regex", args.exclude_polymer_regex if args.exclude_polymer_regex else "NA"])
            w.writerow(["include_regex", args.include_peptide_regex if args.include_peptide_regex else "NA"])

            w.writerow(["n_download_fail", n_download_fail])
            w.writerow(["n_entity_parse_fail", n_entity_parse_fail])
            w.writerow(["n_structure_parse_fail", n_structure_parse_fail])
            w.writerow(["n_no_target_chain_initial_tm_based", n_no_target])
            w.writerow(["n_no_entity_poly_table", n_no_entity_poly_table])

            w.writerow(["n_residue_contact_rows", len(residue_rows_all)])
            w.writerow(["n_pocket_rows", len(pocket_rows_all)])

            w.writerow(["bound_counts_yes", n_bound_yes])
            w.writerow(["bound_counts_no", n_bound_no])
            w.writerow(["bound_counts_na", n_bound_na])

            w.writerow([])
            w.writerow(["top_pdb_by_yes_count", "count_yes"])
            for pid, c in top:
                w.writerow([pid, c])

        logger.info(f"DONE: out_tsv_yes_rows={len(results_yes)} residue_rows={len(residue_rows_all)} pocket_rows={len(pocket_rows_all)}")
        logger.info(f"BOUND (all candidates tested): YES={n_bound_yes} NO={n_bound_no} NA={n_bound_na}")
        logger.info(f"FILES: out_tsv={out_tsv} out_res={out_res} out_pockets={out_pockets} out_stats={out_stats} out_log={Path(args.out_log)}")

    finally:
        try:
            logger.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
