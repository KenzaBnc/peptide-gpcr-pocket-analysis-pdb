#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pdb_chain_to_gpcrdb_segments.py 

Pipeline:
  (pdb_id, target_chain) -> RCSB GraphQL -> polymer entity_id -> UniProt accession(s)
  -> GPCRdb entry_name (via uniprot_mapping.txt)
  -> GPCRdb residues/extended (segments + display_generic_number)
  -> optional merge into by_residue + diagnostics (missing_threshold)

Inputs:
  --pairs_tsv: TSV with columns at least: pdb_id, target_chain
      e.g. run_out/peptide_ligands_gpcr.pockets.gpcrdb.tsv

Outputs (in --outdir):
  1) pdb_chain_to_uniprot_gpcrdb.tsv
     pdb_id, target_chain, entity_id, uniprot_acc, gpcrdb_entry, status, error_msg

  2) gpcrdb_segments_extended.tsv
     gpcrdb_entry, sequence_number, gpcrdb_short, display_generic_number,
     segment_slug, segment_name, segment_category, amino_acid

  3) gpcrdb_segments_bounds.tsv
     gpcrdb_entry, segment_slug, min_sequence_number, max_sequence_number, n_residues

Optional:
  --by_residue + --out_by_residue
    merges segment_slug + display_generic_number into pocket_biophys_by_residue.tsv.

Diagnostics:
  --missing_threshold (default 0.15)
    after merge, reports fraction of rows with gpcrdb != NA but gpcrdb_segment == NA
    globally + per (pdb_id,target_chain); if above threshold => mapping/merge issue flagged.

Join logic:
- gpcrdb_short derived primarily from display_generic_number ("1.30x30" -> "1x30").
- Merge into by_residue on:
    (gpcrdb_entry, sequence_number)  <->  (gpcrdb_entry, sequence_number)

Notes:
- GPCRdb residues/extended sometimes returns "protein_segment" as a string ("TM1", "ECL2", "N-term"...).
  We coerce it and derive a stable segment_category from the slug.

Usage:
python3 scripts/pdb_chain_to_gpcrdb_segments2.py \
  --pairs_tsv run_out/peptide_ligands_gpcr.pockets.gpcrdb.tsv \
  --outdir run_out/gpcrdb_segments_pipeline \
  --timeout 180 --retries 2 --sleep_s 0.0 \
  --by_residue run_out/biophys_annotations/pocket_biophys_by_residue.tsv \
  --out_by_residue run_out/biophys_annotations/pocket_biophys_by_residue.with_gpcrdb_segments.tsv \
  --missing_threshold 0.15
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests


# -----------------------
# Endpoints
# -----------------------
RCSB_GRAPHQL = "https://data.rcsb.org/graphql"
GPCRDB_RESIDUES_EXTENDED = "https://gpcrdb.org/services/residues/extended/{entry}/"
GPCRDB_UNIPROT_MAPPING_URL = "https://files.gpcrdb.org/uniprot_mapping.txt"


# -----------------------
# HTTP helpers
# -----------------------
def http_get_text(url: str, timeout: int = 30, retries: int = 3, sleep_s: float = 1.0) -> str:
    last = None
    for i in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r.text
        except Exception as e:
            last = e
            time.sleep(sleep_s * (i + 1))
    raise RuntimeError(f"[ERROR] GET failed: {url} ({last})")


def http_get_json(url: str, timeout: int = 30, retries: int = 3, sleep_s: float = 1.0) -> Any:
    last = None
    for i in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            time.sleep(sleep_s * (i + 1))
    raise RuntimeError(f"[ERROR] GET failed: {url} ({last})")


def http_post_json(url: str, payload: Dict[str, Any], timeout: int = 30, retries: int = 3, sleep_s: float = 1.0) -> Any:
    last = None
    for i in range(retries):
        try:
            r = requests.post(url, json=payload, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            time.sleep(sleep_s * (i + 1))
    raise RuntimeError(f"[ERROR] POST failed: {url} ({last})")


def safe_str(x) -> str:
    if x is None:
        return "NA"
    s = str(x).strip()
    return s if s and s.upper() != "NA" else "NA"


def norm_pdb(x) -> Optional[str]:
    s = safe_str(x)
    if s == "NA":
        return None
    return s.lower()


# -----------------------
# RCSB: PDB+chain -> entity_id -> UniProt accession(s)
# -----------------------
def rcsb_entity_and_uniprot_for_chain(
    pdb_id: str,
    chain_id: str,
    timeout: int,
    retries: int,
    sleep_s: float
) -> Tuple[Optional[str], List[str]]:
    """
    Return (entity_id, uniprot_ids_list)
    """
    query = """
    query($entry_id: String!) {
      entry(entry_id: $entry_id) {
        polymer_entities {
          rcsb_id
          rcsb_polymer_entity_container_identifiers {
            uniprot_ids
            reference_sequence_identifiers {
              database_name
              database_accession
            }
          }
          polymer_entity_instances {
            rcsb_polymer_entity_instance_container_identifiers {
              auth_asym_id
            }
          }
        }
      }
    }
    """
    payload = {"query": query, "variables": {"entry_id": pdb_id.upper()}}
    data = http_post_json(RCSB_GRAPHQL, payload, timeout=timeout, retries=retries, sleep_s=sleep_s)

    entry = (data or {}).get("data", {}).get("entry", None)
    if not entry:
        return None, []

    chain_id = chain_id.strip()
    for ent in entry.get("polymer_entities", []) or []:
        ent_id = ent.get("rcsb_id")
        inst = ent.get("polymer_entity_instances", []) or []

        hit = False
        for i in inst:
            ids = (i.get("rcsb_polymer_entity_instance_container_identifiers") or {})
            if ids.get("auth_asym_id") == chain_id:
                hit = True
                break
        if not hit:
            continue

        ids_block = (ent.get("rcsb_polymer_entity_container_identifiers") or {})

        # Prefer uniprot_ids list if present
        uids = ids_block.get("uniprot_ids") or []
        uids = [str(u).strip() for u in uids if u and str(u).strip()]
        if uids:
            return ent_id, uids

        # Fallback: reference_sequence_identifiers
        refs = ids_block.get("reference_sequence_identifiers") or []
        out: List[str] = []
        for r in refs:
            db = (r.get("database_name") or "").lower()
            if db in {"uniprot", "uniprotkb"}:
                acc = r.get("database_accession")
                if acc:
                    out.append(str(acc).strip())
        return ent_id, out

    return None, []


# -----------------------
# GPCRdb: UniProt -> entry_name
# -----------------------
def load_gpcrdb_uniprot_mapping(cache_path: Path,
                                timeout: int, retries: int, sleep_s: float) -> Dict[str, str]:
    """
    Download & cache GPCRdb UniProt mapping file.
    Format: <uniprot_acc> <gpcrdb_entry>
    """
    if cache_path.exists():
        txt = cache_path.read_text()
    else:
        txt = http_get_text(GPCRDB_UNIPROT_MAPPING_URL, timeout=timeout, retries=retries, sleep_s=sleep_s)
        cache_path.write_text(txt)

    mapping: Dict[str, str] = {}
    for line in txt.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            mapping[parts[0].strip()] = parts[1].strip()
    return mapping


# -----------------------
# GPCRdb: residues extended
# -----------------------
def fetch_gpcrdb_extended(entry: str, cache_dir: Path,
                          timeout: int, retries: int, sleep_s: float) -> List[Dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{entry}.extended.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    url = GPCRDB_RESIDUES_EXTENDED.format(entry=entry)
    data = http_get_json(url, timeout=timeout, retries=retries, sleep_s=sleep_s)
    if not isinstance(data, list):
        raise RuntimeError(f"[ERROR] Unexpected residues/extended for {entry}: {type(data)}")
    cache_path.write_text(json.dumps(data, indent=2))
    return data


_RX_SHORT = re.compile(r"^\s*(\d+)\.(\d+)x(\d+)\s*$")  # e.g. "1.30x30" -> 1x30
_RX_SHORT2 = re.compile(r"^\s*(\d+)x(\d+)\s*$")       # e.g. "3x32"


def display_to_short(display_gn: str) -> str:
    """
    Convert GPCRdb display_generic_number to gpcrdb_short used in the pipeline
    Examples:
      "1.30x30" -> "1x30"
      "3.32x32" -> "3x32"
      "NA"/""   -> "NA"
      already "3x32" -> "3x32"
    """
    s = safe_str(display_gn)
    if s == "NA":
        return "NA"

    m = _RX_SHORT.match(s)
    if m:
        helix = m.group(1)
        pos = m.group(3)
        return f"{helix}x{pos}"

    m2 = _RX_SHORT2.match(s)
    if m2:
        return f"{m2.group(1)}x{m2.group(2)}"

    # fallback: keep original (rare formats)
    return s


def _coerce_segment(x: Any) -> Dict[str, Any]:
    """
    Force protein_segment to a dict (robust across GPCRdb payload variants).
    - dict => keep
    - str  => {"slug": s, "name": s, "category": "other"}  (we will overwrite category later)
    - None/other => {}
    """
    if isinstance(x, dict):
        return x
    if isinstance(x, str) and x.strip():
        s = x.strip()
        return {"slug": s, "name": s, "category": "other"}
    return {}


def segment_category_from_slug(slug: str) -> str:
    """
    Derive a stable segment category from the segment slug/name.
    Works well with GPCRdb strings like: TM1, ECL2, ICL3, N-term, C-term, H8.
    """
    s = (slug or "").strip()
    if not s or s.upper() in {"NA", "NAN", "NONE"}:
        return "NA"
    u = s.upper().replace(" ", "").replace("_", "-")

    if u.startswith("TM"):
        return "transmembrane"
    if u.startswith("ECL"):
        return "extracellular_loop"
    if u.startswith("ICL"):
        return "intracellular_loop"
    if u in {"N-TERM", "NTERM", "N-TERMINUS"}:
        return "terminus_extracellular"
    if u in {"C-TERM", "CTERM", "C-TERMINUS"}:
        return "terminus_intracellular"
    if u in {"H8", "HELIX8"}:
        return "helix_8"
    return "other"


def parse_extended_rows(entry: str, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Return rows with stable segment columns:
      - segment_slug, segment_name, segment_category
    plus:
      - gpcrdb_short (join key with your "gpcrdb" column like 3x32)
      - display_generic_number (keep for full label)
    """
    rows: List[Dict[str, Any]] = []

    for res in data:
        if not isinstance(res, dict):
            continue

        seqn = safe_str(res.get("sequence_number") or res.get("sequence_position") or res.get("residue_number"))
        aa = safe_str(res.get("amino_acid") or res.get("aa"))
        display_gn = safe_str(res.get("display_generic_number"))

        ps = _coerce_segment(res.get("protein_segment", None))
        seg_slug = safe_str(ps.get("slug"))
        seg_name = safe_str(ps.get("name") or ps.get("label") or ps.get("slug"))
        seg_cat = segment_category_from_slug(seg_slug)

        gpcrdb_short = display_to_short(display_gn)

        rows.append({
            "gpcrdb_entry": entry,
            "sequence_number": seqn,
            "gpcrdb_short": gpcrdb_short,
            "display_generic_number": display_gn,
            "segment_slug": seg_slug,
            "segment_name": seg_name,
            "segment_category": seg_cat,
            "amino_acid": aa,
        })
    return rows


def write_segment_bounds(seg_df: pd.DataFrame, out_tsv: Path):
    """
    Boundaries per segment_slug.
    """
    df = seg_df.copy()
    df["sequence_number_num"] = pd.to_numeric(df["sequence_number"], errors="coerce")
    df = df.dropna(subset=["sequence_number_num"])
    df["sequence_number_num"] = df["sequence_number_num"].astype(int)

    # keep rows with a usable segment slug
    df["segment_slug2"] = df["segment_slug"].astype(str).str.strip()
    df.loc[df["segment_slug2"].str.upper().isin(["NA", "NAN", "NONE", ""]), "segment_slug2"] = pd.NA
    df = df.dropna(subset=["segment_slug2"])

    b = (df.groupby(["gpcrdb_entry", "segment_slug2"])["sequence_number_num"]
         .agg(min_sequence_number="min", max_sequence_number="max", n_residues="count")
         .reset_index()
         .rename(columns={"segment_slug2": "segment_slug"})
         .sort_values(["gpcrdb_entry", "min_sequence_number", "segment_slug"]))

    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    b.to_csv(out_tsv, sep="\t", index=False)


# -----------------------
# Merge into by_residue
# -----------------------
def merge_segment_into_by_residue(by_residue_tsv: Path,
                                  chain_map_tsv: Path,
                                  seg_table_tsv: Path,
                                  out_by_residue: Path) -> Tuple[int, int]:
    """
    Merge GPCRdb segments into by_residue using:
      (gpcrdb_entry, sequence_number)

    Assumes by_residue has:
      - pdb_id
      - target_chain
      - a residue index column: pocket_resi (preferred) or resnum/target_resnum/etc.

    Writes:
      gpcrdb_segment, gpcrdb_segment_category, gpcrdb_display_generic_number
    Returns:
      (n_rows, n_na_segments)
    """
    df = pd.read_csv(by_residue_tsv, sep="\t", dtype=str)

    # --- Load chain map (pdb_id,target_chain -> gpcrdb_entry)
    chain_map = pd.read_csv(chain_map_tsv, sep="\t", dtype=str)

    # Normalize PDB ids
    if "pdb_id" not in df.columns or "target_chain" not in df.columns:
        raise RuntimeError(f"[ERROR] by_residue missing pdb_id/target_chain. cols={list(df.columns)}")
    if {"pdb_id", "target_chain", "gpcrdb_entry"}.issubset(chain_map.columns) is False:
        raise RuntimeError(f"[ERROR] chain_map missing required cols. cols={list(chain_map.columns)}")

    df["pdb_id_norm"] = df["pdb_id"].astype(str).str.lower().str.strip()
    chain_map["pdb_id_norm"] = chain_map["pdb_id"].astype(str).str.lower().str.strip()

    df = df.merge(
        chain_map[["pdb_id_norm", "target_chain", "gpcrdb_entry"]],
        on=["pdb_id_norm", "target_chain"],
        how="left"
    )

    # --- Pick residue index column for sequence_number
    resi_col = None
    for c in ["pocket_resi", "resnum", "target_resnum", "sequence_number"]:
        if c in df.columns:
            resi_col = c
            break
    if resi_col is None:
        raise RuntimeError(f"[ERROR] Cannot find residue index column in by_residue. cols={list(df.columns)}")

    df["sequence_number"] = df[resi_col].astype(str).str.strip()
    df.loc[df["sequence_number"].str.upper().isin(["NA", "NAN", "NONE", ""]), "sequence_number"] = pd.NA

    # --- Load segments table
    seg_df = pd.read_csv(seg_table_tsv, sep="\t", dtype=str)
    required = {"gpcrdb_entry", "sequence_number", "segment_slug", "segment_category", "display_generic_number"}
    missing = sorted(required - set(seg_df.columns))
    if missing:
        raise RuntimeError(f"[ERROR] seg_table missing cols {missing}. cols={list(seg_df.columns)}")

    seg_df["gpcrdb_entry"] = seg_df["gpcrdb_entry"].astype(str).str.strip()
    seg_df["sequence_number"] = seg_df["sequence_number"].astype(str).str.strip()

    # Optional: drop duplicates in seg_df to avoid exploding merges
    seg_df = seg_df.drop_duplicates(subset=["gpcrdb_entry", "sequence_number"])

    # --- Merge on (gpcrdb_entry, sequence_number)
    df = df.merge(
        seg_df[["gpcrdb_entry", "sequence_number", "segment_slug", "segment_category", "display_generic_number"]],
        on=["gpcrdb_entry", "sequence_number"],
        how="left"
    )

    # --- Final columns
    df["gpcrdb_segment"] = df["segment_slug"].fillna("NA")
    df["gpcrdb_segment_category"] = df["segment_category"].fillna("NA")
    df["gpcrdb_display_generic_number"] = df["display_generic_number"].fillna("NA")

    df.drop(columns=["segment_slug", "segment_category", "display_generic_number", "pdb_id_norm"], inplace=True, errors="ignore")

    out_by_residue = Path(out_by_residue)
    out_by_residue.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_by_residue, sep="\t", index=False)

    n_rows = len(df)
    n_na = int((df["gpcrdb_segment"].astype(str).str.upper() == "NA").sum())

    print("[INFO] Merge complete.")
    print("Total rows:", n_rows)
    print("Missing segments:", n_na)

    return n_rows, n_na


def _compute_missing_fraction(df: pd.DataFrame) -> Tuple[int, int, float]:
    """
    fraction of rows where gpcrdb != NA but gpcrdb_segment == NA
    """
    if df.empty:
        return 0, 0, 1.0
    g = df["gpcrdb"].astype(str).str.strip() if "gpcrdb" in df.columns else pd.Series(["NA"] * len(df))
    seg = df["gpcrdb_segment"].astype(str).str.strip() if "gpcrdb_segment" in df.columns else pd.Series(["NA"] * len(df))

    valid = ~g.str.upper().isin(["NA", "NAN", "NONE", ""])
    total = int(valid.sum())
    if total == 0:
        return 0, 0, 1.0

    missing = int((valid & seg.str.upper().isin(["NA", "NAN", "NONE", ""])).sum())
    frac = missing / total
    return total, missing, frac


def report_merge_diagnostics(merged_tsv: Path, missing_threshold: float, out_diag_tsv: Path):
    """
    Print + write diagnostics after merge:
    - global missing fraction
    - per (pdb_id,target_chain) missing fraction
    """
    df = pd.read_csv(merged_tsv, sep="\t", dtype=str)

    # global
    total, missing, frac = _compute_missing_fraction(df)
    print("\n=== GPCRdb merge diagnostics ===")
    print(f"Rows with gpcrdb != NA           : {total}")
    print(f"Rows missing gpcrdb_segment      : {missing} ({frac*100:.1f}%)")
    if frac > missing_threshold:
        print(f">>> WARNING: missing fraction > {missing_threshold*100:.0f}% -> mapping/merge issue likely")
    else:
        print(f">>> OK: missing fraction <= {missing_threshold*100:.0f}%")

    # per pdb/chain
    if {"pdb_id", "target_chain", "gpcrdb", "gpcrdb_segment"}.issubset(df.columns):
        rows: List[Dict[str, Any]] = []
        for (pdb_id, chain), sub in df.groupby(["pdb_id", "target_chain"], dropna=False):
            t, m, f = _compute_missing_fraction(sub)
            rows.append({
                "pdb_id": pdb_id,
                "target_chain": chain,
                "rows_gpcrdb_not_NA": t,
                "rows_missing_segment": m,
                "missing_fraction": f,
                "flag_mapping_needed": "YES" if f > missing_threshold else "NO",
            })
        diag = pd.DataFrame(rows).sort_values(["flag_mapping_needed", "missing_fraction"], ascending=[False, False])
        out_diag_tsv.parent.mkdir(parents=True, exist_ok=True)
        diag.to_csv(out_diag_tsv, sep="\t", index=False)
        print(f"[DONE] Wrote diagnostics table: {out_diag_tsv}")
    else:
        print("[WARN] cannot compute per-(pdb,chain) diagnostics: missing required columns.")


# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser(description="PDB chain -> UniProt -> GPCRdb entry -> residues/extended segments")
    ap.add_argument("--pairs_tsv", required=True, help="TSV containing pdb_id + target_chain")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--timeout", type=int, default=30)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--sleep_s", type=float, default=1.0)

    ap.add_argument("--by_residue", default=None, help="Optional: pocket_biophys_by_residue.tsv")
    ap.add_argument("--out_by_residue", default=None, help="Optional output merged by_residue with segments")

    ap.add_argument("--missing_threshold", type=float, default=0.15,
                    help="After merge: warn if fraction gpcrdb!=NA but gpcrdb_segment==NA exceeds this threshold")

    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pairs = pd.read_csv(Path(args.pairs_tsv), sep="\t", dtype=str)
    if not {"pdb_id", "target_chain"}.issubset(pairs.columns):
        raise ValueError(f"[ERROR] pairs_tsv must contain pdb_id and target_chain. Found: {list(pairs.columns)}")

    pairs["pdb_id"] = pairs["pdb_id"].map(norm_pdb)
    pairs = pairs.dropna(subset=["pdb_id"])
    pairs["target_chain"] = pairs["target_chain"].astype(str).str.strip()
    pairs = pairs.drop_duplicates(subset=["pdb_id", "target_chain"]).reset_index(drop=True)

    # GPCRdb mapping UniProt -> entry
    mapping_cache = outdir / "gpcrdb_uniprot_mapping.txt"
    uniprot_to_gpcrdb = load_gpcrdb_uniprot_mapping(mapping_cache, args.timeout, args.retries, args.sleep_s)

    cache_dir = outdir / "cache"
    gpcrdb_cache = cache_dir / "gpcrdb_extended"

    chain_rows: List[Dict[str, Any]] = []
    all_entries: set[str] = set()

    for _, r in pairs.iterrows():
        pdb_id = r["pdb_id"]
        chain = r["target_chain"]

        status = "OK"
        entity_id: Optional[str] = None
        uniprot: Optional[str] = None
        gpcrdb_entry: Optional[str] = None
        error_msg = "NA"

        try:
            entity_id, uniprots = rcsb_entity_and_uniprot_for_chain(
                pdb_id, chain, args.timeout, args.retries, args.sleep_s
            )

            if not entity_id:
                status = "NO_ENTITY_FOR_CHAIN"
            else:
                uniprots = [u.strip() for u in (uniprots or []) if u and str(u).strip()]
                if not uniprots:
                    status = "NO_UNIPROT"
                else:
                    # choose first UniProt that exists in GPCRdb mapping
                    chosen = None
                    for u in uniprots:
                        if u in uniprot_to_gpcrdb:
                            chosen = u
                            break

                    if chosen is None:
                        chosen = uniprots[0]
                        status = "UNIPROT_NOT_IN_GPCRDB_MAPPING"
                    else:
                        status = "OK"

                    uniprot = chosen
                    gpcrdb_entry = uniprot_to_gpcrdb.get(uniprot)
                    if not gpcrdb_entry:
                        status = "NO_GPCRDB_ENTRY"

        except Exception as e:
            status = "ERROR"
            gpcrdb_entry = None
            error_msg = str(e)[:200]

        if gpcrdb_entry:
            all_entries.add(gpcrdb_entry)

        chain_rows.append({
            "pdb_id": pdb_id,
            "target_chain": chain,
            "entity_id": safe_str(entity_id),
            "uniprot_acc": safe_str(uniprot),
            "gpcrdb_entry": safe_str(gpcrdb_entry),
            "status": status,
            "error_msg": error_msg
        })

    chain_map_tsv = outdir / "pdb_chain_to_uniprot_gpcrdb.tsv"
    pd.DataFrame(chain_rows).to_csv(chain_map_tsv, sep="\t", index=False)
    print(f"[DONE] Wrote chain mapping: {chain_map_tsv}")
    print(f"       unique (pdb,chain)={len(chain_rows)} | gpcrdb_entries={len(all_entries)}")

    # Fetch extended residues for each GPCRdb entry
    seg_rows: List[Dict[str, Any]] = []
    for entry in sorted(all_entries):
        try:
            data = fetch_gpcrdb_extended(entry, gpcrdb_cache, args.timeout, args.retries, args.sleep_s)
            seg_rows.extend(parse_extended_rows(entry, data))
        except Exception as e:
            print(f"[WARN] failed residues/extended for {entry}: {e}", file=sys.stderr)

    seg_tsv = outdir / "gpcrdb_segments_extended.tsv"
    seg_df = pd.DataFrame(seg_rows)
    seg_df.to_csv(seg_tsv, sep="\t", index=False)
    print(f"[DONE] Wrote GPCRdb extended segments: {seg_tsv} (rows={len(seg_df)})")

    # Segment bounds (domain overview)
    bounds_tsv = outdir / "gpcrdb_segments_bounds.tsv"
    if len(seg_df) > 0:
        write_segment_bounds(seg_df, bounds_tsv)
        print(f"[DONE] Wrote segment bounds: {bounds_tsv}")
    else:
        pd.DataFrame(columns=[
            "gpcrdb_entry", "segment_slug", "min_sequence_number", "max_sequence_number", "n_residues"
        ]).to_csv(bounds_tsv, sep="\t", index=False)
        print(f"[WARN] No segment rows; wrote empty bounds table: {bounds_tsv}")

    # Optional merge into by_residue
    if args.by_residue:
        by_residue_path = Path(args.by_residue)
        if not by_residue_path.exists():
            raise FileNotFoundError(by_residue_path)

        out_by = Path(args.out_by_residue) if args.out_by_residue else outdir / (by_residue_path.stem + ".with_segments.tsv")

        if len(seg_df) == 0:
            print("[WARN] seg table is empty, skipping merge into by_residue.", file=sys.stderr)
        else:
            n_rows, n_na = merge_segment_into_by_residue(by_residue_path, chain_map_tsv, seg_tsv, out_by)
            print(f"[DONE] Wrote merged by_residue: {out_by} (rows={n_rows}, gpcrdb_segment=NA={n_na})")

            # diagnostics
            diag_tsv = outdir / "gpcrdb_merge_diagnostics.tsv"
            report_merge_diagnostics(out_by, args.missing_threshold, diag_tsv)


if __name__ == "__main__":
    main()
