#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation des contacts peptide–GPCR : Gemmi/NeighborSearch (via TSV) vs GPCRdb (page /interaction/<PDB>).

Les catégories d’interaction GPCRdb sont conservées telles quelles (pas de simplification) :
  hydrophobic / aromatic / polar / vdw / ionic

Sorties :
  1) out.tsv : comparaison par (pdb_id, gpcrdb_pos)
  2) out.signature_by_pos.tsv : signature consensus par position+segment
  3) out.signature_by_segment.tsv : signature par segment
  4) out.signature_by_segment_and_interaction.tsv : signature segment × type d’interaction

Usage:
    python3 gpcrdb_validate_peptide_interactions_from_html.py \
        pdbs.txt \
        run_out/biophys_annotations/pocket_biophys_by_residue.with_gpcrdb_segments.tsv \
        out/gpcrdb_vs_gemmi.tsv
"""

import re
import sys
import time
import json
from pathlib import Path

import requests
import pandas as pd


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

GPCRDB_INTERACTION_PAGE = "https://gpcrdb.org/interaction/{pdb_id}"
DELAY_S = 0.5
TIMEOUT_S = 30
UA = "Mozilla/5.0 (X11; Linux x86_64) gpcrdb-validate/3.0"


# ─────────────────────────────────────────────────────────────
# UTILS
# ─────────────────────────────────────────────────────────────

def norm_pdb(p: str) -> str:
    return str(p).strip().upper()


def extract_gpcrdb_pos(val) -> str | None:
    """
    Normalise une position GPCRdb vers un format simple:
      - "3x32", "45x51", "12x50", etc.
    GPCRdb fournit parfois "5.36x37" -> converti en "5x37"
    """
    if val is None:
        return None

    s = str(val).strip()
    if not s or s.lower() in {"none", "nan"}:
        return None

    # format étendu type "5.36x37"
    m_ext = re.search(r"(\d+)\s*\.\s*\d+\s*[xX]\s*(\d+)", s)
    if m_ext:
        return f"{m_ext.group(1)}x{m_ext.group(2)}"

    # format simple
    m = re.search(r"(\d+)\s*[xX]\s*(\d+)", s)
    if m:
        return f"{m.group(1)}x{m.group(2)}"

    return None


def standardize_interaction_type_raw(val: str) -> str | None:
    """
    Harmonise le texte brut GPCRdb vers quelques classes intermédiaires :
      hydrophobic / vdw / ionic / aromatic / polar / other
    """
    if val is None:
        return None

    s = str(val).strip().lower()
    if not s:
        return None

    if "hydroph" in s:
        return "hydrophobic"
    if "van der waals" in s or "vdw" in s or "van" in s:
        return "vdw"
    if "ionic" in s or "salt" in s:
        return "ionic"
    if "aromatic" in s or "pi" in s:
        return "aromatic"
    if "polar" in s or "hbond" in s or "hydrogen" in s:
        return "polar"

    return "other"


def segment_from_gpcrdb_pos(gp: str) -> str | None:
    """
    Segmentation canonique dérivée de gpcrdb_pos.
    """
    if not gp:
        return None
    gp = gp.strip()

    m_tm = re.fullmatch(r"([1-8])x(\d+)", gp)
    if m_tm:
        h = int(m_tm.group(1))
        if 1 <= h <= 7:
            return f"TM{h}"
        if h == 8:
            return "H8"
        return None

    m_loop = re.fullmatch(r"(\d{2})x(\d+)", gp)
    if m_loop:
        code = m_loop.group(1)
        return {
            "12": "ICL1",
            "23": "ECL1",
            "34": "ICL2",
            "45": "ECL2",
            "56": "ICL3",
            "67": "ECL3",
            "78": "C-term/ECL3",
        }.get(code, f"Loop{code}")

    return None


# ─────────────────────────────────────────────────────────────
# LOAD GEMMI/PIPELINE TSV
# ─────────────────────────────────────────────────────────────

def load_gemmi_contacts(with_segments_tsv: str) -> pd.DataFrame:
    """
    Charge:
      run_out/biophys_annotations/pocket_biophys_by_residue.with_gpcrdb_segments.tsv

    Produit:
      pdb_id, gpcrdb_pos, segment_gemmi
    """
    df = pd.read_csv(with_segments_tsv, sep="\t", dtype=str)
    if "pdb_id" not in df.columns:
        raise SystemExit(f"[ERROR] Colonne manquante: pdb_id dans {with_segments_tsv}")

    if "gpcrdb_display_generic_number" in df.columns:
        raw = df["gpcrdb_display_generic_number"]
    elif "gpcrdb" in df.columns:
        raw = df["gpcrdb"]
    else:
        raise SystemExit(
            f"[ERROR] Colonnes manquantes: besoin de gpcrdb_display_generic_number OU gpcrdb "
            f"dans {with_segments_tsv}"
        )

    df["pdb_id"] = df["pdb_id"].str.upper().str.strip()
    df["gpcrdb_pos"] = raw.apply(extract_gpcrdb_pos)

    if "gpcrdb_segment" in df.columns:
        df["segment_gemmi"] = df["gpcrdb_segment"].astype(str).str.strip()
        df.loc[df["segment_gemmi"].isin(["", "nan", "None"]), "segment_gemmi"] = None
    else:
        df["segment_gemmi"] = None

    df = df[df["gpcrdb_pos"].notna()].copy()
    return df


# ─────────────────────────────────────────────────────────────
# FETCH GPCRdb interactions (HTML/JS parsing)
# ─────────────────────────────────────────────────────────────

def _extract_js_interactions_array(html: str) -> list[dict]:
    """
    Extrait le tableau JS:
      interactions = [{...}, {...}, ...];
    """
    m = re.search(r"interactions\s*=\s*(\[\{.*?\}\])\s*;", html, flags=re.DOTALL)
    if not m:
        return []

    txt = m.group(1).strip()

    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        txt2 = txt.replace("'", '"')
        try:
            return json.loads(txt2)
        except json.JSONDecodeError:
            return []


def fetch_gpcrdb_interactions_from_html(pdb_id: str) -> pd.DataFrame:
    """
    Retourne un DF avec:
      pdb_id | gpcrdb_pos | segment_gpcrdb | gpcrdb_interaction_types
    Les types sont normalisés (hydrophobic/aromatic/polar/vdw/ionic) sans simplification.
    """
    pid = norm_pdb(pdb_id)
    url = GPCRDB_INTERACTION_PAGE.format(pdb_id=pid)

    headers = {"User-Agent": UA}
    r = requests.get(url, headers=headers, timeout=TIMEOUT_S)
    r.raise_for_status()

    html = r.text
    items = _extract_js_interactions_array(html)
    if not items:
        return pd.DataFrame()

    rows = []
    for it in items:
        gp = extract_gpcrdb_pos(it.get("gpcrdb"))
        if not gp:
            continue

        seg = it.get("segment")
        if seg is not None:
            seg = str(seg).strip() or None

        itype = standardize_interaction_type_raw(it.get("type"))

        rows.append({
            "pdb_id": pid,
            "gpcrdb_pos": gp,
            "segment_gpcrdb": seg,
            "interaction_type": itype,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    agg = (
        df.groupby(["pdb_id", "gpcrdb_pos", "segment_gpcrdb"], dropna=False)
          .agg({
              "interaction_type": lambda s: sorted({x for x in s.dropna().tolist()}),
          })
          .reset_index()
    )

    agg["gpcrdb_interaction_types"] = agg["interaction_type"].apply(
        lambda xs: "; ".join(xs) if xs else None
    )

    agg = agg.drop(columns=["interaction_type"])
    return agg


# ─────────────────────────────────────────────────────────────
# COMPARE
# ─────────────────────────────────────────────────────────────

def compare_contacts_for_pdb(gemmi_df: pd.DataFrame, gpcrdb_df: pd.DataFrame, pdb_id: str) -> pd.DataFrame:
    """
    Compare les positions GPCRdb pour un PDB donné.
    Sortie: 1 ligne par gpcrdb_pos union, avec:
      pdb_id, gpcrdb_pos, segment_gemmi, segment_gpcrdb, source,
      gpcrdb_interaction_types
    """
    pid = norm_pdb(pdb_id)

    gpos = set(gemmi_df.loc[gemmi_df["pdb_id"] == pid, "gpcrdb_pos"].dropna().tolist())
    dpos = set(gpcrdb_df.loc[gpcrdb_df["pdb_id"] == pid, "gpcrdb_pos"].dropna().tolist())

    allpos = sorted(gpos | dpos)

    rows = []
    for pos in allpos:
        in_g = pos in gpos
        in_d = pos in dpos

        if in_g and in_d:
            src = "both"
        elif in_g:
            src = "gemmi_only"
        else:
            src = "gpcrdb_only"

        seg_g = gemmi_df.loc[
            (gemmi_df["pdb_id"] == pid) & (gemmi_df["gpcrdb_pos"] == pos),
            "segment_gemmi"
        ].dropna().unique().tolist()
        seg_g = seg_g[0] if seg_g else None

        sub = gpcrdb_df.loc[(gpcrdb_df["pdb_id"] == pid) & (gpcrdb_df["gpcrdb_pos"] == pos)]

        seg_d = sub["segment_gpcrdb"].dropna().unique().tolist()
        seg_d = seg_d[0] if seg_d else None

        itypes = sub["gpcrdb_interaction_types"].dropna().unique().tolist()
        itypes = itypes[0] if itypes else None

        rows.append({
            "pdb_id": pid,
            "gpcrdb_pos": pos,
            "segment_gemmi": seg_g,
            "segment_gpcrdb": seg_d,
            "source": src,
            "gpcrdb_interaction_types": itypes,
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) != 4:
        print(
            "Usage: gpcrdb_validate_peptide_interactions_from_html.py "
            "<pdbs.txt> <with_segments.tsv> <out.tsv>",
            file=sys.stderr
        )
        sys.exit(2)

    pdbs_txt, with_segments_tsv, out_tsv = sys.argv[1:]
    out_path = Path(out_tsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load PDB list ────────────────────────────────────────
    pdbs = []
    with open(pdbs_txt, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            pdbs.append(norm_pdb(line))

    if not pdbs:
        raise SystemExit(f"[ERROR] Liste PDB vide: {pdbs_txt}")

    print(f"[INFO] {len(pdbs)} structures à traiter")

    # ── Load Gemmi TSV ───────────────────────────────────────
    gemmi_df = load_gemmi_contacts(with_segments_tsv)
    print(f"[INFO] Gemmi rows loaded: {len(gemmi_df)}")

    # ── Fetch GPCRdb /interaction JS ─────────────────────────
    gpcrdb_all = []
    for i, pdb in enumerate(pdbs, 1):
        print(f"  [{i}/{len(pdbs)}] GPCRdb fetch (JS): {pdb}")
        try:
            df = fetch_gpcrdb_interactions_from_html(pdb)
            if not df.empty:
                gpcrdb_all.append(df)
        except requests.RequestException as e:
            print(f"  [WARN] fetch failed {pdb}: {e}", file=sys.stderr)
        time.sleep(DELAY_S)

    if not gpcrdb_all:
        raise SystemExit("[ERROR] Aucune interaction GPCRdb récupérée (JS parsing)")

    gpcrdb_df = pd.concat(gpcrdb_all, ignore_index=True)
    print(f"[INFO] GPCRdb rows: {len(gpcrdb_df)}")

    # ── Compare per PDB ──────────────────────────────────────
    cmp_all = []
    for pdb in pdbs:
        cmp = compare_contacts_for_pdb(gemmi_df, gpcrdb_df, pdb)
        if not cmp.empty:
            cmp_all.append(cmp)

    if not cmp_all:
        raise SystemExit("[ERROR] Aucune comparaison générée")

    cmp_df = pd.concat(cmp_all, ignore_index=True)

    # ── Add canonical segments ───────────────────────────────
    cmp_df["segment_canonical"] = cmp_df["gpcrdb_pos"].apply(segment_from_gpcrdb_pos)
    cmp_df["segment_final"] = (
        cmp_df["segment_canonical"]
        .fillna(cmp_df["segment_gpcrdb"])
        .fillna(cmp_df["segment_gemmi"])
    )

    # ── Export 1: table brute comparaison ────────────────────
    cmp_df.to_csv(out_path, sep="\t", index=False)

    # ── Export 2: signature by position ──────────────────────
    sig_pos = (
        cmp_df.groupby(["gpcrdb_pos", "segment_final", "source"], dropna=False)
              .size()
              .reset_index(name="n_structures")
              .sort_values(["segment_final", "gpcrdb_pos", "source"])
    )
    sig_pos_path = out_path.with_suffix(".signature_by_pos.tsv")
    sig_pos.to_csv(sig_pos_path, sep="\t", index=False)

    # ── Export 3: signature by segment ───────────────────────
    sig_seg = (
        cmp_df.groupby(["segment_final", "source"], dropna=False)
              .size()
              .reset_index(name="n_positions")
              .sort_values(["segment_final", "source"])
    )
    sig_seg_path = out_path.with_suffix(".signature_by_segment.tsv")
    sig_seg.to_csv(sig_seg_path, sep="\t", index=False)

    # ── Export 4: signature segment × interaction type ───────
    tmp = cmp_df.dropna(subset=["gpcrdb_interaction_types"]).copy()
    if len(tmp) > 0:
        tmp["interaction"] = tmp["gpcrdb_interaction_types"].str.split(r"\s*;\s*")
        tmp = tmp.explode("interaction")
        tmp["interaction"] = tmp["interaction"].astype(str).str.strip()
        tmp = tmp[tmp["interaction"] != ""].copy()

        sig_seg_int = (
            tmp.groupby(["segment_final", "interaction", "source"], dropna=False)
               .size()
               .reset_index(name="n_positions")
               .sort_values(["segment_final", "interaction", "source"])
        )
    else:
        sig_seg_int = pd.DataFrame(columns=["segment_final", "interaction", "source", "n_positions"])

    sig_seg_int_path = out_path.with_suffix(".signature_by_segment_and_interaction.tsv")
    sig_seg_int.to_csv(sig_seg_int_path, sep="\t", index=False)

    # ── Quick stats ──────────────────────────────────────────
    total = len(cmp_df)
    n_both = int((cmp_df["source"] == "both").sum())
    n_gemmi = int((cmp_df["source"] == "gemmi_only").sum())
    n_gpcr = int((cmp_df["source"] == "gpcrdb_only").sum())

    print(f"""
[DONE] Outputs:
  - {out_path}
  - {sig_pos_path}
  - {sig_seg_path}
  - {sig_seg_int_path}

[STATS] (positions uniques = lignes du tableau comparaison)
  Total         : {total}
  both          : {n_both}  ({(100*n_both/total if total else 0):.1f}%)
  gemmi_only    : {n_gemmi} ({(100*n_gemmi/total if total else 0):.1f}%)
  gpcrdb_only   : {n_gpcr}  ({(100*n_gpcr/total if total else 0):.1f}%)
""".rstrip())


if __name__ == "__main__":
    main()
