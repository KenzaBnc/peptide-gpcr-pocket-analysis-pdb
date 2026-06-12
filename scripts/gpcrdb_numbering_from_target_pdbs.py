#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gpcrdb_numbering_from_target_pdbs.py

But:
- Prend un TSV "peptide ligands" (ou contacts) contenant au moins: pdb_id, target_chain
- Pour chaque (pdb_id, target_chain):
    - charge target_pdbs/<pdb_id>.pdb (PDB "target-only")
    - envoie le PDB sur l'API GPCRdb assign_generic_numbers
    - récupère directement le PDB numéroté (bfactor sur CA)
    - extrait un mapping (chain, resnum, icode) -> gpcrdb_generic_number (ex: 3x50)
- Ecrit un TSV final + stats + log.
- Cache des PDB numérotés dans <out_dir>/numbered_pdbs

IMPORTANT:
- Ce script fait des requêtes web vers gpcrdb.org (API stable).

Usage:
python3 gpcrdb_numbering_from_target_pdbs.py \
  --in_tsv peptide_ligands_gpcr.tsv \
  --target_pdb_dir target_pdbs \
  --out_dir gpcrdb_numbering \
  --timeout 180 \
  --sleep_s 0.0 \
  --overwrite

Outputs:
- <out_dir>/gpcrdb_numbering.mapping.tsv (mapping par résidu)
- <out_dir>/gpcrdb_numbering.per_pdb_summary.tsv (résumé par PDB)
- <out_dir>/gpcrdb_numbering.stats.tsv (stats globales)
- <out_dir>/gpcrdb_numbering.log
- <out_dir>/numbered_pdbs/<pdb_id>.gpcrdb_numbered.pdb (cache)
"""

import argparse
import csv
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import requests
import gemmi

# API stable (utilisée par AlloViz etc.)
ASSIGN_URL = "https://gpcrdb.org/services/structure/assign_generic_numbers"


# -----------------------------
# Logging (console + file)
# -----------------------------
class SimpleLogger:
    def __init__(self, log_path: Path, also_stdout: bool = True):
        self.log_path = Path(log_path)
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
    ap.add_argument("--in_tsv", required=True, help="TSV contenant au moins: pdb_id, target_chain")
    ap.add_argument("--target_pdb_dir", required=True, help="Dossier contenant les PDB target-only: <pdb_id>.pdb")
    ap.add_argument("--out_dir", required=True, help="Dossier de sortie")
    ap.add_argument("--timeout", type=int, default=180, help="Timeout HTTP (s)")
    ap.add_argument("--sleep_s", type=float, default=0.0, help="Sleep après POST (s) - inutile avec l'API, gardé pour compat.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite les PDB numérotés + TSV outputs")
    ap.add_argument(
        "--only_if_missing",
        action="store_true",
        help="Ne traite un PDB que si son PDB numéroté est manquant (ignore overwrite).",
    )
    ap.add_argument(
        "--max_rows_per_pdb",
        type=int,
        default=0,
        help="Optionnel: limiter le nombre de lignes de mapping écrites par PDB (0 = pas de limite).",
    )
    # mini-robustesse réseau
    ap.add_argument("--retries", type=int, default=2, help="Nb de retries HTTP en cas d'échec transitoire.")
    ap.add_argument("--retry_sleep", type=float, default=2.0, help="Pause (s) entre retries.")
    return ap.parse_args()


def read_tsv(path: Path) -> List[Dict[str, str]]:
    with Path(path).open("r", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        if not r.fieldnames:
            raise SystemExit(f"TSV vide/sans header: {path}")
        return [{k: (v if v is not None else "") for k, v in row.items()} for row in r]


def collect_pairs(rows: List[Dict[str, str]]) -> List[Tuple[str, str]]:
    """
    Collecte les paires uniques (pdb_id, target_chain) depuis le TSV.
    """
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


# -----------------------------
# GPCRdb API call
# -----------------------------
def _post_with_retries(
    sess: requests.Session,
    url: str,
    files,
    timeout: int,
    retries: int,
    retry_sleep: float,
):
    last_err = None
    for attempt in range(1, retries + 2):  # 1 try + N retries
        try:
            r = sess.post(url, files=files, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:
            last_err = e
            if attempt <= retries:
                time.sleep(max(0.0, float(retry_sleep)))
            else:
                raise last_err


def submit_and_download_gpcrdb_numbered_pdb(
    pdb_path: Path,
    out_pdb: Path,
    timeout: int = 180,
    sleep_s: float = 0.0,  # conservé mais pas nécessaire avec l'API
    overwrite: bool = False,
    retries: int = 2,
    retry_sleep: float = 2.0,
    logger: Optional[SimpleLogger] = None,
) -> Path:
    """
    Envoie un PDB à l'API GPCRdb et récupère directement le PDB numéroté.
    Cache: out_pdb
    """
    pdb_path = Path(pdb_path)
    out_pdb = Path(out_pdb)

    if not pdb_path.exists():
        raise FileNotFoundError(pdb_path)

    if (not overwrite) and out_pdb.exists() and out_pdb.stat().st_size > 0:
        return out_pdb

    sess = requests.Session()
    sess.headers.update({"User-Agent": "gpcrdb-numbering-script/1.0"})

    # IMPORTANT: envoyer (filename, bytes, mimetype)
    files = {"pdb_file": (pdb_path.name, pdb_path.read_bytes(), "chemical/x-pdb")}

    resp = _post_with_retries(
        sess=sess,
        url=ASSIGN_URL,
        files=files,
        timeout=timeout,
        retries=retries,
        retry_sleep=retry_sleep,
    )

    if sleep_s and sleep_s > 0:
        time.sleep(float(sleep_s))

    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    out_pdb.write_bytes(resp.content)

    # sanity check: vérifier que c'est bien un PDB et pas du HTML/JSON erreur
    head = resp.content[:300].decode("latin-1", errors="ignore")
    head_low = head.lower()
    if "<html" in head_low:
        raise RuntimeError("GPCRdb API: réponse HTML (probable erreur serveur / input).")
    if ("ATOM" not in head) and ("HEADER" not in head) and ("MODEL" not in head):
        # l'API peut renvoyer un message texte
        raise RuntimeError(f"GPCRdb API: sortie inattendue. Début: {head[:200]!r}")

    if logger:
        logger.info(f"GPCRdb API OK -> {out_pdb}")
    return out_pdb


# -----------------------------
# Mapping extraction
# -----------------------------
def gpcrdb_num_from_bfactor(b: float) -> str:
    """
    Convertit un bfactor CA en GPCRdb generic number (format '3x50', '5x461', etc.)
    Logique calquée sur AlloViz:
      - bfactor > 1  : round(b,2) -> "3x50"
      - bfactor < 0  : bulge -> round(-b + 0.001, 3) -> "5x461"
      - sinon (0..1) : NA (GPCRdb laisse parfois des 1.00 résiduels)
    """
    if b is None:
        return "NA"
    b = float(b)
    if 0.0 <= b <= 1.0:
        return "NA"
    if b > 1.0:
        return f"{b:.2f}".replace(".", "x")
    return f"{(-b + 0.001):.3f}".replace(".", "x")


def build_gpcrdb_mapping_from_numbered_pdb(
    numbered_pdb_path: Path,
    only_chain: Optional[str] = None,
) -> Dict[Tuple[str, int, str], str]:
    """
    Retourne mapping:
      (chain, resnum, icode) -> gpcrdb_number

    only_chain: si fourni, limite à une chaîne (ex: target_chain)
    """
    st = gemmi.read_structure(str(numbered_pdb_path))
    if len(st) == 0:
        return {}

    model = st[0]
    mapping: Dict[Tuple[str, int, str], str] = {}

    for ch in model:
        cid = ch.name
        if only_chain and cid != only_chain:
            continue

        for res in ch:
            resnum = int(res.seqid.num)
            icode = (res.seqid.icode or "").strip()

            ca = None
            for at in res:
                if at.name.strip() == "CA":
                    ca = at
                    break
            if ca is None:
                continue

            g = gpcrdb_num_from_bfactor(float(ca.b_iso))
            if g != "NA":
                mapping[(cid, resnum, icode)] = g

    return mapping


# -----------------------------
# Main pipeline
# -----------------------------
def main():
    args = parse_args()

    in_tsv = Path(args.in_tsv)
    target_pdb_dir = Path(args.target_pdb_dir)
    out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    numbered_dir = out_dir / "numbered_pdbs"
    numbered_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "gpcrdb_numbering.log"
    logger = SimpleLogger(log_path, also_stdout=True)

    try:
        rows = read_tsv(in_tsv)
        pairs = collect_pairs(rows)
        if not pairs:
            logger.error("Aucun (pdb_id, target_chain) trouvé dans in_tsv.")
            raise SystemExit(1)

        logger.info(f"START: n_pairs={len(pairs)} in_tsv={in_tsv}")
        logger.info(f"target_pdb_dir={target_pdb_dir}")
        logger.info(f"out_dir={out_dir}")
        logger.info(
            f"timeout={args.timeout} sleep_s={args.sleep_s} overwrite={bool(args.overwrite)} "
            f"only_if_missing={bool(args.only_if_missing)} retries={args.retries} retry_sleep={args.retry_sleep}"
        )
        logger.info(f"GPCRdb API endpoint: {ASSIGN_URL}")

        mapping_rows: List[Dict[str, str]] = []
        per_pdb_rows: List[Dict[str, str]] = []

        n_ok = 0
        n_fail = 0
        n_missing_target_pdb = 0
        n_empty_mapping = 0
        n_uploaded = 0
        n_cached = 0

        for i, (pdb_id, chain) in enumerate(pairs, start=1):
            target_pdb = target_pdb_dir / f"{pdb_id}.pdb"
            numbered_pdb = numbered_dir / f"{pdb_id}.gpcrdb_numbered.pdb"

            if (not target_pdb.exists()) or target_pdb.stat().st_size == 0:
                logger.warning(f"[{i}/{len(pairs)}] SKIP {pdb_id} chain={chain} (missing target_pdb={target_pdb})")
                n_missing_target_pdb += 1
                continue

            # cache policy
            if args.only_if_missing and numbered_pdb.exists() and numbered_pdb.stat().st_size > 0:
                logger.info(f"[{i}/{len(pairs)}] CACHED {pdb_id} -> {numbered_pdb}")
                n_cached += 1
            else:
                try:
                    logger.info(f"[{i}/{len(pairs)}] GPCRdb API assign_generic_numbers {pdb_id} chain={chain}")
                    submit_and_download_gpcrdb_numbered_pdb(
                        pdb_path=target_pdb,
                        out_pdb=numbered_pdb,
                        timeout=args.timeout,
                        sleep_s=args.sleep_s,
                        overwrite=args.overwrite,
                        retries=int(args.retries),
                        retry_sleep=float(args.retry_sleep),
                        logger=logger,
                    )
                    n_uploaded += 1
                except Exception as e:
                    logger.warning(f"[{i}/{len(pairs)}] FAIL api {pdb_id} chain={chain} :: {e}")
                    n_fail += 1
                    continue

            # mapping extraction
            try:
                mapping = build_gpcrdb_mapping_from_numbered_pdb(numbered_pdb, only_chain=chain)
                if not mapping:
                    logger.warning(f"[{i}/{len(pairs)}] EMPTY mapping {pdb_id} chain={chain} from {numbered_pdb}")
                    n_empty_mapping += 1

                # per-pdb summary
                per_pdb_rows.append(
                    {
                        "pdb_id": pdb_id,
                        "target_chain": chain,
                        "target_pdb": str(target_pdb),
                        "numbered_pdb": str(numbered_pdb),
                        "n_mapped_residues": str(len(mapping)),
                    }
                )

                # mapping rows (ordre déterministe)
                items = sorted(mapping.items(), key=lambda x: (x[0][0], x[0][1], x[0][2], x[1]))
                max_rows = int(args.max_rows_per_pdb or 0)
                if max_rows > 0:
                    items = items[:max_rows]

                for (cid, resnum, icode), gnum in items:
                    mapping_rows.append(
                        {
                            "pdb_id": pdb_id,
                            "target_chain": chain,
                            "chain": cid,
                            "resnum": str(resnum),
                            "icode": icode if icode else "NA",
                            "gpcrdb_generic_number": gnum,
                        }
                    )

                logger.info(f"[{i}/{len(pairs)}] OK {pdb_id} chain={chain} n_mapped={len(mapping)}")
                n_ok += 1

            except Exception as e:
                logger.warning(f"[{i}/{len(pairs)}] FAIL mapping {pdb_id} chain={chain} :: {e}")
                n_fail += 1
                continue

        # Write outputs
        out_mapping = out_dir / "gpcrdb_numbering.mapping.tsv"
        out_per_pdb = out_dir / "gpcrdb_numbering.per_pdb_summary.tsv"
        out_stats = out_dir / "gpcrdb_numbering.stats.tsv"

        logger.info("WRITE outputs...")

        # mapping TSV
        with out_mapping.open("w", encoding="utf-8", newline="") as f:
            fields = ["pdb_id", "target_chain", "chain", "resnum", "icode", "gpcrdb_generic_number"]
            w = csv.DictWriter(f, delimiter="\t", fieldnames=fields)
            w.writeheader()
            for r in mapping_rows:
                w.writerow(r)

        # per pdb TSV
        with out_per_pdb.open("w", encoding="utf-8", newline="") as f:
            fields = ["pdb_id", "target_chain", "target_pdb", "numbered_pdb", "n_mapped_residues"]
            w = csv.DictWriter(f, delimiter="\t", fieldnames=fields)
            w.writeheader()
            for r in per_pdb_rows:
                w.writerow(r)

        # stats TSV
        with out_stats.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["metric", "value"])
            w.writerow(["in_tsv", str(in_tsv)])
            w.writerow(["target_pdb_dir", str(target_pdb_dir)])
            w.writerow(["out_dir", str(out_dir)])
            w.writerow(["n_pairs_requested", len(pairs)])
            w.writerow(["n_missing_target_pdb", n_missing_target_pdb])
            w.writerow(["n_uploaded", n_uploaded])
            w.writerow(["n_cached", n_cached])
            w.writerow(["n_ok", n_ok])
            w.writerow(["n_fail", n_fail])
            w.writerow(["n_empty_mapping", n_empty_mapping])
            w.writerow(["n_mapping_rows_written", len(mapping_rows)])
            w.writerow(["timeout", args.timeout])
            w.writerow(["sleep_s", args.sleep_s])
            w.writerow(["overwrite", str(bool(args.overwrite))])
            w.writerow(["only_if_missing", str(bool(args.only_if_missing))])
            w.writerow(["max_rows_per_pdb", str(int(args.max_rows_per_pdb or 0))])
            w.writerow(["retries", str(int(args.retries))])
            w.writerow(["retry_sleep", str(float(args.retry_sleep))])
            w.writerow(["gpcrdb_api", ASSIGN_URL])

        logger.info(f"DONE: ok={n_ok} fail={n_fail} missing_target_pdb={n_missing_target_pdb}")
        logger.info(f"FILES: mapping={out_mapping} per_pdb={out_per_pdb} stats={out_stats} log={log_path}")

    finally:
        try:
            logger.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
