#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_svg_mapping_from_template_resnums.py

Construit un TSV de mapping SVG <-> gpcrdb_pos
en utilisant les IDs de résidus du SVG GPCRdb, supposés basés sur target_resnum.

Hypothèse observée dans tes SVG :
- cercle résidu : id = "<resnum>"
- texte AA      : id = "<resnum>t"

Entrées :
- un SVG template GPCRdb
- un TSV avec les positions consensus (colonne gpcrdb_pos)
- un TSV annotation template avec au moins :
    pdb_id, gpcrdb_pos (ou gpcrdb), target_resnum
- un pdb_id template

Sortie :
- un TSV de mapping compatible avec ton script SVG

Exemple :
python3 scripts/build_svg_mapping_from_template_resnums.py \
  --svg templates/classA_snakeplot.svg \
  --positions_tsv out/consensus_validable/consensus_Class_A_thr50.validable.tsv \
  --annot_tsv run_out/pocket_biophys_by_residue.with_gpcrdb_segments.with_target_resnum.tsv \
  --template_pdb 7MBY \
  --out_tsv templates/classA_snakeplot_mapping.tsv
"""

import argparse
import csv
import re
import xml.etree.ElementTree as ET
from pathlib import Path


def norm_pdb(x: str) -> str:
    return str(x).strip().upper()


def clean_text(s: str) -> str:
    if s is None:
        return ""
    return " ".join(str(s).split()).strip()


def simplify_gpcrdb_pos(pos: str):
    if pos is None:
        return None
    s = str(pos).strip().replace(" ", "").replace("×", "x")
    if not s:
        return None

    m0 = re.fullmatch(r"(\d{1,2})x(\d{1,3})", s)
    if m0:
        return f"{int(m0.group(1))}x{int(m0.group(2))}"

    m = re.fullmatch(r"(\d+)\.(\d+)x(\d+)", s)
    if m:
        a = int(m.group(1))
        b = int(m.group(2))
        c = int(m.group(3))
        if b >= 40:
            return f"{b}x{c}"
        return f"{a}x{c}"

    m2 = re.findall(r"(\d{1,2})x(\d{1,3})", s)
    if m2:
        x, y = m2[-1]
        return f"{int(x)}x{int(y)}"

    return None


def load_positions_tsv(path: str):
    positions = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if "gpcrdb_pos" not in reader.fieldnames:
            raise ValueError(f"[positions_tsv] colonne gpcrdb_pos manquante dans {path}")
        for row in reader:
            gp = simplify_gpcrdb_pos(row.get("gpcrdb_pos"))
            if gp:
                positions.append(gp)

    seen = set()
    out = []
    for p in positions:
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out


def load_svg_ids(svg_path: str):
    tree = ET.parse(svg_path)
    root = tree.getroot()

    ids = set()
    for elem in root.iter():
        elem_id = elem.attrib.get("id")
        if elem_id:
            ids.add(elem_id)
    return ids


def load_template_annotation(annot_tsv: str, template_pdb: str, numbering_tsv: str = None):
    rows = []
    with open(annot_tsv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames = reader.fieldnames or []

        if "pdb_id" not in fieldnames:
            raise ValueError("[annot_tsv] colonne pdb_id manquante")

        # colonne gpcrdb
        gp_col = None
        for cand in ["gpcrdb_pos", "gpcrdb", "gpcrdb_display_generic_number"]:
            if cand in fieldnames:
                gp_col = cand
                break
        if gp_col is None:
            raise ValueError("[annot_tsv] aucune colonne gpcrdb_pos/gpcrdb/gpcrdb_display_generic_number trouvée")

        if "target_resnum" not in fieldnames:
            raise ValueError("[annot_tsv] colonne target_resnum manquante")

        for row in reader:
            if norm_pdb(row.get("pdb_id", "")) != norm_pdb(template_pdb):
                continue

            gp = simplify_gpcrdb_pos(row.get(gp_col))
            resnum = clean_text(row.get("target_resnum"))
            aa = clean_text(row.get("aa"))

            if not gp or not resnum:
                continue

            rows.append({
                "gpcrdb_pos": gp,
                "target_resnum": resnum,
                "aa": aa,
            })

    # déduplication par gpcrdb_pos
    out = {}
    for r in rows:
        gp = r["gpcrdb_pos"]
        if gp not in out:
            out[gp] = r

    # Fallback: use full GPCRdb numbering TSV for positions not found in pocket TSV
    if numbering_tsv:
        try:
            with open(numbering_tsv, "r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter="\t")
                for row in reader:
                    if len(row) < 6:
                        continue
                    pdb_id, _chain1, _chain2, resnum, _icode, gp_raw = row[:6]
                    if norm_pdb(pdb_id) != norm_pdb(template_pdb):
                        continue
                    gp = simplify_gpcrdb_pos(gp_raw)
                    if not gp or not resnum or gp in out:
                        continue
                    out[gp] = {
                        "gpcrdb_pos": gp,
                        "target_resnum": resnum.strip(),
                        "aa": "",
                    }
        except Exception as e:
            print(f"[WARN] could not load numbering_tsv {numbering_tsv}: {e}")

    return out


def build_mapping_rows(svg_ids, positions, template_map):
    rows = []

    for gp in positions:
        circle_id = ""
        text_id = ""

        if gp in template_map:
            resnum = template_map[gp]["target_resnum"]

            cand_circle = str(resnum)
            cand_text = f"{resnum}t"

            if cand_circle in svg_ids:
                circle_id = cand_circle
            if cand_text in svg_ids:
                text_id = cand_text

        rows.append({
            "gpcrdb_pos": gp,
            "circle_id": circle_id,
            "text_id": text_id,
            "gpcrdb_label_id": "",
            "top_aa_label_id": "",
            "freq_label_id": "",
            "segment_label_id": "",
            "extra_ids": "",
        })

    return rows


def write_mapping_tsv(rows, out_tsv: str):
    fieldnames = [
        "gpcrdb_pos",
        "circle_id",
        "text_id",
        "gpcrdb_label_id",
        "top_aa_label_id",
        "freq_label_id",
        "segment_label_id",
        "extra_ids",
    ]
    Path(out_tsv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_tsv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows):
    n_total = len(rows)
    n_circle = sum(1 for r in rows if r["circle_id"])
    n_text = sum(1 for r in rows if r["text_id"])

    print(f"[INFO] positions total   : {n_total}")
    print(f"[INFO] circle_id found : {n_circle}")
    print(f"[INFO] text_id found   : {n_text}")
    print()
    print("Aperçu :")
    for r in rows[:40]:
        print(
            f"{r['gpcrdb_pos']}\t"
            f"circle={r['circle_id']}\t"
            f"text={r['text_id']}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--svg", required=True)
    ap.add_argument("--positions_tsv", required=True)
    ap.add_argument("--annot_tsv", required=True)
    ap.add_argument("--template_pdb", required=True)
    ap.add_argument("--out_tsv", required=True)
    ap.add_argument("--numbering_tsv", default=None,
                    help="Optional full GPCRdb numbering TSV (fallback for positions not in pocket)")
    args = ap.parse_args()

    svg_ids = load_svg_ids(args.svg)
    positions = load_positions_tsv(args.positions_tsv)
    template_map = load_template_annotation(args.annot_tsv, args.template_pdb,
                                            numbering_tsv=args.numbering_tsv)

    rows = build_mapping_rows(svg_ids, positions, template_map)
    write_mapping_tsv(rows, args.out_tsv)

    print_summary(rows)
    print()
    print(f"[DONE] TSV écrit : {args.out_tsv}")


if __name__ == "__main__":
    main()
