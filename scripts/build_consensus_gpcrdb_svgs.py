#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_consensus_gpcrdb_svgs.py

Colorer et annoter des SVG GPCRdb (snakeplot + helix box) avec un consensus de classe.

Système à 3 niveaux de conservation biophysique :
  Fort    (≥ 70 %) : couleur biophys pleine, contour noir épais (3 px)
  Modéré  (50–70 %) : couleur biophys pleine, contour fin (#444)
  Faible  (< 50 %) : cercle gris clair, contour tiret #888, texte AA masqué
"""

import argparse
import csv
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd


# ============================================================
# Palette cohérente avec le reste de la pipeline
# ============================================================

BIOPHYS_COLORS = {
    "hydrophobic":  "#222222",
    "aromatic":     "#f28e2b",
    "positive":     "#1f4ed8",
    "negative":     "#e31a1c",
    "polar":        "#c61fc6",
    "structural":   "#2bd11f",
    "NA":           "#bdbdbd",
}

TEXT_ON_DARK = {"#222222", "#1f4ed8", "#e31a1c", "#c61fc6", "#2bd11f"}

# Couleurs des niveaux de conservation
LEVEL_STYLES = {
    "strong":   {"stroke": "#000000", "stroke_width": 3.0, "dasharray": "none",  "fill_override": None},
    "moderate": {"stroke": "#444444", "stroke_width": 1.0, "dasharray": "none",  "fill_override": None},
    "weak":     {"stroke": "#888888", "stroke_width": 1.0, "dasharray": "4 2",   "fill_override": "#e0e0e0"},
}


# ============================================================
# Helpers généraux
# ============================================================

def safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def clamp(x, a, b):
    return max(a, min(b, x))


def parse_ids(s: str):
    if s is None:
        return []
    s = str(s).strip()
    if not s:
        return []
    parts = [p.strip() for p in s.replace(";", ",").split(",")]
    return [p for p in parts if p]


def text_color_for_fill(fill_hex: str):
    return "white" if str(fill_hex).lower() in {c.lower() for c in TEXT_ON_DARK} else "black"


def biophys_level(pct: float) -> str:
    """Retourne le niveau de conservation biophysique (float ∈ [0,1])."""
    if pct >= 0.70:
        return "strong"
    if pct >= 0.50:
        return "moderate"
    return "weak"


def normalize_major_biophys(x: str) -> str:
    if x is None:
        return "NA"
    s = str(x).strip()
    if not s:
        return "NA"
    # Aliases de compatibilité avec anciens labels
    aliases = {
        "polar_uncharged":    "polar",
        "nonpolar_aliphatic": "hydrophobic",
        "other":              "NA",
    }
    return aliases.get(s, s)


# ============================================================
# SVG utilities
# ============================================================

SVG_NS = {"svg": "http://www.w3.org/2000/svg"}
ET.register_namespace("", "http://www.w3.org/2000/svg")


def load_svg(svg_path: str):
    tree = ET.parse(svg_path)
    root = tree.getroot()
    return tree, root


def find_by_id(root, node_id: str):
    if not node_id:
        return None
    node = root.find(f".//*[@id='{node_id}']")
    if node is not None:
        return node
    for elem in root.iter():
        if elem.attrib.get("id") == node_id:
            return elem
    return None


def set_style_attr(elem, key, value):
    if elem is None:
        return
    style = elem.attrib.get("style", "")
    style_dict = {}
    if style:
        for part in style.split(";"):
            if ":" in part:
                k, v = part.split(":", 1)
                style_dict[k.strip()] = v.strip()
    style_dict[key] = str(value)
    elem.attrib["style"] = "; ".join(f"{k}:{v}" for k, v in style_dict.items())


def set_fill(elem, color):
    if elem is None:
        return
    elem.set("fill", color)
    set_style_attr(elem, "fill", color)


def set_stroke(elem, color, width=None):
    if elem is None:
        return
    elem.set("stroke", color)
    set_style_attr(elem, "stroke", color)
    if width is not None:
        elem.set("stroke-width", str(width))
        set_style_attr(elem, "stroke-width", str(width))


def set_opacity(elem, opacity):
    if elem is None:
        return
    elem.set("opacity", f"{opacity:.3f}")
    set_style_attr(elem, "opacity", f"{opacity:.3f}")


def set_text(elem, txt):
    if elem is None:
        return
    elem.text = str(txt)


def set_text_style(elem, fill=None, font_weight=None, font_size=None):
    if elem is None:
        return
    if fill is not None:
        elem.set("fill", fill)
        set_style_attr(elem, "fill", fill)
    if font_weight is not None:
        elem.set("font-weight", str(font_weight))
        set_style_attr(elem, "font-weight", str(font_weight))
    if font_size is not None:
        elem.set("font-size", str(font_size))
        set_style_attr(elem, "font-size", str(font_size))


def add_title_child(elem, text):
    if elem is None:
        return
    for child in list(elem):
        if child.tag.endswith("title"):
            elem.remove(child)
    title = ET.Element("title")
    title.text = str(text)
    elem.insert(0, title)


def expand_svg_canvas_for_legend(root, extra_width=280):
    view_box = root.attrib.get("viewBox", "").strip()
    if view_box:
        parts = view_box.replace(",", " ").split()
        if len(parts) == 4:
            x0, y0, w, h = map(float, parts)
            root.set("viewBox", f"{x0} {y0} {w + extra_width} {h}")
            return x0 + w + 20
    width_attr = root.attrib.get("width", "").strip()
    try:
        width_val = float(str(width_attr).replace("px", ""))
        root.set("width", str(width_val + extra_width))
        return width_val + 20
    except Exception:
        return 20


def add_svg_legend(root, class_label: str, x=20, y=20):
    """Légende avec palette biophysique + explication des 3 niveaux de conservation."""
    g = ET.SubElement(root, "g", attrib={"id": f"legend_{class_label.replace(' ', '_')}"})

    # Titre
    t = ET.SubElement(g, "text", attrib={
        "x": str(x), "y": str(y),
        "font-size": "14", "font-weight": "bold", "fill": "black",
    })
    t.text = f"{class_label} consensus pocket"

    # ── Palette biophysique ──────────────────────────────────
    biophys_items = [
        ("hydrophobic (A/V/I/L/M/C)", BIOPHYS_COLORS["hydrophobic"]),
        ("aromatic (F/W/Y)",           BIOPHYS_COLORS["aromatic"]),
        ("positive (K/R/H)",           BIOPHYS_COLORS["positive"]),
        ("negative (D/E)",             BIOPHYS_COLORS["negative"]),
        ("polar (S/T/N/Q)",            BIOPHYS_COLORS["polar"]),
        ("structural (G/P)",           BIOPHYS_COLORS["structural"]),
    ]

    yy = y + 22
    for label, color in biophys_items:
        ET.SubElement(g, "rect", attrib={
            "x": str(x), "y": str(yy - 11),
            "width": "14", "height": "14",
            "fill": color, "stroke": "black", "stroke-width": "0.8",
        })
        txt = ET.SubElement(g, "text", attrib={
            "x": str(x + 22), "y": str(yy),
            "font-size": "12", "fill": "black",
        })
        txt.text = label
        yy += 20

    yy += 10

    # ── Niveaux de conservation biophysique ─────────────────
    hdr = ET.SubElement(g, "text", attrib={
        "x": str(x), "y": str(yy),
        "font-size": "13", "font-weight": "bold", "fill": "#333333",
    })
    hdr.text = "Biophysical class conservation*"
    yy += 18

    level_items = [
        ("strong",   "≥ 70 %: solid fill, thick border"),
        ("moderate", "50–70 %: solid fill, thin border"),
        ("weak",     "< 50 %: heterogeneous (grey, dashed)"),
    ]
    for level, desc in level_items:
        st = LEVEL_STYLES[level]
        fill = "white"
        ET.SubElement(g, "circle", attrib={
            "cx": str(x + 7), "cy": str(yy - 4),
            "r": "7",
            "fill": fill,
            "stroke": st["stroke"],
            "stroke-width": str(st["stroke_width"]),
            "stroke-dasharray": st["dasharray"],
        })
        txt = ET.SubElement(g, "text", attrib={
            "x": str(x + 22), "y": str(yy),
            "font-size": "11", "fill": "black",
        })
        txt.text = desc
        yy += 20

    note = ET.SubElement(g, "text", attrib={
        "x": str(x), "y": str(yy + 6),
        "font-size": "10", "fill": "#666666",
        "font-style": "italic",
    })
    note.text = "* % of structures sharing the dominant class"
    yy2 = yy + 18
    note2 = ET.SubElement(g, "text", attrib={
        "x": str(x), "y": str(yy2),
        "font-size": "10", "fill": "#666666",
        "font-style": "italic",
    })
    note2.text = "  (regardless of the exact amino acid)"


# ============================================================
# Lecture des tables
# ============================================================

def load_consensus_table(path: str):
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            gp = str(row.get("gpcrdb_pos", "")).strip()
            if not gp:
                continue
            top_aa = str(row.get("top_aa", "")).strip()
            freq = safe_float(row.get("freq_structures"), None)
            if freq is None:
                freq = safe_float(row.get("freq_top_aa"), 0.0)
            if freq is None:
                freq = 0.0
            major = normalize_major_biophys(row.get("major_biophys_class"))
            segment = str(row.get("segment_gpcrdb", "")).strip()
            out[gp] = {
                "top_aa": top_aa,
                "freq_structures": freq,
                "major_biophys_class": major,
                "segment_gpcrdb": segment,
            }
    return out


def load_mapping_table(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            gp = str(row.get("gpcrdb_pos", "")).strip()
            if not gp:
                continue
            rows.append({
                "gpcrdb_pos": gp,
                "circle_id": str(row.get("circle_id", "")).strip(),
                "text_id": str(row.get("text_id", "")).strip(),
                "gpcrdb_label_id": str(row.get("gpcrdb_label_id", "")).strip(),
                "top_aa_label_id": str(row.get("top_aa_label_id", "")).strip(),
                "freq_label_id": str(row.get("freq_label_id", "")).strip(),
                "segment_label_id": str(row.get("segment_label_id", "")).strip(),
                "extra_ids": parse_ids(row.get("extra_ids", "")),
            })
    return rows


# ============================================================
# Biophysical conservation helpers
# ============================================================

def _pocket_biophys(row):
    if str(row.get("is_aromatic", "0")) == "1":    return "aromatic"
    if str(row.get("is_pos", "0")) == "1":          return "positive"
    if str(row.get("is_neg", "0")) == "1":          return "negative"
    if str(row.get("is_polar", "0")) == "1":        return "polar"
    if str(row.get("is_hydrophobic", "0")) == "1":  return "hydrophobic"
    aa = str(row.get("aa", "")).upper()
    if aa in ("G", "P"):
        return "structural"
    return "other"


def compute_biophys_conservation(pocket_tsv: str, class_simple: str) -> dict:
    """
    Returns {gpcrdb_pos: pct_biophys} where pct_biophys ∈ [0, 1] is the
    fraction of structures at that position sharing the dominant biophys class.
    """
    df = pd.read_csv(pocket_tsv, sep="\t", dtype=str)
    df["class_simple"] = df["gpcr_class"].apply(
        lambda x: "A" if "A" in str(x) else "B"
    )
    df = df[df["class_simple"] == class_simple].copy()
    df["biophys"] = df.apply(_pocket_biophys, axis=1)

    result = {}
    for pos, grp in df.groupby("gpcrdb"):
        vc = grp["biophys"].value_counts()
        result[str(pos)] = vc.iloc[0] / len(grp) if len(grp) > 0 else 0.0
    return result


def compute_dominant_biophys(pocket_tsv: str, class_simple: str) -> dict:
    """Returns {gpcrdb_pos: dominant_biophys_class_str}."""
    df = pd.read_csv(pocket_tsv, sep="\t", dtype=str)
    df["class_simple"] = df["gpcr_class"].apply(
        lambda x: "A" if "A" in str(x) else "B"
    )
    df = df[df["class_simple"] == class_simple].copy()
    df["biophys"] = df.apply(_pocket_biophys, axis=1)

    result = {}
    for pos, grp in df.groupby("gpcrdb"):
        vc = grp["biophys"].value_counts()
        result[str(pos)] = vc.index[0] if len(vc) > 0 else "other"
    return result


def compute_dominant_aa(pocket_tsv: str, class_simple: str) -> dict:
    """Returns {gpcrdb_pos: dominant_aa_letter} from pocket data.
    Ties broken alphabetically (consistent with freq TSV sort)."""
    df = pd.read_csv(pocket_tsv, sep="\t", dtype=str)
    df["class_simple"] = df["gpcr_class"].apply(
        lambda x: "A" if "A" in str(x) else "B"
    )
    df = df[df["class_simple"] == class_simple].copy()

    result = {}
    for pos, grp in df.groupby("gpcrdb"):
        vc = grp["aa"].value_counts()
        if len(vc) == 0:
            continue
        max_count = vc.iloc[0]
        top_aas = sorted(vc[vc == max_count].index.tolist())
        result[str(pos)] = top_aas[0]
    return result


def load_dominant_aa_from_freq(freq_tsv: str) -> dict:
    """Returns {gpcrdb_pos: dominant_aa} from a pre-computed frequency TSV.
    The TSV must have columns gpcrdb_pos, aa, freq — first row per position is dominant."""
    import re
    df = pd.read_csv(freq_tsv, sep="\t", dtype=str)
    result = {}
    for pos, grp in df.groupby("gpcrdb_pos", sort=False):
        pos_clean = str(pos).strip().replace("×", "x")
        m = re.fullmatch(r"(\d{1,2})x(\d{1,3})", pos_clean)
        if m:
            pos_clean = f"{int(m.group(1))}x{int(m.group(2))}"
        grp_sorted = grp.sort_values(["freq", "aa"], ascending=[False, True])
        aa = str(grp_sorted.iloc[0]["aa"]).strip()
        if pos_clean not in result:
            result[pos_clean] = aa
    return result


# ============================================================
# Projection consensus -> SVG
# ============================================================

def apply_consensus_to_svg(
    svg_path: str,
    mapping_tsv: str,
    consensus_tsv: str,
    out_svg: Path,
    class_label: str,
    diagram_type: str,
    replace_text_with_top_aa: bool = True,
    recolor_text: bool = True,
    add_title_tooltips: bool = True,
    add_legend: bool = True,
    biophys_conservation: dict = None,
    dominant_biophys: dict = None,
    dominant_aa: dict = None,
):
    consensus = load_consensus_table(consensus_tsv)
    mapping_rows = load_mapping_table(mapping_tsv)

    tree, root = load_svg(svg_path)
    n_colored = 0

    # ── Grisage de tous les cercles : seules les positions consensus ressortiront ──
    SVG_NS_URI = "http://www.w3.org/2000/svg"
    for ns_tag in [f"{{{SVG_NS_URI}}}circle", "circle"]:
        for elem in root.iter(ns_tag):
            set_fill(elem, "#e8e8e8")
            set_stroke(elem, "#cccccc", width=0.5)
            set_style_attr(elem, "stroke-dasharray", "none")
            set_style_attr(elem, "stroke-width", "0.5")

    # ── Grisage du texte de tous les résidus ─────────────────────────────────
    for ns_tag in [f"{{{SVG_NS_URI}}}text", "text"]:
        for elem in root.iter(ns_tag):
            cls = elem.attrib.get("class", "")
            if "rtext" in cls:
                set_style_attr(elem, "fill", "#cccccc")
                elem.attrib.pop("fill", None)

    for row in mapping_rows:
        gp = row["gpcrdb_pos"]
        if gp not in consensus:
            continue

        info = consensus[gp]
        top_aa  = info["top_aa"]
        freq    = info["freq_structures"]
        major   = info["major_biophys_class"]
        segment = info["segment_gpcrdb"]

        # Use dominant biophys from pocket data only when consensus table has none
        if major == "NA" and dominant_biophys and gp in dominant_biophys:
            major = dominant_biophys[gp]

        # Use dominant AA from pocket data only when consensus table has none
        if not top_aa and dominant_aa and gp in dominant_aa:
            top_aa = dominant_aa[gp]

        # ── Niveau de conservation biophysique ──────────────
        pct = biophys_conservation.get(gp, 0.5) if biophys_conservation else 0.5
        level = biophys_level(pct)
        st = LEVEL_STYLES[level]

        # Consensus positions are always colored (fill_override ignored)
        fill_color = BIOPHYS_COLORS.get(major, BIOPHYS_COLORS["NA"])
        show_text  = True

        circle  = find_by_id(root, row["circle_id"])
        text    = find_by_id(root, row["text_id"])
        gp_label    = find_by_id(root, row["gpcrdb_label_id"])
        top_aa_label = find_by_id(root, row["top_aa_label_id"])
        freq_label  = find_by_id(root, row["freq_label_id"])
        seg_label   = find_by_id(root, row["segment_label_id"])
        extras = [find_by_id(root, x) for x in row["extra_ids"]]

        if circle is not None:
            set_fill(circle, fill_color)
            set_stroke(circle, st["stroke"], width=st["stroke_width"])
            set_opacity(circle, 1.0)
            circle.set("stroke-dasharray", st["dasharray"])
            set_style_attr(circle, "stroke-dasharray", st["dasharray"])
            set_style_attr(circle, "display", "inline")

            if add_title_tooltips:
                add_title_child(
                    circle,
                    f"{class_label} | {diagram_type} | {gp} | "
                    f"AA={top_aa} | biophys={major} | "
                    f"biophys_cons={pct:.0%} ({level}) | seg={segment}"
                )

        if text is not None:
            set_style_attr(text, "display", "inline")
            if show_text:
                if replace_text_with_top_aa and top_aa:
                    set_text(text, top_aa)
                if recolor_text:
                    set_text_style(
                        text,
                        fill=text_color_for_fill(fill_color),
                        font_weight="bold",
                    )
            else:
                # Position hétérogène : masquer l'AA
                set_text(text, "")
                set_style_attr(text, "display", "none")

            if add_title_tooltips:
                add_title_child(
                    text,
                    f"{gp} | AA={top_aa} | biophys={major} | cons={pct:.0%}"
                )

        if gp_label is not None:
            set_text(gp_label, gp)
            set_text_style(gp_label, fill="black", font_weight="bold", font_size="11")

        if top_aa_label is not None and top_aa and show_text:
            set_text(top_aa_label, top_aa)
            set_text_style(top_aa_label, fill=fill_color, font_weight="bold", font_size="11")

        if freq_label is not None:
            set_text(freq_label, f"{freq:.2f}")
            set_text_style(freq_label, fill="dimgray", font_size="10")

        if seg_label is not None and segment:
            set_text(seg_label, segment)
            set_text_style(seg_label, fill="dimgray", font_size="10")

        for ex in extras:
            if ex is None:
                continue
            set_stroke(ex, fill_color, width=max(1.2, st["stroke_width"]))
            set_opacity(ex, 1.0)

        n_colored += 1

    if add_legend:
        legend_x = expand_svg_canvas_for_legend(root, extra_width=300)
        add_svg_legend(root, class_label=class_label, x=legend_x, y=24)

    out_svg.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out_svg, encoding="utf-8", xml_declaration=True)
    print(f"[DONE] {class_label} {diagram_type}: {out_svg} | positions colored={n_colored}")


# ============================================================
# CLI
# ============================================================

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--consensus_a", required=True)
    ap.add_argument("--consensus_b", required=True)
    ap.add_argument("--snakeplot_a_svg", required=True)
    ap.add_argument("--snakeplot_a_map", required=True)
    ap.add_argument("--helixbox_a_svg", required=True)
    ap.add_argument("--helixbox_a_map", required=True)
    ap.add_argument("--snakeplot_b_svg", required=True)
    ap.add_argument("--snakeplot_b_map", required=True)
    ap.add_argument("--helixbox_b_svg", required=True)
    ap.add_argument("--helixbox_b_map", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--pocket_tsv", default=None,
                    help="pocket_biophys_by_residue TSV pour la conservation biophysique")
    ap.add_argument("--freq_a", default=None,
                    help="Freq TSV Class A (position_aa_frequencies) pour le AA dominant")
    ap.add_argument("--freq_b", default=None,
                    help="Freq TSV Class B (position_aa_frequencies) pour le AA dominant")
    return ap.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    conserv_a = compute_biophys_conservation(args.pocket_tsv, "A") if args.pocket_tsv else None
    conserv_b = compute_biophys_conservation(args.pocket_tsv, "B") if args.pocket_tsv else None
    dominant_a = compute_dominant_biophys(args.pocket_tsv, "A") if args.pocket_tsv else None
    dominant_b = compute_dominant_biophys(args.pocket_tsv, "B") if args.pocket_tsv else None
    # Prefer freq TSV for dominant AA (canonical, tie-broken alphabetically)
    dom_aa_a = load_dominant_aa_from_freq(args.freq_a) if args.freq_a else (
        compute_dominant_aa(args.pocket_tsv, "A") if args.pocket_tsv else None)
    dom_aa_b = load_dominant_aa_from_freq(args.freq_b) if args.freq_b else (
        compute_dominant_aa(args.pocket_tsv, "B") if args.pocket_tsv else None)

    apply_consensus_to_svg(
        svg_path=args.snakeplot_a_svg,
        mapping_tsv=args.snakeplot_a_map,
        consensus_tsv=args.consensus_a,
        out_svg=outdir / "Class_A.consensus_snakeplot.svg",
        class_label="Class A",
        diagram_type="snakeplot",
        biophys_conservation=conserv_a,
        dominant_biophys=dominant_a,
        dominant_aa=dom_aa_a,
    )
    apply_consensus_to_svg(
        svg_path=args.helixbox_a_svg,
        mapping_tsv=args.helixbox_a_map,
        consensus_tsv=args.consensus_a,
        out_svg=outdir / "Class_A.consensus_helixbox.svg",
        class_label="Class A",
        diagram_type="helixbox",
        biophys_conservation=conserv_a,
        dominant_biophys=dominant_a,
        dominant_aa=dom_aa_a,
    )
    apply_consensus_to_svg(
        svg_path=args.snakeplot_b_svg,
        mapping_tsv=args.snakeplot_b_map,
        consensus_tsv=args.consensus_b,
        out_svg=outdir / "Class_B.consensus_snakeplot.svg",
        class_label="Class B",
        diagram_type="snakeplot",
        biophys_conservation=conserv_b,
        dominant_biophys=dominant_b,
        dominant_aa=dom_aa_b,
    )
    apply_consensus_to_svg(
        svg_path=args.helixbox_b_svg,
        mapping_tsv=args.helixbox_b_map,
        consensus_tsv=args.consensus_b,
        out_svg=outdir / "Class_B.consensus_helixbox.svg",
        class_label="Class B",
        diagram_type="helixbox",
        biophys_conservation=conserv_b,
        dominant_biophys=dominant_b,
        dominant_aa=dom_aa_b,
    )

    print(f"[DONE] All SVG outputs written to: {outdir}")


if __name__ == "__main__":
    main()
