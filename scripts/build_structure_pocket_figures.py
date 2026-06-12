#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_structure_pocket_figures.py

Génère, pour une structure PDB donnée, les 3 mêmes figures que le consensus de classe :
  1. Snake plot coloré (SVG)
  2. Helixbox coloré (SVG)
  3. Radar KD (Kyte-Doolittle) — profil de la poche de cette structure
  4. WebLogo — composition en AA de la poche de cette structure

Usage :
  python3 scripts/build_structure_pocket_figures.py \
    --pocket_tsv run_out/biophys_annotations/pocket_biophys_by_residue.with_gpcrdb_segments.with_target_resnum.tsv \
    --pdb_id 8wz2 \
    --outdir out/structure_pocket_figures \
    --templates_dir templates
"""

import argparse
import csv
import re
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import logomaker


# ============================================================
# Constants
# ============================================================

AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")

KD_SCALE = {
    "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5, "M": 1.9, "A": 1.8,
    "G": -0.4, "T": -0.7, "S": -0.8, "W": -0.9, "Y": -1.3, "P": -1.6,
    "H": -3.2, "E": -3.5, "Q": -3.5, "D": -3.5, "N": -3.5, "K": -3.9, "R": -4.5,
}
KD_MIN, KD_MAX = -4.5, 4.5

AA_TO_BIOPHYS = {
    "A": "hydrophobic", "V": "hydrophobic", "I": "hydrophobic",
    "L": "hydrophobic", "M": "hydrophobic", "C": "hydrophobic",
    "F": "aromatic", "W": "aromatic", "Y": "aromatic",
    "K": "positive", "R": "positive", "H": "positive",
    "D": "negative", "E": "negative",
    "S": "polar", "T": "polar", "N": "polar", "Q": "polar",
    "G": "structural", "P": "structural",
}

BIOPHYS_COLORS = {
    "hydrophobic": "#222222",
    "aromatic":    "#f28e2b",
    "positive":    "#1f4ed8",
    "negative":    "#e31a1c",
    "polar":       "#c61fc6",
    "structural":  "#6dbe6d",
    "NA":          "#bdbdbd",
}

TEXT_ON_DARK = {"#222222", "#1f4ed8", "#e31a1c", "#c61fc6", "#6dbe6d"}

LEVEL_STYLES = {
    "strong":   {"stroke": "#000000", "stroke_width": 3.0, "dasharray": "none",  "fill_override": None},
    "moderate": {"stroke": "#444444", "stroke_width": 1.0, "dasharray": "none",  "fill_override": None},
    "weak":     {"stroke": "#888888", "stroke_width": 1.0, "dasharray": "4 2",   "fill_override": "#e0e0e0"},
}


# ============================================================
# Helpers
# ============================================================

def simplify_gpcrdb_pos(pos):
    if pos is None or (isinstance(pos, float) and np.isnan(pos)):
        return None
    s = str(pos).strip().replace(" ", "").replace("×", "x")
    if not s:
        return None
    m0 = re.fullmatch(r"(\d{1,2})x(\d{1,3})", s)
    if m0:
        return f"{int(m0.group(1))}x{int(m0.group(2))}"
    m = re.fullmatch(r"(\d+)\.(\d+)x(\d+)", s)
    if m:
        a, b, c = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return f"{b}x{c}" if b >= 40 else f"{a}x{c}"
    m2 = re.findall(r"(\d{1,2})x(\d{1,3})", s)
    if m2:
        x, y = m2[-1]
        return f"{int(x)}x{int(y)}"
    return None


def gpcrdb_sort_key(pos):
    try:
        a, b = pos.split("x")
        return (int(a), int(b))
    except Exception:
        return (999999, 999999)


def aa_to_biophys(aa):
    return AA_TO_BIOPHYS.get(str(aa).strip().upper()[:1], "polar")


def aa_to_kd(aa):
    return KD_SCALE.get(str(aa).strip().upper()[:1], np.nan)


def kd_shifted(kd):
    return np.nan if pd.isna(kd) else kd - KD_MIN


def text_color_for_fill(fill_hex):
    return "white" if str(fill_hex).lower() in {c.lower() for c in TEXT_ON_DARK} else "black"


def safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


# ============================================================
# Data loading
# ============================================================

def load_pocket_for_structure(pocket_tsv: str, pdb_id: str) -> pd.DataFrame:
    df = pd.read_csv(pocket_tsv, sep="\t", dtype=str)
    df["pdb_id_norm"] = df["pdb_id"].str.strip().str.upper()
    df = df[df["pdb_id_norm"] == pdb_id.strip().upper()].copy()
    if df.empty:
        raise ValueError(f"Aucune ligne trouvée pour pdb_id={pdb_id}")

    df["gpcrdb_pos"] = df["gpcrdb"].map(simplify_gpcrdb_pos)

    if "is_unmapped_gpcrdb" in df.columns:
        df["is_unmapped_gpcrdb"] = pd.to_numeric(df["is_unmapped_gpcrdb"], errors="coerce").fillna(1).astype(int)
        df = df[df["is_unmapped_gpcrdb"] == 0].copy()

    df = df.dropna(subset=["gpcrdb_pos"]).copy()
    df["aa"] = df["aa"].astype(str).str.strip().str.upper().str[:1]
    df = df[df["aa"].isin(AA_ORDER)].copy()
    df["kd"] = df["aa"].map(aa_to_kd)
    df["kd_shifted"] = df["kd"].map(kd_shifted)
    df["biophys"] = df["aa"].map(aa_to_biophys)

    # Dédoublonner par gpcrdb_pos (garder le premier)
    df = df.drop_duplicates(subset=["gpcrdb_pos"]).copy()
    df = df.sort_values("gpcrdb_pos", key=lambda s: s.map(gpcrdb_sort_key)).copy()

    return df


def detect_class(df: pd.DataFrame) -> str:
    for col in ["gpcr_class", "class_simple"]:
        if col in df.columns:
            val = df[col].dropna().iloc[0] if not df[col].dropna().empty else ""
            if "A" in str(val):
                return "A"
            if "B" in str(val):
                return "B"
    raise ValueError("Impossible de détecter la classe GPCR (A ou B) depuis les données.")


# ============================================================
# Snake plot / Helixbox (SVG coloring)
# ============================================================

SVG_NS = {"svg": "http://www.w3.org/2000/svg"}
ET.register_namespace("", "http://www.w3.org/2000/svg")


def find_by_id(root, node_id):
    if not node_id:
        return None
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


def set_text(elem, txt):
    if elem is None:
        return
    elem.text = str(txt)


SVG_NS_URI = "http://www.w3.org/2000/svg"

def add_gpcrdb_label(root, circle, gpcrdb_pos, fill_color):
    """Add GPCRdb position label to the right of a colored circle with a connector line."""
    if circle is None:
        return
    try:
        cx = float(circle.attrib.get("cx", 0))
        cy = float(circle.attrib.get("cy", 0))
        r  = float(circle.attrib.get("r", 11))
    except ValueError:
        return
    offset_x = r + 4
    label_x  = cx + offset_x + 2
    # Connector line
    line = ET.SubElement(root, f"{{{SVG_NS_URI}}}line")
    line.set("x1", str(cx + r))
    line.set("y1", str(cy))
    line.set("x2", str(label_x - 1))
    line.set("y2", str(cy))
    line.set("stroke", fill_color)
    line.set("stroke-width", "1")
    line.set("style", "display:inline;")
    # Background rectangle for readability
    text_w = len(gpcrdb_pos) * 7
    bg = ET.SubElement(root, f"{{{SVG_NS_URI}}}rect")
    bg.set("x", str(label_x - 1))
    bg.set("y", str(cy - 8))
    bg.set("width", str(text_w + 2))
    bg.set("height", "12")
    bg.set("fill", "black")
    bg.set("opacity", "0.7")
    bg.set("rx", "2")
    # Text label
    text_elem = ET.SubElement(root, f"{{{SVG_NS_URI}}}text")
    text_elem.set("x", str(label_x))
    text_elem.set("y", str(cy + 4))
    text_elem.set("text-anchor", "start")
    text_elem.set("font-size", "11")
    text_elem.set("font-weight", "bold")
    text_elem.set("font-family", "Arial, sans-serif")
    text_elem.set("fill", fill_color)
    text_elem.set("style", f"display:inline; fill:{fill_color}; font-size:11px; font-weight:bold;")
    text_elem.text = gpcrdb_pos


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


def load_mapping_table(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            gp = str(row.get("gpcrdb_pos", "")).strip()
            if not gp:
                continue
            extra = str(row.get("extra_ids", "")).strip()
            extra_ids = [e.strip() for e in extra.replace(";", ",").split(",") if e.strip()]
            rows.append({
                "gpcrdb_pos": gp,
                "circle_id": str(row.get("circle_id", "")).strip(),
                "text_id": str(row.get("text_id", "")).strip(),
                "extra_ids": extra_ids,
            })
    return rows


def expand_svg_canvas_for_legend(root, extra_width=300):
    view_box = root.attrib.get("viewBox", "").strip()
    if view_box:
        parts = view_box.replace(",", " ").split()
        if len(parts) == 4:
            x0, y0, w, h = map(float, parts)
            root.set("viewBox", f"{x0} {y0} {w + extra_width} {h}")
            return x0 + w + 20
    return 20


def add_svg_legend(root, pdb_id: str, class_label: str, n_colored: int, n_total: int, x=20, y=20):
    g = ET.SubElement(root, "g", attrib={"id": "structure_legend"})

    t = ET.SubElement(g, "text", attrib={
        "x": str(x), "y": str(y),
        "font-size": "14", "font-weight": "bold", "fill": "black",
    })
    t.text = f"{pdb_id.upper()} — {class_label} pocket"

    t2 = ET.SubElement(g, "text", attrib={
        "x": str(x), "y": str(y + 18),
        "font-size": "11", "fill": "#555555",
    })
    t2.text = f"{n_colored}/{n_total} pocket residues mapped to template"

    biophys_items = [
        ("hydrophobic (A/V/I/L/M/C)", BIOPHYS_COLORS["hydrophobic"]),
        ("aromatic (F/W/Y)",           BIOPHYS_COLORS["aromatic"]),
        ("positive (K/R/H)",           BIOPHYS_COLORS["positive"]),
        ("negative (D/E)",             BIOPHYS_COLORS["negative"]),
        ("polar (S/T/N/Q)",            BIOPHYS_COLORS["polar"]),
        ("structural (G/P)",           BIOPHYS_COLORS["structural"]),
    ]

    yy = y + 42
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


def color_svg_for_structure(
    svg_path: str,
    mapping_tsv: str,
    pocket_df: pd.DataFrame,
    pdb_id: str,
    class_label: str,
    diagram_type: str,
    out_svg: Path,
):
    pocket_by_pos = pocket_df.set_index("gpcrdb_pos").to_dict(orient="index")
    mapping_rows = load_mapping_table(mapping_tsv)

    tree = ET.parse(svg_path)
    root = tree.getroot()

    # Reset ALL residue circles to white before applying biophysical colors
    for row in mapping_rows:
        circle = find_by_id(root, row["circle_id"])
        if circle is not None:
            set_fill(circle, "#ffffff")
            set_stroke(circle, "#cccccc", width=0.5)

    n_colored = 0
    n_total = len(pocket_df)

    for row in mapping_rows:
        gp = row["gpcrdb_pos"]
        if gp not in pocket_by_pos:
            continue

        info = pocket_by_pos[gp]
        aa = info["aa"]
        biophys = info["biophys"]
        kd = info["kd"]
        seg = str(info.get("gpcrdb_segment", info.get("gpcrdb_segment_gpcrdb", ""))).strip()

        fill_color = BIOPHYS_COLORS.get(biophys, BIOPHYS_COLORS["NA"])
        st = LEVEL_STYLES["strong"]  # freq=1.0 → toujours strong

        circle = find_by_id(root, row["circle_id"])
        text   = find_by_id(root, row["text_id"])

        if circle is not None:
            set_fill(circle, fill_color)
            set_stroke(circle, st["stroke"], width=st["stroke_width"])
            circle.set("stroke-dasharray", "none")
            set_style_attr(circle, "stroke-dasharray", "none")
            set_style_attr(circle, "display", "inline")
            add_title_child(circle, f"{pdb_id.upper()} | {gp} | {aa} | {biophys} | KD={kd:.1f} | {seg}")
            add_gpcrdb_label(root, circle, gp, fill_color)
            n_colored += 1

        if text is not None:
            set_style_attr(text, "display", "inline")
            set_text(text, aa)
            set_text_style(text, fill=text_color_for_fill(fill_color), font_weight="bold")

        for xid in row["extra_ids"]:
            extra = find_by_id(root, xid)
            if extra is not None:
                set_fill(extra, fill_color)

    legend_x = expand_svg_canvas_for_legend(root, extra_width=300)
    add_svg_legend(root, pdb_id, class_label, n_colored, n_total, x=legend_x, y=30)

    out_svg.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(out_svg), encoding="unicode", xml_declaration=False)
    print(f"  [SVG {diagram_type}] {n_colored}/{n_total} positions colorées → {out_svg}")


# ============================================================
# Radar KD
# ============================================================

def closed_polygon(values):
    vals = np.asarray(values, dtype=float)
    return np.concatenate([vals, [vals[0]]])


def closed_angles(n):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.concatenate([angles, [angles[0]]])


def _cluster_indices(angles, close_thr_deg=14.0):
    if len(angles) == 0:
        return []
    thr = np.deg2rad(close_thr_deg)
    order = np.argsort(angles)
    clusters = [[order[0]]]
    for k in range(1, len(order)):
        prev_idx, cur_idx = order[k - 1], order[k]
        if abs(angles[cur_idx] - angles[prev_idx]) < thr:
            clusters[-1].append(cur_idx)
        else:
            clusters.append([cur_idx])
    if len(clusters) > 1:
        circ_dist = min(
            abs(angles[clusters[0][0]] - angles[clusters[-1][-1]]),
            2 * np.pi - abs(angles[clusters[0][0]] - angles[clusters[-1][-1]]),
        )
        if circ_dist < thr:
            clusters[0] = clusters[-1] + clusters[0]
            clusters.pop(-1)
    return clusters


def _stagger_radii(n, base_r, step=0.42):
    return [base_r + k * step for k in range(n)]


def add_position_labels(ax, positions, angles, seg_map, pos_r_base=10.55, seg_r_base=9.92):
    clusters = _cluster_indices(angles)
    pos_r_map = {}
    seg_r_map = {}
    for cluster in clusters:
        radii = _stagger_radii(len(cluster), 0.0, 0.42)
        for idx, dr in zip(cluster, radii):
            pos_r_map[idx] = pos_r_base + dr
            seg_r_map[idx] = seg_r_base + dr

    for i, pos in enumerate(positions):
        ang = angles[i]
        x_side = np.cos(ang)
        if x_side > 0.30:
            ha, xoff = "left", 6
        elif x_side < -0.30:
            ha, xoff = "right", -6
        else:
            ha, xoff = "center", 0

        ax.annotate(pos, xy=(ang, pos_r_map.get(i, pos_r_base)), xytext=(xoff, 0),
                    textcoords="offset points", ha=ha, va="center",
                    fontsize=9, color="black", annotation_clip=False)
        seg = seg_map.get(pos, "")
        if seg:
            ax.annotate(seg, xy=(ang, seg_r_map.get(i, seg_r_base)), xytext=(xoff, 0),
                        textcoords="offset points", ha=ha, va="center",
                        fontsize=7.5, color="dimgray", annotation_clip=False)


def add_aa_letters(ax, positions, angles, pocket_df, outer_r_base=11.20):
    pocket_by_pos = pocket_df.set_index("gpcrdb_pos")
    clusters = _cluster_indices(angles)
    outer_r_map = {}
    for cluster in clusters:
        radii = _stagger_radii(len(cluster), outer_r_base, 0.58)
        for idx, r in zip(cluster, radii):
            outer_r_map[idx] = r

    for i, pos in enumerate(positions):
        ang = angles[i]
        if pos not in pocket_by_pos.index:
            continue
        aa = pocket_by_pos.loc[pos, "aa"]
        biophys = pocket_by_pos.loc[pos, "biophys"]
        color = BIOPHYS_COLORS.get(biophys, "#999999")
        x_side = np.cos(ang)
        if x_side > 0.30:
            ha, base_x = "left", 10
        elif x_side < -0.30:
            ha, base_x = "right", -10
        else:
            ha, base_x = "center", 0
        r = outer_r_map.get(i, outer_r_base)
        ax.annotate(aa, xy=(ang, r), xytext=(base_x, 0),
                    textcoords="offset points", ha=ha, va="center",
                    fontsize=18, color=color, fontweight="bold", annotation_clip=False)


def plot_radar_kd(pocket_df: pd.DataFrame, pdb_id: str, out_png: Path, out_pdf: Path = None):
    positions = pocket_df["gpcrdb_pos"].tolist()
    kd_vals = pocket_df["kd_shifted"].tolist()

    seg_col = None
    for c in ["gpcrdb_segment", "segment_gpcrdb"]:
        if c in pocket_df.columns:
            seg_col = c
            break
    seg_map = {}
    if seg_col:
        seg_map = pocket_df.set_index("gpcrdb_pos")[seg_col].dropna().to_dict()

    n = len(positions)
    angles = closed_angles(n)
    values_closed = closed_polygon(kd_vals)

    fig_w = min(22, max(12.5, 12.5 + 0.4 * max(0, n - 10)))
    fig_h = min(18, max(9.5, 9.5 + 0.3 * max(0, n - 10)))
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = plt.subplot(111, polar=True)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])
    ax.set_ylim(0, 11.0)
    yticks = [0, 2, 4, 6, 8, 9]
    ax.set_yticks(yticks)
    ax.set_yticklabels([])
    ax.grid(True, alpha=0.5)
    ylabels = ["-4.5", "-2.5", "-0.5", "1.5", "3.5", "4.5"]
    theta_lab = np.deg2rad(235)
    for r, lab in zip(yticks, ylabels):
        ax.text(theta_lab, r, lab, fontsize=8, color="black", ha="center", va="center")

    ax.plot(angles, values_closed, linewidth=2.5, color="#1f4ed8")
    ax.fill(angles, np.nan_to_num(values_closed, nan=0.0), alpha=0.18, color="#1f4ed8")

    fig.suptitle(f"{pdb_id.upper()} — pocket KD radar", fontsize=16, y=0.975)

    add_position_labels(ax, positions, angles[:-1], seg_map)
    add_aa_letters(ax, positions, angles[:-1], pocket_df)

    legend_handles = [
        Patch(facecolor=BIOPHYS_COLORS["hydrophobic"], edgecolor="black", label="hydrophobic (A/V/I/L/M/C)"),
        Patch(facecolor=BIOPHYS_COLORS["aromatic"],    edgecolor="black", label="aromatic (F/W/Y)"),
        Patch(facecolor=BIOPHYS_COLORS["positive"],    edgecolor="black", label="positive (K/R/H)"),
        Patch(facecolor=BIOPHYS_COLORS["negative"],    edgecolor="black", label="negative (D/E)"),
        Patch(facecolor=BIOPHYS_COLORS["polar"],       edgecolor="black", label="polar (S/T/N/Q)"),
        Patch(facecolor=BIOPHYS_COLORS["structural"],  edgecolor="black", label="structural (G/P)"),
    ]
    ax.legend(handles=legend_handles, title="Residue classes",
              loc="upper right", bbox_to_anchor=(1.26, 1.05),
              fontsize=9, title_fontsize=10, frameon=True)

    fig.subplots_adjust(left=0.08, right=0.88, top=0.82, bottom=0.10)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=250)
    if out_pdf:
        fig.savefig(out_pdf)
    plt.close(fig)
    print(f"  [Radar KD] {n} positions → {out_png}")


# ============================================================
# WebLogo
# ============================================================

def plot_weblogo(pocket_df: pd.DataFrame, pdb_id: str, out_png: Path, out_pdf: Path = None):
    positions = pocket_df["gpcrdb_pos"].tolist()

    seg_col = None
    for c in ["gpcrdb_segment", "segment_gpcrdb"]:
        if c in pocket_df.columns:
            seg_col = c
            break
    seg_map = {}
    if seg_col:
        seg_map = pocket_df.set_index("gpcrdb_pos")[seg_col].dropna().to_dict()

    # Matrice de comptage : 1 séquence = 1 pour chaque position
    rows = []
    for pos in positions:
        aa_counts = {aa: 0 for aa in AA_ORDER}
        sub = pocket_df[pocket_df["gpcrdb_pos"] == pos]["aa"]
        for aa in sub:
            aa = str(aa).strip().upper()[:1]
            if aa in aa_counts:
                aa_counts[aa] += 1
        rows.append(aa_counts)

    count_df = pd.DataFrame(rows, index=range(len(positions)))

    color_scheme = {aa: BIOPHYS_COLORS.get(AA_TO_BIOPHYS.get(aa, "polar"), "#999999") for aa in AA_ORDER}

    fig_w = max(12, min(32, 0.72 * len(positions) + 7))
    fig_h = 6.4
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(1, 2, width_ratios=[14, 3.8],
                          left=0.06, right=0.97, top=0.90, bottom=0.20, wspace=0.08)
    ax = fig.add_subplot(gs[0, 0])
    ax_leg = fig.add_subplot(gs[0, 1])

    logomaker.Logo(count_df, ax=ax, color_scheme=color_scheme,
                   shade_below=0.5, fade_below=0.5, stack_order="big_on_top")

    ax.set_title(f"Pocket WebLogo — {pdb_id.upper()}", fontsize=16)
    ax.set_ylabel("Counts", fontsize=12)
    ax.set_xlabel("GPCRdb pocket positions", fontsize=12, labelpad=30)
    ax.set_xticks(np.arange(len(positions)))
    ax.set_xticklabels(positions, rotation=60, ha="right", fontsize=9)

    segment_labels = [seg_map.get(pos, "") for pos in positions]
    y_text = -0.14
    for i, seg in enumerate(segment_labels):
        ax.text(i, y_text, seg, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=8, color="dimgray", clip_on=False)
    ax.text(1.01, y_text, "segments", transform=ax.transAxes,
            ha="left", va="top", fontsize=8, color="dimgray")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax_leg.axis("off")
    legend_handles = [
        Patch(facecolor=BIOPHYS_COLORS["hydrophobic"], edgecolor="black", label="hydrophobic (A/V/I/L/M/C)"),
        Patch(facecolor=BIOPHYS_COLORS["aromatic"],    edgecolor="black", label="aromatic (F/W/Y)"),
        Patch(facecolor=BIOPHYS_COLORS["positive"],    edgecolor="black", label="positive (K/R/H)"),
        Patch(facecolor=BIOPHYS_COLORS["negative"],    edgecolor="black", label="negative (D/E)"),
        Patch(facecolor=BIOPHYS_COLORS["polar"],       edgecolor="black", label="polar (S/T/N/Q)"),
        Patch(facecolor=BIOPHYS_COLORS["structural"],  edgecolor="black", label="structural (G/P)"),
    ]
    ax_leg.legend(handles=legend_handles, title="Residue classes",
                  loc="upper left", fontsize=10, title_fontsize=11,
                  frameon=True, borderpad=0.7, labelspacing=0.6)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=250, bbox_inches="tight")
    if out_pdf:
        fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  [WebLogo] {len(positions)} positions → {out_png}")


# ============================================================
# Main
# ============================================================

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pocket_tsv", required=True,
                    help="pocket_biophys_by_residue.with_gpcrdb_segments.with_target_resnum.tsv")
    ap.add_argument("--pdb_id", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--templates_dir", default="templates")
    return ap.parse_args()


def main():
    args = parse_args()
    pdb_id = args.pdb_id.strip().lower()
    outdir = Path(args.outdir) / pdb_id.upper()
    outdir.mkdir(parents=True, exist_ok=True)
    tpl = Path(args.templates_dir)

    print(f"\n=== {pdb_id.upper()} ===")
    pocket_df = load_pocket_for_structure(args.pocket_tsv, pdb_id)
    print(f"  {len(pocket_df)} résidus de poche avec mapping GPCRdb")

    gpcr_class = detect_class(pocket_df)
    class_label = f"Class {'A' if gpcr_class == 'A' else 'B'}"
    print(f"  Classe détectée : {class_label}")

    if gpcr_class == "A":
        snake_svg   = tpl / "classA_snakeplot.svg"
        snake_map   = tpl / "classA_snakeplot_mapping.tsv"
        helix_svg   = tpl / "classA_helixbox.svg"
        helix_map   = tpl / "classA_helixbox_mapping.tsv"
    else:
        snake_svg   = tpl / "classB_snakeplot.svg"
        snake_map   = tpl / "classB_snakeplot_mapping.tsv"
        helix_svg   = tpl / "classB_helixbox.svg"
        helix_map   = tpl / "classB_helixbox_mapping.tsv"

    # 1. Snake plot
    color_svg_for_structure(
        svg_path=str(snake_svg),
        mapping_tsv=str(snake_map),
        pocket_df=pocket_df,
        pdb_id=pdb_id,
        class_label=class_label,
        diagram_type="snakeplot",
        out_svg=outdir / f"{pdb_id.upper()}_snakeplot.svg",
    )

    # 2. Helixbox
    color_svg_for_structure(
        svg_path=str(helix_svg),
        mapping_tsv=str(helix_map),
        pocket_df=pocket_df,
        pdb_id=pdb_id,
        class_label=class_label,
        diagram_type="helixbox",
        out_svg=outdir / f"{pdb_id.upper()}_helixbox.svg",
    )

    # 3. Radar KD
    plot_radar_kd(
        pocket_df=pocket_df,
        pdb_id=pdb_id,
        out_png=outdir / f"{pdb_id.upper()}_kd_radar.png",
        out_pdf=outdir / f"{pdb_id.upper()}_kd_radar.pdf",
    )

    # 4. WebLogo
    plot_weblogo(
        pocket_df=pocket_df,
        pdb_id=pdb_id,
        out_png=outdir / f"{pdb_id.upper()}_pocket_weblogo.png",
        out_pdf=outdir / f"{pdb_id.upper()}_pocket_weblogo.pdf",
    )

    print(f"\n[DONE] Figures pour {pdb_id.upper()} → {outdir}\n")


if __name__ == "__main__":
    main()
