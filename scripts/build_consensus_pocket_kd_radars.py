#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_consensus_pocket_kd_radars.py  (v2 — anti-overlap, final)

Changements finaux :
- add_external_letters : staggering radial pour les positions angulairement proches
- add_axis_position_and_segment_labels : staggering radial pour positions + segments
- _cluster_indices : gestion correcte du wrap-around circulaire (0 / 2π)
- savefig : suppression de bbox_inches="tight" pour ne pas casser la géométrie
- radial_step légèrement réduit pour un rendu plus compact
"""

import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# ============================================================
# Constants
# ============================================================

AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")

KD_SCALE = {
    "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5, "M": 1.9, "A": 1.8,
    "G": -0.4, "T": -0.7, "S": -0.8, "W": -0.9, "Y": -1.3, "P": -1.6,
    "H": -3.2, "E": -3.5, "Q": -3.5, "D": -3.5, "N": -3.5, "K": -3.9, "R": -4.5,
}

KD_MIN = -4.5
KD_MAX = 4.5

AA_TO_BIOPHYS = {
    # Schéma personnalisé 6 classes (Lehninger 7e éd. comme référence générale).
    # C → hydrophobic (KD = +2.5). G/P → structural (contraintes conformationnelles).
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
    "structural":     "#6dbe6d",
}

CLASS_TO_FILETAG = {
    "Class A": "Class_A",
    "Class B": "Class_B",
}


# ============================================================
# Helpers
# ============================================================

def norm_pdb(x):
    return str(x).strip().upper()


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


def simplify_class_label(s):
    s = str(s)
    if "Class A" in s:
        return "Class A"
    if "Class B" in s:
        return "Class B"
    return "Other"


def aa_to_biophys(aa):
    return AA_TO_BIOPHYS.get(str(aa).strip().upper()[:1], "polar")


def aa_to_color(aa):
    return BIOPHYS_COLORS[aa_to_biophys(aa)]


def aa_to_kd(aa):
    return KD_SCALE.get(str(aa).strip().upper()[:1], np.nan)


def kd_to_shifted(kd):
    return np.nan if pd.isna(kd) else kd - KD_MIN


# ============================================================
# Loaders
# ============================================================

def load_pocket_table(path):
    df = pd.read_csv(path, sep="\t", dtype=str)
    required = ["pdb_id", "gpcr_class", "aa"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[pocket_tsv] colonnes manquantes: {missing}")

    df["pdb_id"] = df["pdb_id"].map(norm_pdb)
    df["class_simple"] = df["gpcr_class"].map(simplify_class_label)

    if "gpcrdb" in df.columns:
        df["gpcrdb_pos"] = df["gpcrdb"].map(simplify_gpcrdb_pos)
    elif "gpcrdb_display_generic_number" in df.columns:
        df["gpcrdb_pos"] = df["gpcrdb_display_generic_number"].map(simplify_gpcrdb_pos)
    else:
        raise ValueError("[pocket_tsv] colonne gpcrdb manquante")

    df["aa"] = df["aa"].astype(str).str.strip().str.upper().str[:1]
    df = df[df["aa"].isin(AA_ORDER)].copy()

    if "is_unmapped_gpcrdb" in df.columns:
        df["is_unmapped_gpcrdb"] = (
            pd.to_numeric(df["is_unmapped_gpcrdb"], errors="coerce")
            .fillna(1).astype(int)
        )
        df = df[df["is_unmapped_gpcrdb"] == 0].copy()

    df = df.dropna(subset=["pdb_id", "class_simple", "gpcrdb_pos", "aa"]).copy()
    df["kd"] = df["aa"].map(aa_to_kd)
    df["kd_shifted"] = df["kd"].map(kd_to_shifted)
    return df


def load_consensus_positions(path):
    df = pd.read_csv(path, sep="\t")
    if "gpcrdb_pos" not in df.columns:
        raise ValueError(f"[consensus] colonne gpcrdb_pos manquante dans {path}")
    df["gpcrdb_pos"] = df["gpcrdb_pos"].map(simplify_gpcrdb_pos)
    df = df.dropna(subset=["gpcrdb_pos"]).drop_duplicates(subset=["gpcrdb_pos"]).copy()
    keep_cols = ["gpcrdb_pos"]
    for c in ["freq_structures", "segment_gpcrdb", "gpcrdb_interaction_types"]:
        if c in df.columns:
            keep_cols.append(c)
    return df[keep_cols].copy()


# ============================================================
# Data builders
# ============================================================

def build_class_data(pocket_df, consensus_df, class_label):
    positions = sorted(
        consensus_df["gpcrdb_pos"].unique().tolist(), key=gpcrdb_sort_key
    )
    sub = pocket_df[pocket_df["class_simple"] == class_label].copy()
    sub = sub[sub["gpcrdb_pos"].isin(positions)].copy()

    if sub.empty:
        return positions, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}

    dedup = (
        sub.groupby(["pdb_id", "gpcrdb_pos"], as_index=False)
        .agg({
            "aa": lambda x: x.mode().iloc[0] if len(x.mode()) else x.iloc[0],
            "kd": lambda x: x.mode().iloc[0] if len(x.mode()) else x.iloc[0],
            "kd_shifted": lambda x: x.mode().iloc[0] if len(x.mode()) else x.iloc[0],
        })
    )

    pivot_aa = dedup.pivot(
        index="pdb_id", columns="gpcrdb_pos", values="aa"
    ).reindex(columns=positions)
    pivot_kd = dedup.pivot(
        index="pdb_id", columns="gpcrdb_pos", values="kd_shifted"
    ).reindex(columns=positions)

    rows = []
    for pos in positions:
        vals = pivot_aa[pos].dropna().astype(str) if pos in pivot_aa.columns else pd.Series(dtype=str)
        n = len(vals)
        vc = vals.value_counts() if n > 0 else pd.Series(dtype=int)
        for aa, count in vc.items():
            freq = count / n if n > 0 else np.nan
            rows.append({
                "gpcrdb_pos": pos, "aa": aa, "count": int(count),
                "freq": float(freq), "kd": aa_to_kd(aa),
                "kd_shifted": kd_to_shifted(aa_to_kd(aa)),
                "biophys_class": aa_to_biophys(aa),
            })

    dist_df = pd.DataFrame(rows)

    mean_rows = []
    meta = consensus_df.set_index("gpcrdb_pos").to_dict(orient="index")
    for pos in positions:
        sub_pos = dist_df[dist_df["gpcrdb_pos"] == pos].copy()
        kd_mean = float((sub_pos["freq"] * sub_pos["kd"]).sum()) if not sub_pos.empty else np.nan
        row = {"gpcrdb_pos": pos, "kd_mean": kd_mean,
               "kd_mean_shifted": kd_to_shifted(kd_mean)}
        if pos in meta:
            for k, v in meta[pos].items():
                if k != "gpcrdb_pos":
                    row[k] = v
        mean_rows.append(row)

    mean_df = pd.DataFrame(mean_rows)

    seg_map = {}
    if "segment_gpcrdb" in consensus_df.columns:
        seg_map = (
            consensus_df[["gpcrdb_pos", "segment_gpcrdb"]]
            .dropna().drop_duplicates(subset=["gpcrdb_pos"])
            .set_index("gpcrdb_pos")["segment_gpcrdb"].to_dict()
        )

    return positions, pivot_aa, pivot_kd, dist_df, mean_df, seg_map


def export_position_frequency_table(dist_df, mean_df, out_tsv):
    if dist_df.empty:
        pd.DataFrame().to_csv(out_tsv, sep="\t", index=False)
        return
    merged = dist_df.merge(
        mean_df[["gpcrdb_pos", "kd_mean", "kd_mean_shifted"]],
        on="gpcrdb_pos", how="left"
    )
    merged = merged.sort_values(
        ["gpcrdb_pos", "freq", "aa"], ascending=[True, False, True]
    )
    merged.to_csv(out_tsv, sep="\t", index=False)


# ============================================================
# Radar helpers
# ============================================================

def closed_polygon(values):
    vals = np.asarray(values, dtype=float)
    return np.concatenate([vals, [vals[0]]])


def closed_angles(n_axes):
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False)
    return np.concatenate([angles, [angles[0]]])


def font_size_from_freq(freq, fmin=0.0, fmax=1.0, smin=8, smax=24):
    if pd.isna(freq):
        return smin
    freq = max(fmin, min(fmax, float(freq)))
    if fmax == fmin:
        return smin
    return smin + (smax - smin) * ((freq - fmin) / (fmax - fmin))


def _cluster_indices(angles, close_thr_deg=14.0):
    """
    Regroupe les indices dont les angles sont proches, en tenant compte
    aussi du wrap-around circulaire (dernier angle proche du premier).
    """
    if len(angles) == 0:
        return []

    thr = np.deg2rad(close_thr_deg)
    order = np.argsort(angles)

    clusters = [[order[0]]]
    for k in range(1, len(order)):
        prev_idx = order[k - 1]
        cur_idx = order[k]
        if abs(angles[cur_idx] - angles[prev_idx]) < thr:
            clusters[-1].append(cur_idx)
        else:
            clusters.append([cur_idx])

    if len(clusters) > 1:
        first_idx = clusters[0][0]
        last_idx = clusters[-1][-1]

        ang_first = angles[first_idx]
        ang_last = angles[last_idx]

        circ_dist = min(abs(ang_first - ang_last), 2 * np.pi - abs(ang_first - ang_last))
        if circ_dist < thr:
            clusters[0] = clusters[-1] + clusters[0]
            clusters.pop(-1)

    return clusters


def _stagger_radii(n, base_r, step=0.55):
    return [base_r + k * step for k in range(n)]


def add_axis_position_and_segment_labels(
    ax, positions, angles, seg_map,
    pos_r_base=10.55,
    seg_r_base=9.92,
    close_thr_deg=14.0,
    radial_step=0.42,
):
    """
    Draw position labels and segment labels with radial staggering for
    angularly close positions.
    """
    clusters = _cluster_indices(angles, close_thr_deg)

    pos_r_map = {}
    seg_r_map = {}
    for cluster in clusters:
        radii = _stagger_radii(len(cluster), 0.0, radial_step)
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

        pos_r = pos_r_map.get(i, pos_r_base)
        seg_r = seg_r_map.get(i, seg_r_base)

        ax.annotate(
            pos, xy=(ang, pos_r), xytext=(xoff, 0),
            textcoords="offset points",
            ha=ha, va="center", fontsize=10, color="black",
            fontweight="normal", annotation_clip=False,
        )

        seg = seg_map.get(pos, "")
        if seg:
            ax.annotate(
                seg, xy=(ang, seg_r), xytext=(xoff, 0),
                textcoords="offset points",
                ha=ha, va="center", fontsize=8, color="dimgray",
                annotation_clip=False,
            )


def add_external_letters(
    ax, positions, angles, dist_df, seg_map,
    outer_r_base=11.20,
    line_step=0.50,
    close_thr_deg=14.0,
    radial_step=0.58,
    max_letters_per_position=None,
    min_freq=0.0,
):
    """
    Draw AA letters with radial staggering for angularly close positions.
    """
    if len(angles) == 0:
        return

    clusters = _cluster_indices(angles, close_thr_deg)

    outer_r_map = {}
    for cluster in clusters:
        radii = _stagger_radii(len(cluster), outer_r_base, radial_step)
        for idx, r in zip(cluster, radii):
            outer_r_map[idx] = r

    for i, pos in enumerate(positions):
        ang = angles[i]
        sub = dist_df[dist_df["gpcrdb_pos"] == pos].copy()
        if sub.empty:
            continue

        sub = sub[sub["freq"] >= min_freq].sort_values(
            ["freq", "aa"], ascending=[False, True]
        )
        if max_letters_per_position is not None:
            sub = sub.head(max_letters_per_position)

        x_side = np.cos(ang)
        if x_side > 0.30:
            ha, base_x = "left", 10
        elif x_side < -0.30:
            ha, base_x = "right", -10
        else:
            ha, base_x = "center", 0

        current_r = outer_r_map.get(i, outer_r_base)

        for _, r in sub.iterrows():
            aa = r["aa"]
            freq = r["freq"]
            fs = font_size_from_freq(freq, smin=8, smax=24)

            ax.annotate(
                aa, xy=(ang, current_r), xytext=(base_x, 0),
                textcoords="offset points",
                ha=ha, va="center",
                fontsize=fs, color=aa_to_color(aa),
                fontweight="bold", annotation_clip=False,
            )
            current_r += line_step


def add_biophys_legend(ax, title="Residue classes"):
    handles = [
        Patch(facecolor=BIOPHYS_COLORS["hydrophobic"], edgecolor="black", label="hydrophobic (A/V/I/L/M/C)"),
        Patch(facecolor=BIOPHYS_COLORS["aromatic"],    edgecolor="black", label="aromatic (F/W/Y)"),
        Patch(facecolor=BIOPHYS_COLORS["positive"],    edgecolor="black", label="positive (K/R/H)"),
        Patch(facecolor=BIOPHYS_COLORS["negative"],    edgecolor="black", label="negative (D/E)"),
        Patch(facecolor=BIOPHYS_COLORS["polar"],       edgecolor="black", label="polar (S/T/N/Q)"),
        Patch(facecolor=BIOPHYS_COLORS["structural"],     edgecolor="black", label="structural (G/P)"),
    ]
    ax.legend(
        handles=handles, title=title,
        loc="upper right", bbox_to_anchor=(1.26, 1.05),
        fontsize=9, title_fontsize=10, frameon=True,
    )


def style_radar_ax(ax, positions, angles):
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])
    ax.set_ylim(0, 11.0)

    yticks = [0, 2, 4, 6, 8, 9]  # 9 = KD_MAX (4.5) shifted by -KD_MIN (4.5)
    ax.set_yticks(yticks)
    ax.set_yticklabels([])
    ax.grid(True, alpha=0.5)

    ylabels = ["-4.5", "-2.5", "-0.5", "1.5", "3.5", "4.5"]
    theta_lab = np.deg2rad(235)
    for r, lab in zip(yticks, ylabels):
        ax.text(theta_lab, r, lab, fontsize=8, color="black", ha="center", va="center")


def add_figure_title(fig, title, per_structure=False):
    fig.suptitle(title, fontsize=16, y=0.975 if per_structure else 0.965)


def compute_figure_size(n_positions, per_structure=False):
    base_w = 12.5 if not per_structure else 13.5
    base_h = 9.5  if not per_structure else 10.0
    extra = max(0, n_positions - 10)
    return min(base_w + 0.45 * extra, 22), min(base_h + 0.35 * extra, 18)


def finalize_radar_layout(fig, legend_mode="biophys"):
    if legend_mode == "per_structure":
        fig.subplots_adjust(left=0.08, right=0.78, top=0.82, bottom=0.10)
    else:
        fig.subplots_adjust(left=0.08, right=0.88, top=0.82, bottom=0.10)


# ============================================================
# Plots
# ============================================================

def plot_radar_mean(
    class_label, positions, mean_df, dist_df, seg_map,
    out_png, out_pdf=None,
    max_letters_per_position=None, min_freq=0.0,
):
    if mean_df.empty:
        print(f"[WARN] radar mean non généré pour {class_label}: aucune donnée.")
        return

    values = mean_df.set_index("gpcrdb_pos").reindex(positions)["kd_mean_shifted"].to_numpy(dtype=float)
    angles = closed_angles(len(positions))
    values_closed = closed_polygon(values)

    fig_w, fig_h = compute_figure_size(len(positions), per_structure=False)
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = plt.subplot(111, polar=True)

    style_radar_ax(ax=ax, positions=positions, angles=angles)
    ax.plot(angles, values_closed, linewidth=2.2)
    ax.fill(angles, values_closed, alpha=0.18)

    add_figure_title(fig, f"{class_label} — consensus pocket KD radar (mean profile)")

    add_axis_position_and_segment_labels(
        ax=ax, positions=positions, angles=angles[:-1], seg_map=seg_map,
    )
    add_external_letters(
        ax=ax, positions=positions, angles=angles[:-1],
        dist_df=dist_df, seg_map=seg_map,
        max_letters_per_position=max_letters_per_position, min_freq=min_freq,
    )
    add_biophys_legend(ax, title="Residue classes")
    finalize_radar_layout(fig, legend_mode="biophys")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=250)
    if out_pdf is not None:
        fig.savefig(out_pdf)
    plt.close(fig)


def plot_radar_per_structure(
    class_label, positions, pivot_kd, dist_df, seg_map,
    out_png, out_pdf=None,
    max_structures=None, min_positions_per_structure=3,
    alpha=0.35, linewidth=1.1,
    max_letters_per_position=None, min_freq=0.0,
):
    if pivot_kd.empty:
        print(f"[WARN] radar per-structure non généré pour {class_label}: aucune donnée.")
        return

    plot_df = pivot_kd.copy()
    n_non_na = plot_df.notna().sum(axis=1)
    plot_df = plot_df[n_non_na >= min_positions_per_structure].copy()

    if plot_df.empty:
        print(f"[WARN] radar per-structure non généré pour {class_label}: aucune structure assez complète.")
        return

    if max_structures is not None and len(plot_df) > max_structures:
        plot_df = plot_df.iloc[:max_structures].copy()

    angles = closed_angles(len(positions))
    fig_w, fig_h = compute_figure_size(len(positions), per_structure=True)
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = plt.subplot(111, polar=True)

    style_radar_ax(ax=ax, positions=positions, angles=angles)

    cmap = plt.cm.get_cmap("tab20", len(plot_df))
    for idx, (pdb_id, row) in enumerate(plot_df.iterrows()):
        vals = row.reindex(positions).to_numpy(dtype=float)
        vals_closed = closed_polygon(vals)
        ax.plot(angles, vals_closed, linewidth=linewidth,
                alpha=alpha + 0.15, color=cmap(idx), label=pdb_id)
        ax.fill(angles, np.nan_to_num(vals_closed, nan=0.0),
                alpha=0.05, color=cmap(idx))

    add_figure_title(
        fig,
        f"{class_label} — consensus pocket KD radar (1 polygon per structure)",
        per_structure=True,
    )

    add_axis_position_and_segment_labels(
        ax=ax, positions=positions, angles=angles[:-1], seg_map=seg_map,
    )
    add_external_letters(
        ax=ax, positions=positions, angles=angles[:-1],
        dist_df=dist_df, seg_map=seg_map,
        max_letters_per_position=max_letters_per_position, min_freq=min_freq,
    )
    add_biophys_legend(ax, title="Residue classes")

    ax.legend(
        loc="center left", bbox_to_anchor=(1.28, 0.42),
        fontsize=7, frameon=True, ncol=1,
        title="Structures", title_fontsize=8,
    )
    finalize_radar_layout(fig, legend_mode="per_structure")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=250)
    if out_pdf is not None:
        fig.savefig(out_pdf)
    plt.close(fig)


# ============================================================
# Runner
# ============================================================

def run_for_one_class(
    pocket_df, consensus_path, class_label, outdir, mode,
    max_structures, min_positions_per_structure,
    max_letters_per_position, min_freq_for_letters,
):
    class_tag = CLASS_TO_FILETAG.get(class_label, class_label.replace(" ", "_"))
    consensus_df = load_consensus_positions(consensus_path)
    positions, pivot_aa, pivot_kd, dist_df, mean_df, seg_map = build_class_data(
        pocket_df=pocket_df, consensus_df=consensus_df, class_label=class_label,
    )

    class_out = outdir / class_tag
    class_out.mkdir(parents=True, exist_ok=True)

    if pivot_aa.empty:
        print(f"[WARN] aucune donnée exploitable pour {class_label}")
        return

    pivot_aa.to_csv(class_out / f"{class_tag}.alignment_aa.tsv", sep="\t", index=True, index_label="pdb_id")
    pivot_kd.to_csv(class_out / f"{class_tag}.alignment_kd_shifted.tsv", sep="\t", index=True, index_label="pdb_id")
    export_position_frequency_table(dist_df, mean_df, class_out / f"{class_tag}.position_aa_frequencies.tsv")

    if mode in ("mean", "both"):
        plot_radar_mean(
            class_label=class_label, positions=positions,
            mean_df=mean_df, dist_df=dist_df, seg_map=seg_map,
            out_png=class_out / f"{class_tag}.consensus_kd_radar_mean.png",
            out_pdf=class_out / f"{class_tag}.consensus_kd_radar_mean.pdf",
            max_letters_per_position=max_letters_per_position,
            min_freq=min_freq_for_letters,
        )

    if mode in ("per_structure", "both"):
        plot_radar_per_structure(
            class_label=class_label, positions=positions,
            pivot_kd=pivot_kd, dist_df=dist_df, seg_map=seg_map,
            out_png=class_out / f"{class_tag}.consensus_kd_radar_per_structure.png",
            out_pdf=class_out / f"{class_tag}.consensus_kd_radar_per_structure.pdf",
            max_structures=max_structures,
            min_positions_per_structure=min_positions_per_structure,
            max_letters_per_position=max_letters_per_position,
            min_freq=min_freq_for_letters,
        )

    print(f"[DONE] {class_label} -> {class_out}")


# ============================================================
# CLI
# ============================================================

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pocket_tsv", required=True)
    ap.add_argument("--consensus_a", required=True)
    ap.add_argument("--consensus_b", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--mode", default="both", choices=["mean", "per_structure", "both"])
    ap.add_argument("--max_structures", type=int, default=20)
    ap.add_argument("--min_positions_per_structure", type=int, default=3)
    ap.add_argument("--max_letters_per_position", type=int, default=None)
    ap.add_argument("--min_freq_for_letters", type=float, default=0.0)
    return ap.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pocket_df = load_pocket_table(args.pocket_tsv)

    run_for_one_class(
        pocket_df=pocket_df, consensus_path=args.consensus_a,
        class_label="Class A", outdir=outdir, mode=args.mode,
        max_structures=args.max_structures,
        min_positions_per_structure=args.min_positions_per_structure,
        max_letters_per_position=args.max_letters_per_position,
        min_freq_for_letters=args.min_freq_for_letters,
    )

    run_for_one_class(
        pocket_df=pocket_df, consensus_path=args.consensus_b,
        class_label="Class B", outdir=outdir, mode=args.mode,
        max_structures=args.max_structures,
        min_positions_per_structure=args.min_positions_per_structure,
        max_letters_per_position=args.max_letters_per_position,
        min_freq_for_letters=args.min_freq_for_letters,
    )

    print(f"[DONE] All outputs written to: {outdir}")


if __name__ == "__main__":
    main()
