#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_peptide_contact_maps_with_consensus.py

Ajouts (NEW):
- Bande "receptor residue AA" (1 lettre) pour chaque position GPCRdb affichée
- Bande "receptor biophys class" (hydrophobic/polar/+/−/aromatic/special/other)
  récupérées depuis target_annot_tsv = pocket_biophys_by_residue.with_gpcrdb_segments.tsv

Fix (NEW):
- Légende simplifiée: UNE seule légende (biophys classes, commune peptide + receptor)
- Légende déplacée à droite (hors axe) pour éviter chevauchement
- Titre remonté + layout ajusté (top/right) pour éviter chevauchement avec bandes/legend

Segments en haut:
- toujours affichés en blocs NOIRS/BLANCS (pas de couleurs).

Usage:
python3 scripts/plot_peptide_contact_maps_with_consensus.py \
  --pairs_tsv run_out/biophys_annotations/peptide_contacts.contacts_pairs.tsv \
  --seqs_tsv run_out/biophys_annotations/peptide_contacts.peptide_sequences.tsv \
  --target_annot_tsv run_out/biophys_annotations/pocket_biophys_by_residue.with_gpcrdb_segments.tsv \
  --consensus_dir out/consensus_validable_strict \
  --outdir out/peptide_biophys/maps \
  --mode both \
  --classes "Class A,Class B" \
  --threshold 0.50 \
  --cutoff 5.0 \
  --consensus_x_mode intersection
"""

import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


# -----------------------
# Normalisation helpers
# -----------------------

def norm_pdb(x: str) -> str:
    return str(x).strip().upper()


def simplify_gpcrdb_pos(pos: str) -> str:
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


def gpcrdb_sort_key(pos: str):
    try:
        a, b = pos.split("x")
        return (int(a), int(b))
    except Exception:
        return (999999, 999999)


# -----------------------
# Biophys classes (peptide + receptor)
# -----------------------

def biophys_class_from_aa(aa: str) -> str:
    # Nelson & Cox, Lehninger Principles of Biochemistry, 8th ed. (2021)
    aa = (aa or "X").upper()
    if aa in set("GAPVLIM"):
        return "nonpolar_aliphatic"
    if aa in set("FWY"):
        return "aromatic"
    if aa in set("STCNQ"):
        return "polar"
    if aa in set("KRH"):
        return "positive"
    if aa in set("DE"):
        return "negative"
    return "other"


CLASS_TO_CODE = {
    "other": 0,
    "nonpolar_aliphatic": 1,
    "polar": 2,
    "positive": 3,
    "negative": 4,
    "aromatic": 5,
}

BIOPHYS_CMAP = ListedColormap([
    "#eaeaea",  # other
    "#88c34a",  # nonpolar_aliphatic
    "#7e57c2",  # polar
    "#1e88e5",  # positive
    "#e53935",  # negative
    "#ffb74d",  # aromatic
])


# -----------------------
# Loaders (robustes)
# -----------------------

def load_pairs(pairs_tsv: str, cutoff: float) -> pd.DataFrame:
    df = pd.read_csv(pairs_tsv, sep="\t")

    rename = {}
    for c in df.columns:
        lc = c.lower().strip()
        if lc in ("gpcrdb", "gpcrdb_pos", "gpcrdb_position", "gpcrdb_display_generic_number"):
            rename[c] = "gpcrdb_pos"
        elif lc in ("peptide_res_index", "pep_res_index", "peptide_index", "pep_i", "peptide_pos_index"):
            rename[c] = "peptide_res_index"
        elif lc in ("peptide_aa", "pep_aa", "peptide_aa1"):
            rename[c] = "peptide_aa"
        elif lc in ("min_dist", "mindist", "min_distance"):
            rename[c] = "min_dist"
        elif lc == "pdb_id":
            rename[c] = "pdb_id"
        elif lc in ("target_chain", "receptor_chain"):
            rename[c] = "target_chain"
        elif lc in ("peptide_chain", "ligand_chain"):
            rename[c] = "peptide_chain"
        elif lc in ("segment", "gpcrdb_segment"):
            rename[c] = "segment"

    df = df.rename(columns=rename)

    needed = {"pdb_id", "target_chain", "peptide_chain", "peptide_res_index", "peptide_aa", "gpcrdb_pos", "min_dist"}
    missing = sorted(list(needed - set(df.columns)))
    if missing:
        raise ValueError(f"[pairs_tsv] colonnes manquantes: {missing}. Colonnes dispo: {df.columns.tolist()}")

    df["pdb_id"] = df["pdb_id"].map(norm_pdb)
    df["target_chain"] = df["target_chain"].astype(str).str.strip()
    df["peptide_chain"] = df["peptide_chain"].astype(str).str.strip()

    df["gpcrdb_pos"] = df["gpcrdb_pos"].map(simplify_gpcrdb_pos)
    df = df.dropna(subset=["gpcrdb_pos"]).copy()

    df["peptide_res_index"] = pd.to_numeric(df["peptide_res_index"], errors="coerce")
    df = df.dropna(subset=["peptide_res_index"]).copy()
    df["peptide_res_index"] = df["peptide_res_index"].astype(int)

    df["peptide_aa"] = df["peptide_aa"].astype(str).str.upper().str[0]

    df["min_dist"] = pd.to_numeric(df["min_dist"], errors="coerce")
    df = df.dropna(subset=["min_dist"]).copy()

    df["contact"] = (df["min_dist"].astype(float) <= float(cutoff)).astype(int)
    return df


def load_seqs(seqs_tsv: str) -> pd.DataFrame:
    df = pd.read_csv(seqs_tsv, sep="\t")

    rename = {}
    for c in df.columns:
        lc = c.lower().strip()
        if lc == "pdb_id":
            rename[c] = "pdb_id"
        elif lc in ("target_chain", "receptor_chain"):
            rename[c] = "target_chain"
        elif lc in ("peptide_chain", "ligand_chain"):
            rename[c] = "peptide_chain"
        elif lc in ("peptide_seq", "sequence", "peptide_sequence"):
            rename[c] = "peptide_seq"
        elif lc in ("peptide_length", "pep_len", "length"):
            rename[c] = "peptide_length"

    df = df.rename(columns=rename)

    if "peptide_seq" not in df.columns:
        raise ValueError(
            f"[seqs_tsv] colonne manquante: peptide_seq/sequence/peptide_sequence. Colonnes: {df.columns.tolist()}"
        )

    df["pdb_id"] = df["pdb_id"].map(norm_pdb)
    df["target_chain"] = df["target_chain"].astype(str).str.strip() if "target_chain" in df.columns else ""
    df["peptide_chain"] = df["peptide_chain"].astype(str).str.strip() if "peptide_chain" in df.columns else ""
    df["peptide_seq"] = df["peptide_seq"].astype(str).str.upper().str.replace(r"\s+", "", regex=True)

    if "peptide_length" not in df.columns:
        df["peptide_length"] = df["peptide_seq"].str.len()
    else:
        df["peptide_length"] = pd.to_numeric(df["peptide_length"], errors="coerce")
        df.loc[df["peptide_length"].isna(), "peptide_length"] = df["peptide_seq"].str.len()

    return df


def load_target_segments(target_annot_tsv: str) -> pd.DataFrame:
    df = pd.read_csv(target_annot_tsv, sep="\t")
    df["pdb_id"] = df["pdb_id"].map(norm_pdb)

    if "gpcrdb" in df.columns:
        df["gpcrdb_pos"] = df["gpcrdb"].map(simplify_gpcrdb_pos)
    elif "gpcrdb_display_generic_number" in df.columns:
        df["gpcrdb_pos"] = df["gpcrdb_display_generic_number"].map(simplify_gpcrdb_pos)
    else:
        raise ValueError(f"[target_annot_tsv] pas de colonne gpcrdb/gpcrdb_display_generic_number. Colonnes: {df.columns.tolist()}")

    if "gpcrdb_segment" not in df.columns:
        raise ValueError(f"[target_annot_tsv] colonne manquante: gpcrdb_segment.")

    out = df[["pdb_id", "gpcrdb_pos", "gpcrdb_segment"]].dropna(subset=["pdb_id", "gpcrdb_pos", "gpcrdb_segment"]).copy()
    out["gpcrdb_segment"] = out["gpcrdb_segment"].astype(str).str.strip()

    out = (
        out.groupby(["pdb_id", "gpcrdb_pos"])["gpcrdb_segment"]
        .agg(lambda x: x.mode().iloc[0] if len(x.mode()) else x.iloc[0])
        .reset_index()
    )
    return out


def load_receptor_biophys(target_annot_tsv: str) -> pd.DataFrame:
    """
    Récupère pour chaque (pdb_id, gpcrdb_pos):
      - rec_aa (1 lettre si dispo)
      - rec_biophys_class (dérivée soit des flags, soit de l'AA)
    """
    df = pd.read_csv(target_annot_tsv, sep="\t")
    df["pdb_id"] = df["pdb_id"].map(norm_pdb)

    # gpcrdb_pos
    if "gpcrdb_pos" in df.columns:
        df["gpcrdb_pos"] = df["gpcrdb_pos"].map(simplify_gpcrdb_pos)
    elif "gpcrdb" in df.columns:
        df["gpcrdb_pos"] = df["gpcrdb"].map(simplify_gpcrdb_pos)
    elif "gpcrdb_display_generic_number" in df.columns:
        df["gpcrdb_pos"] = df["gpcrdb_display_generic_number"].map(simplify_gpcrdb_pos)
    else:
        raise ValueError("[target_annot_tsv] pas de colonne gpcrdb_pos/gpcrdb/gpcrdb_display_generic_number")

    df = df.dropna(subset=["pdb_id", "gpcrdb_pos"]).copy()

    # AA: on cherche une colonne plausible
    aa_col = None
    for cand in ["aa", "aa1", "oneletter", "res_aa", "residue_aa"]:
        if cand in df.columns:
            aa_col = cand
            break
    if aa_col is None:
        df["rec_aa"] = "X"
    else:
        df["rec_aa"] = df[aa_col].astype(str).str.upper().str.strip().str[:1]
        df.loc[df["rec_aa"].str.len() == 0, "rec_aa"] = "X"

    # flags si dispo (sinon dérive de AA)
    flag_cols = ["is_hydrophobic", "is_polar", "is_pos", "is_neg", "is_aromatic", "is_other"]
    has_flags = all(c in df.columns for c in flag_cols)

    if has_flags:
        for c in flag_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

        def class_from_flags(r):
            if r["is_pos"] == 1:
                return "positive"
            if r["is_neg"] == 1:
                return "negative"
            if r["is_aromatic"] == 1:
                return "aromatic"
            if r["is_polar"] == 1:
                return "polar"
            if r["is_hydrophobic"] == 1:
                return "hydrophobic"
            if r["is_other"] == 1:
                return "other"
            return "other"

        df["rec_biophys_class"] = df.apply(class_from_flags, axis=1)
    else:
        df["rec_biophys_class"] = df["rec_aa"].map(biophys_class_from_aa)

    out = df[["pdb_id", "gpcrdb_pos", "rec_aa", "rec_biophys_class"]].copy()

    out = (
        out.groupby(["pdb_id", "gpcrdb_pos"], as_index=False)
           .agg({
               "rec_aa": lambda x: x.mode().iloc[0] if len(x.mode()) else x.iloc[0],
               "rec_biophys_class": lambda x: x.mode().iloc[0] if len(x.mode()) else x.iloc[0],
           })
    )
    return out


def load_class_map(target_annot_tsv: str) -> pd.DataFrame:
    df = pd.read_csv(target_annot_tsv, sep="\t")
    if "gpcr_class" not in df.columns:
        raise ValueError("[target_annot_tsv] colonne gpcr_class manquante (ex: 'Class A (Rhodopsin)').")
    df["pdb_id"] = df["pdb_id"].map(norm_pdb)

    out = df[["pdb_id", "gpcr_class"]].dropna().drop_duplicates().copy()

    def simplify_class(s):
        s = str(s)
        if "Class A" in s:
            return "Class A"
        if "Class B" in s:
            return "Class B"
        return "NA"

    out["class_simple"] = out["gpcr_class"].map(simplify_class)
    return out


def load_consensus_positions(consensus_dir: str, class_label: str, thr: float) -> pd.DataFrame:
    safe = class_label.replace(" ", "_")
    thr_int = int(round(thr * 100))

    p_valid = Path(consensus_dir) / f"consensus_{safe}_thr{thr_int}.validable.tsv"
    p_pos   = Path(consensus_dir) / f"consensus_{safe}_thr{thr_int}.positions.tsv"

    if p_valid.exists():
        path = p_valid
    elif p_pos.exists():
        path = p_pos
    else:
        raise FileNotFoundError(
            f"Aucun consensus trouvé. Attendu: {p_valid.name} ou {p_pos.name} dans {consensus_dir}"
        )

    df = pd.read_csv(path, sep="\t")
    if "gpcrdb_pos" not in df.columns:
        raise ValueError(f"[{path.name}] colonne gpcrdb_pos manquante. Colonnes: {df.columns.tolist()}")

    df["gpcrdb_pos"] = df["gpcrdb_pos"].map(simplify_gpcrdb_pos)
    df = df.dropna(subset=["gpcrdb_pos"]).drop_duplicates(subset=["gpcrdb_pos"]).copy()
    df["gpcrdb_pos"] = df["gpcrdb_pos"].astype(str).str.strip()

    print(f"[INFO] {class_label}: consensus loaded from {path} | n_pos={df['gpcrdb_pos'].nunique()}")
    return df


# -----------------------
# Matrices & segments
# -----------------------

def make_contact_matrix(pairs_sub: pd.DataFrame, x_positions: list[str], pep_len: int) -> np.ndarray:
    pos_to_j = {p: i for i, p in enumerate(x_positions)}
    mat = np.zeros((pep_len, len(x_positions)), dtype=int)

    for _, r in pairs_sub.iterrows():
        i = int(r["peptide_res_index"]) - 1
        if i < 0 or i >= pep_len:
            continue
        j = pos_to_j.get(r["gpcrdb_pos"])
        if j is None:
            continue
        if int(r["contact"]) == 1:
            mat[i, j] = 1

    return mat


def build_segment_band_for_structure(segs: pd.DataFrame, pdb_id: str, x_positions: list[str]) -> list[str]:
    d = segs.loc[segs["pdb_id"] == pdb_id].set_index("gpcrdb_pos")["gpcrdb_segment"].to_dict()
    return [d.get(p, "NA") for p in x_positions]


def build_receptor_ann_for_structure(recann: pd.DataFrame, pdb_id: str, x_positions: list[str]):
    """
    Retourne:
      rec_aa_band  : liste de 1-letter AA (ou 'X')
      rec_cls_band : liste classes biophys (ou 'other')
    """
    sub = recann[recann["pdb_id"] == pdb_id].set_index("gpcrdb_pos")
    aa_map = sub["rec_aa"].to_dict() if "rec_aa" in sub.columns else {}
    cl_map = sub["rec_biophys_class"].to_dict() if "rec_biophys_class" in sub.columns else {}

    rec_aa = [aa_map.get(p, "X") for p in x_positions]
    rec_cl = [cl_map.get(p, "other") for p in x_positions]
    return rec_aa, rec_cl


def compress_segments(band: list[str]):
    if not band:
        return []
    blocks = []
    start = 0
    cur = band[0]
    for i in range(1, len(band)):
        if band[i] != cur:
            blocks.append((start, i - 1, cur))
            start = i
            cur = band[i]
    blocks.append((start, len(band) - 1, cur))
    return blocks


def draw_segment_band_text(ax, band: list[str], fontsize: int = 9):
    if not band:
        return
    ax2 = ax.inset_axes([0, 1.01, 1, 0.07], transform=ax.transAxes)
    ax2.set_xlim(-0.5, len(band) - 0.5)
    ax2.set_ylim(0, 1)
    ax2.axis("off")

    for (s, e, lab) in compress_segments(band):
        rect = plt.Rectangle((s - 0.5, 0), (e - s + 1), 1, facecolor="white", edgecolor="black", linewidth=0.8)
        ax2.add_patch(rect)
        xc = (s + e) / 2.0
        ax2.text(xc, 0.5, str(lab), ha="center", va="center", fontsize=fontsize)

    ax2.text(1.005, 0.5, "segments", transform=ax2.transAxes, ha="left", va="center", fontsize=fontsize, color="black")


def draw_receptor_aa_band(ax, rec_aa_band: list[str], y0: float, height: float, fontsize: int = 9):
    """
    Bande AA récepteur: texte centré par colonne, sans couleur.
    """
    if not rec_aa_band:
        return
    ax2 = ax.inset_axes([0, y0, 1, height], transform=ax.transAxes)
    ax2.set_xlim(-0.5, len(rec_aa_band) - 0.5)
    ax2.set_ylim(0, 1)
    ax2.axis("off")

    rect = plt.Rectangle((-0.5, 0), len(rec_aa_band), 1, facecolor="white", edgecolor="black", linewidth=0.6)
    ax2.add_patch(rect)

    for j, aa in enumerate(rec_aa_band):
        ax2.text(j, 0.5, str(aa), ha="center", va="center", fontsize=fontsize, family="monospace", color="black")

    ax2.text(1.005, 0.5, "receptor AA", transform=ax2.transAxes, ha="left", va="center", fontsize=fontsize, color="black")


def draw_receptor_biophys_band(ax, rec_cls_band: list[str], y0: float, height: float):
    """
    Bande biophys récepteur: couleurs par colonne.
    """
    if not rec_cls_band:
        return
    codes = np.array([CLASS_TO_CODE.get(c, 0) for c in rec_cls_band], dtype=int)

    ax2 = ax.inset_axes([0, y0, 1, height], transform=ax.transAxes)
    ax2.imshow(codes.reshape(1, -1), aspect="auto", cmap=BIOPHYS_CMAP, vmin=0, vmax=6)
    ax2.set_xticks([])
    ax2.set_yticks([])
    for spine in ax2.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)

    ax2.text(1.005, 0.5, "receptor biophys", transform=ax2.transAxes, ha="left", va="center", fontsize=9, color="black")


# -----------------------
# Plot
# -----------------------

def plot_contact_map(
    out_png: Path,
    pdb_id: str,
    target_chain: str,
    peptide_chain: str,
    peptide_seq: str,
    x_positions: list[str],
    mat: np.ndarray,
    segment_band: list[str],
    rec_aa_band: list[str],
    rec_cls_band: list[str],
    cutoff: float,
    title_suffix: str,
):
    pep_len = len(peptide_seq)
    npos = len(x_positions)

    cmap = ListedColormap(["#2b0040", "#ffe84a"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

    fig_w = max(10, min(22, 0.40 * npos + 6))
    fig_h = max(8, min(18, 0.30 * pep_len + 5))
    fig = plt.figure(figsize=(fig_w, fig_h))

    gs = fig.add_gridspec(
        nrows=2, ncols=2,
        height_ratios=[10, 1.4],
        width_ratios=[1.2, 12],
        hspace=0.05, wspace=0.05
    )

    ax_left = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[0, 1])
    ax_bottom = fig.add_subplot(gs[1, 1])

    # main matrix
    ax.imshow(mat, aspect="auto", cmap=cmap, norm=norm)
    ax.set_ylabel("Peptide residue index", fontsize=12)
    ax.set_xlabel("GPCRdb positions", fontsize=12)

    ax.set_yticks(np.arange(pep_len))
    ax.set_yticklabels([str(i + 1) for i in range(pep_len)], fontsize=10)

    if npos == 0:
        xt = []
    else:
        step = 2 if npos > 28 else 1
        xt = np.arange(0, npos, step)

    ax.set_xticks(xt)
    ax.set_xticklabels([x_positions[i] for i in xt], rotation=60, ha="right", fontsize=9)
    ax.tick_params(axis="x", pad=6)

    ax.set_xticks(np.arange(-0.5, npos, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, pep_len, 1), minor=True)
    ax.grid(which="minor", linestyle=":", linewidth=0.35, alpha=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    # --- top annotation bands (from bottom to top):
    # 1) segments (no colors) at y=1.01
    # 2) receptor AA at y=1.09
    # 3) receptor biophys at y=1.17
    if npos > 0:
        if segment_band and len(segment_band) == npos:
            draw_segment_band_text(ax, segment_band, fontsize=9)
        if rec_aa_band and len(rec_aa_band) == npos:
            draw_receptor_aa_band(ax, rec_aa_band, y0=1.09, height=0.07, fontsize=9)
        if rec_cls_band and len(rec_cls_band) == npos:
            draw_receptor_biophys_band(ax, rec_cls_band, y0=1.17, height=0.05)

    # peptide left band (biophys)
    pep_classes = [biophys_class_from_aa(a) for a in peptide_seq]
    pep_codes = np.array([CLASS_TO_CODE.get(c, 0) for c in pep_classes], dtype=int).reshape(-1, 1)

    ax_left.imshow(pep_codes, aspect="auto", cmap=BIOPHYS_CMAP, vmin=0, vmax=6)
    ax_left.set_xticks([])
    ax_left.set_yticks(np.arange(pep_len))
    ax_left.set_yticklabels(list(peptide_seq), fontsize=10)

    # bottom peptide seq
    ax_bottom.axis("off")
    ax_bottom.text(0.0, 0.2, peptide_seq, fontsize=13, family="monospace", transform=ax_bottom.transAxes)

    # -------------------------
    # FIX: title higher + one legend outside + adjust layout
    # -------------------------
    fig.suptitle(
        f"Peptide–GPCR contact map — {pdb_id} (target {target_chain} / peptide {peptide_chain})\n"
        f"(cutoff={cutoff:.1f} Å | peptide_len={pep_len} | n_gpcrdb_pos={npos}){title_suffix}",
        fontsize=16,
        y=1.12
    )

    legend_items = ["nonpolar_aliphatic", "polar", "positive", "negative", "aromatic", "other"]
    handles = []
    for item in legend_items:
        code = CLASS_TO_CODE[item]
        handles.append(
            plt.Line2D(
                [0], [0],
                marker="s",
                color="none",
                markerfacecolor=BIOPHYS_CMAP.colors[code],
                markersize=10,
                label=item
            )
        )

    ax.legend(
        handles=handles,
        title="Biophys classes\n(pep + receptor)",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        framealpha=0.95,
        fontsize=9,
        title_fontsize=9,
        borderaxespad=0.0
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig.subplots_adjust(
        top=0.78,     # reserve space for bands + title
        right=0.82,   # reserve space for legend (outside)
        bottom=0.22
    )

    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


# -----------------------
# Main
# -----------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_tsv", required=True)
    ap.add_argument("--seqs_tsv", required=True)
    ap.add_argument("--target_annot_tsv", required=True)
    ap.add_argument("--consensus_dir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--mode", default="both", choices=["structure", "consensus", "both"])
    ap.add_argument("--classes", default="Class A,Class B")
    ap.add_argument("--threshold", type=float, default=0.50)
    ap.add_argument("--cutoff", type=float, default=5.0)
    ap.add_argument("--max_structures_per_class", type=int, default=500)
    ap.add_argument("--consensus_x_mode", default="intersection", choices=["intersection", "fixed"])
    return ap.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pairs = load_pairs(args.pairs_tsv, cutoff=args.cutoff)
    seqs = load_seqs(args.seqs_tsv)
    segs = load_target_segments(args.target_annot_tsv)
    recann = load_receptor_biophys(args.target_annot_tsv)   # NEW
    class_map = load_class_map(args.target_annot_tsv)

    pairs = pairs.merge(class_map[["pdb_id", "class_simple"]], on="pdb_id", how="left")
    seqs = seqs.merge(class_map[["pdb_id", "class_simple"]], on="pdb_id", how="left")

    wanted_classes = [c.strip() for c in args.classes.split(",") if c.strip()]

    for cl in wanted_classes:
        cl_out = outdir / cl.replace(" ", "_")
        cl_out.mkdir(parents=True, exist_ok=True)

        seqs_cl = seqs[seqs["class_simple"] == cl].copy()
        if seqs_cl.empty:
            print(f"[WARN] aucune structure trouvée pour {cl}")
            continue

        # --------------------
        # MODE 1: structure-specific
        # --------------------
        if args.mode in ("structure", "both"):
            sub_out = cl_out / "structure_specific"
            sub_out.mkdir(parents=True, exist_ok=True)

            for _, srow in seqs_cl.head(args.max_structures_per_class).iterrows():
                pdb_id = norm_pdb(srow["pdb_id"])
                tchain = str(srow["target_chain"]).strip()
                pchain = str(srow["peptide_chain"]).strip()
                pep_seq = str(srow["peptide_seq"]).strip().upper()
                pep_len = len(pep_seq)

                psub = pairs[
                    (pairs["pdb_id"] == pdb_id) &
                    (pairs["target_chain"] == tchain) &
                    (pairs["peptide_chain"] == pchain)
                ].copy()
                if psub.empty or pep_len == 0:
                    continue

                xpos = sorted(psub["gpcrdb_pos"].dropna().unique().tolist(), key=gpcrdb_sort_key)
                if len(xpos) == 0:
                    continue

                mat = make_contact_matrix(psub, xpos, pep_len)
                band = build_segment_band_for_structure(segs, pdb_id, xpos)
                rec_aa_band, rec_cls_band = build_receptor_ann_for_structure(recann, pdb_id, xpos)

                out_png = sub_out / f"{pdb_id}_{tchain}_{pchain}.structure.png"
                plot_contact_map(
                    out_png=out_png,
                    pdb_id=pdb_id,
                    target_chain=tchain,
                    peptide_chain=pchain,
                    peptide_seq=pep_seq,
                    x_positions=xpos,
                    mat=mat,
                    segment_band=band,
                    rec_aa_band=rec_aa_band,
                    rec_cls_band=rec_cls_band,
                    cutoff=args.cutoff,
                    title_suffix="",
                )

        # --------------------
        # MODE 2: consensus (par classe)
        # --------------------
        if args.mode in ("consensus", "both"):
            sub_out = cl_out / "consensus"
            sub_out.mkdir(parents=True, exist_ok=True)

            cons = load_consensus_positions(args.consensus_dir, cl, args.threshold)
            xpos_cons = sorted(cons["gpcrdb_pos"].dropna().unique().tolist(), key=gpcrdb_sort_key)

            for _, srow in seqs_cl.head(args.max_structures_per_class).iterrows():
                pdb_id = norm_pdb(srow["pdb_id"])
                tchain = str(srow["target_chain"]).strip()
                pchain = str(srow["peptide_chain"]).strip()
                pep_seq = str(srow["peptide_seq"]).strip().upper()
                pep_len = len(pep_seq)

                psub = pairs[
                    (pairs["pdb_id"] == pdb_id) &
                    (pairs["target_chain"] == tchain) &
                    (pairs["peptide_chain"] == pchain)
                ].copy()
                if psub.empty or pep_len == 0:
                    continue

                pos_struct = set(psub["gpcrdb_pos"].dropna().astype(str).str.strip().tolist())

                if args.consensus_x_mode == "intersection":
                    xpos = [p for p in xpos_cons if p in pos_struct]
                else:
                    xpos = list(xpos_cons)

                if len(xpos) == 0:
                    continue

                mat = make_contact_matrix(psub, xpos, pep_len)
                band = build_segment_band_for_structure(segs, pdb_id, xpos)
                rec_aa_band, rec_cls_band = build_receptor_ann_for_structure(recann, pdb_id, xpos)

                out_png = sub_out / f"{pdb_id}_{tchain}_{pchain}.consensus_{args.consensus_x_mode}.png"
                plot_contact_map(
                    out_png=out_png,
                    pdb_id=pdb_id,
                    target_chain=tchain,
                    peptide_chain=pchain,
                    peptide_seq=pep_seq,
                    x_positions=xpos,
                    mat=mat,
                    segment_band=band,
                    rec_aa_band=rec_aa_band,
                    rec_cls_band=rec_cls_band,
                    cutoff=args.cutoff,
                    title_suffix=f" | X={args.consensus_x_mode} of {cl} consensus (thr={int(args.threshold*100)}%)",
                )

    print("[DONE] contact maps generated:", outdir)


if __name__ == "__main__":
    main()
