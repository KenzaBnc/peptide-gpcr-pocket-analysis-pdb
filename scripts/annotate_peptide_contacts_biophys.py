#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
annotate_peptide_contacts_biophys.py

BUT:
- Parcours un ensemble (pdb_id, target_chain, peptide_chain)
- Charge le mmCIF local (cif_cache/<pdb>.cif)
- Calcule les contacts résidu–résidu (min distance atomique) sous cutoff
- Annote les résidus peptide:
    - AA 1-lettre (avec mapping résidus modifiés -> canonique)
    - classe biophys (hydrophobic/polar/positive/negative/aromatic/special/other)
    - index dans la séquence peptide (1..N)
- Optionnel: merge annotations target (gpcrdb_pos, segment, etc.)

NOUVEAU (FIX X):
- Mapping étendu des résidus modifiés (MSE->M, SEP->S, TPO->T, PTR->Y, etc.)
- Report des résidus encore inconnus (*.unknown_residues.tsv)


python3 scripts/annotate_peptide_contacts_biophys.py \
  --contacts_tsv run_out/peptide_ligands_gpcr.tsv \
  --cif_cache cif_cache \
  --out_prefix run_out/biophys_annotations/peptide_contacts \
  --cutoff 5.0 \
  --target_annot_tsv run_out/biophys_annotations/pocket_biophys_by_residue.with_gpcrdb_segments.tsv
  
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import gemmi


# -----------------------
# Mapping AA (3-letter -> 1-letter)
# -----------------------

AA_3TO1_STD = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

# Résidus modifiés fréquents (mmCIF/PDB) -> AA canonique
# on pourra enrichir si le report détecte d'autres codes)
AA_3TO1_MOD = {
    # S / T / Y phosphorylés
    "SEP": "S", "TPO": "T", "PTR": "Y",

    # Selenomethionine
    "MSE": "M",

    # Histidines protonation/tautomères
    "HID": "H", "HIE": "H", "HIP": "H", "HSD": "H", "HSE": "H", "HSP": "H",

    # Cys oxydées / modifiées (souvent en structures)
    "CSO": "C", "CSD": "C", "CME": "C", "CYX": "C", "CSS": "C", "SCH": "C",

    # Met oxydée
    "MSO": "M",

    # Proline hydroxylée
    "HYP": "P",

    # Lys/Arg modifiées fréquentes
    "MLY": "K", "ALY": "K", "LLP": "K",
    "M3L": "K", "M2L": "K",
    "CIR": "R",

    # Asp/Glu amidées / variantes
    "ASX": "B",  # Asp/Asn ambigu -> on garde B (sera classé "other" si non géré)
    "GLX": "Z",  # Glu/Gln ambigu -> idem
    "UNK": "X",
    
    "PCA": "Q",   # pyroglutamate (pGlu)
    "TYS": "Y",   # sulfotyrosine
    "TYC": "Y",   # tyrosine modified
    "DTR": "W",   # D-tryptophan
    "DPN": "F",   # likely D-phenylalanine-like
    "THO": "T",   # hydroxy-threonine / threonine variant
    "RGI": "R", # <-- à éviter sans validation (optionnel)

}

# Certains fichiers mettent directement 1-letter (rare) ou noms "D" etc.
AA_1LETTER_SET = set(list("ACDEFGHIKLMNPQRSTVWY") + ["X", "B", "Z"])


def resname_to_aa1(resname: str) -> str:
    """
    Convertit un résidu mmCIF (3-letter, ou code modifié) en AA 1-letter.
    Retourne 'X' si inconnu.
    """
    if resname is None:
        return "X"
    s = str(resname).strip().upper()
    if not s:
        return "X"
    if s in AA_1LETTER_SET and len(s) == 1:
        return s
    if s in AA_3TO1_STD:
        return AA_3TO1_STD[s]
    if s in AA_3TO1_MOD:
        # si B/Z (ambigu) -> on peut préférer X pour éviter sur-interprétation
        aa = AA_3TO1_MOD[s]
        if aa in ("B", "Z"):
            return "X"
        return aa
    return "X"


# -----------------------
# Biophys class
# -----------------------

def aa_class(aa1: str) -> str:
    # Custom 6-class scheme (Lehninger 7e éd. as general reference).
    # C → hydrophobic (KD = +2.5, classé hydrophobe comme AVILM).
    # G/P → structural (contraintes conformationnelles particulières).
    aa1 = (aa1 or "X").upper()
    if aa1 in set("FWY"):
        return "aromatic"
    if aa1 in set("AVILMC"):
        return "hydrophobic"
    if aa1 in set("STNQ"):
        return "polar"
    if aa1 in set("KRH"):
        return "positive"
    if aa1 in set("DE"):
        return "negative"
    if aa1 in set("GP"):
        return "structural"
    return "other"


# -----------------------
# Gemmi helpers
# -----------------------

def get_chain(model, chain_id: str):
    chain_id = str(chain_id).strip()
    for ch in model:
        if str(ch.name).strip() == chain_id:
            return ch
    return None


def iter_residues(chain):
    for res in chain:
        # on saute waters / ligands non polymères si besoin
        yield res


def min_dist_between_residues(resA, resB):
    """
    min distance atomique entre deux résidus (ignore H).
    """
    dmin = None
    for a in resA:
        if a.element.name == "H":
            continue
        for b in resB:
            if b.element.name == "H":
                continue
            d = a.pos.dist(b.pos)
            if dmin is None or d < dmin:
                dmin = d
    return dmin


def extract_peptide_sequence(pchain):
    """
    Extrait une séquence peptide basée sur l'ordre des résidus dans la chaîne.
    Retourne:
      - table (peptide_resnum, peptide_icode, peptide_resname, peptide_aa1, peptide_pos_index)
      - peptide_seq (string)
    """
    rows = []
    pos = 0
    for res in iter_residues(pchain):
        pos += 1
        resnum = int(res.seqid.num)
        icode = (res.seqid.icode.strip() if hasattr(res.seqid, "icode") else "")
        resname = str(res.name).upper()
        # caps / terminus modifications that are not amino-acids
        if resname in {"NH2", "ACE", "NME"}:
            continue
        aa1 = resname_to_aa1(resname)

        rows.append({
            "peptide_resnum": resnum,
            "peptide_icode": icode,
            "peptide_resname": resname,
            "peptide_aa1": aa1,
            "peptide_pos_index": pos,
        })

    tab = pd.DataFrame(rows)
    pep_seq = "".join(tab["peptide_aa1"].tolist()) if not tab.empty else ""
    return tab, pep_seq


# -----------------------
# Target annot loader (optionnel)
# -----------------------

def load_target_annotations(target_annot_tsv: str) -> pd.DataFrame:
    out = pd.read_csv(target_annot_tsv, sep="\t", dtype=str)

    # --- harmoniser target_resnum
    if "target_resnum" not in out.columns:
        if "sequence_number" in out.columns:
            out["target_resnum"] = out["sequence_number"]
        elif "pocket_resi" in out.columns:
            out["target_resnum"] = out["pocket_resi"]

    out["pdb_id"] = out["pdb_id"].astype(str).str.upper()

    # colonnes minimales attendues
    needed = {"pdb_id", "target_chain", "target_resnum"}
    miss = sorted(list(needed - set(out.columns)))
    if miss:
        raise ValueError(f"[target_annot_tsv] colonnes manquantes: {miss} ; colonnes={out.columns.tolist()}")

    out["target_chain"] = out["target_chain"].astype(str).str.strip()
    out["target_resnum"] = pd.to_numeric(out["target_resnum"], errors="coerce")
    out = out.dropna(subset=["target_resnum"]).copy()
    out["target_resnum"] = out["target_resnum"].astype(int)

    # --- garder les annotations GPCRdb si elles existent
    # IMPORTANT: on NE GARDE PAS peptide_chain / pocket_chain etc. -> évite les conflits au merge.
    keep_cols = ["pdb_id", "target_chain", "target_resnum"]

    # gpcrdb_pos possible via gpcrdb_display_generic_number ou gpcrdb
    if "gpcrdb_pos" in out.columns:
        keep_cols.append("gpcrdb_pos")
    elif "gpcrdb_display_generic_number" in out.columns:
        out["gpcrdb_pos"] = out["gpcrdb_display_generic_number"]
        keep_cols.append("gpcrdb_pos")
    elif "gpcrdb" in out.columns:
        out["gpcrdb_pos"] = out["gpcrdb"]
        keep_cols.append("gpcrdb_pos")

    if "gpcrdb_segment" in out.columns:
        out["segment"] = out["gpcrdb_segment"]
        keep_cols.append("segment")
    elif "segment" in out.columns:
        keep_cols.append("segment")

    out = out[keep_cols].copy()

    # --- dédoublonnage robuste (un mapping majoritaire par (pdb, chain, resnum))
    agg = {}
    if "gpcrdb_pos" in out.columns:
        agg["gpcrdb_pos"] = lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]
    if "segment" in out.columns:
        agg["segment"] = lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]

    if agg:
        out = (
            out.groupby(["pdb_id", "target_chain", "target_resnum"], as_index=False)
               .agg(agg)
        )
    else:
        out = out.drop_duplicates(subset=["pdb_id", "target_chain", "target_resnum"])

    return out

# -----------------------
# CLI
# -----------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contacts_tsv", required=True,
                    help="TSV contenant au moins: pdb_id, target_chain, peptide_chain (une ligne par complexe)")
    ap.add_argument("--cif_cache", required=True, help="Dossier contenant les mmCIF (<pdb>.cif)")
    ap.add_argument("--out_prefix", required=True, help="Préfixe de sortie (sans extension)")
    ap.add_argument("--cutoff", type=float, default=5.0, help="cutoff distance (Å) pour contact résidu–résidu")
    ap.add_argument("--target_annot_tsv", default=None,
                    help="Optionnel: TSV d'annotations target (merge sur pdb_id,target_chain,target_resnum)")
    return ap.parse_args()


# -----------------------
# MAIN
# -----------------------

def main():
    args = parse_args()
    cif_cache = Path(args.cif_cache)
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    base = pd.read_csv(args.contacts_tsv, sep="\t")
    required = {"pdb_id", "target_chain", "peptide_chain"}
    if not required.issubset(set(base.columns)):
        raise ValueError(f"contacts_tsv doit contenir {sorted(required)} ; colonnes = {base.columns.tolist()}")

    base["pdb_id"] = base["pdb_id"].astype(str).str.upper()
    base["target_chain"] = base["target_chain"].astype(str)
    base["peptide_chain"] = base["peptide_chain"].astype(str)
    base = base.drop_duplicates(subset=["pdb_id","target_chain","peptide_chain"]).reset_index(drop=True)

    target_annot = None
    if args.target_annot_tsv:
        target_annot = load_target_annotations(args.target_annot_tsv)

    all_pairs = []
    all_pepseq = []
    unknown_rows = []  # NEW report

    for _, row in base.iterrows():
        pdb = row["pdb_id"]
        tchain_id = row["target_chain"]
        pchain_id = row["peptide_chain"]

        cif_path = cif_cache / f"{pdb.lower()}.cif"
        if not cif_path.exists():
            cif_path = cif_cache / f"{pdb}.cif"
        if not cif_path.exists():
            print(f"[WARN] missing cif for {pdb}: {cif_path}")
            continue

        st = gemmi.read_structure(str(cif_path))
        model = st[0]

        tchain = get_chain(model, tchain_id)
        pchain = get_chain(model, pchain_id)
        if tchain is None or pchain is None:
            print(f"[WARN] chain not found for {pdb}: target={tchain_id} peptide={pchain_id}")
            continue

        pep_table, pep_seq = extract_peptide_sequence(pchain)

        # NEW: report unknown peptide residues
        if not pep_table.empty:
            unk = pep_table[pep_table["peptide_aa1"] == "X"].copy()
            if not unk.empty:
                for _, ur in unk.iterrows():
                    unknown_rows.append({
                        "pdb_id": pdb,
                        "peptide_chain": pchain_id,
                        "peptide_resnum": int(ur["peptide_resnum"]),
                        "peptide_icode": str(ur["peptide_icode"] or ""),
                        "peptide_resname": str(ur["peptide_resname"]),
                        "reason": "unmapped_resname_to_aa1"
                    })

        all_pepseq.append({
            "pdb_id": pdb,
            "target_chain": tchain_id,
            "peptide_chain": pchain_id,
            "peptide_length": int(len(pep_seq)),
            "peptide_sequence": pep_seq
        })

        if pep_table.empty:
            print(f"[WARN] empty peptide sequence for {pdb} chain {pchain_id}")
            continue

        pep_key_to_pos = {}
        for _, r in pep_table.iterrows():
            key = (int(r["peptide_resnum"]), str(r["peptide_icode"] or ""))
            pep_key_to_pos[key] = int(r["peptide_pos_index"])

        t_residues = list(iter_residues(tchain))
        p_residues = list(iter_residues(pchain))

        for tres in t_residues:
            t_resnum = int(tres.seqid.num)
            t_icode = (tres.seqid.icode.strip() if hasattr(tres.seqid, "icode") else "")
            t_resname = str(tres.name).upper()

            for pres in p_residues:
                p_resnum = int(pres.seqid.num)
                p_icode = (pres.seqid.icode.strip() if hasattr(pres.seqid, "icode") else "")
                p_resname = str(pres.name).upper()

                d = min_dist_between_residues(tres, pres)
                if d is None or d > args.cutoff:
                    continue

                aa1 = resname_to_aa1(p_resname)
                pclass = aa_class(aa1)
                ppos = pep_key_to_pos.get((p_resnum, p_icode), None)

                all_pairs.append({
                    "pdb_id": pdb,
                    "target_chain": tchain_id,
                    "peptide_chain": pchain_id,
                    "target_resnum": t_resnum,
                    "target_icode": t_icode,
                    "target_resname": t_resname,
                    "peptide_resnum": p_resnum,
                    "peptide_icode": p_icode,
                    "peptide_resname": p_resname,
                    "peptide_aa1": aa1,
                    "peptide_class": pclass,
                    "peptide_pos_index": ppos,
                    "min_dist": float(d)
                })

    pairs = pd.DataFrame(all_pairs)
    pepseq = pd.DataFrame(all_pepseq)

    if pairs.empty:
        raise SystemExit("[ERROR] Aucun contact résidu–résidu détecté. Vérifie cif_cache et les chain IDs.")

    if target_annot is not None:
        pairs = pairs.merge(
            target_annot,
            on=["pdb_id", "target_chain", "target_resnum"],
            how="left"
        )

    out_pairs = str(out_prefix) + ".contacts_pairs.tsv"
    pairs.to_csv(out_pairs, sep="\t", index=False)

    grp_cols = ["pdb_id","target_chain","peptide_chain","peptide_pos_index",
                "peptide_resnum","peptide_resname","peptide_aa1","peptide_class"]
    summ = (
        pairs.dropna(subset=["peptide_pos_index"])
             .groupby(grp_cols, dropna=False)
             .agg(
                 n_target_residues=("target_resnum","nunique"),
                 min_dist_min=("min_dist","min"),
                 min_dist_mean=("min_dist","mean"),
                 n_pairs=("min_dist","size"),
             )
             .reset_index()
             .sort_values(["pdb_id","peptide_pos_index"])
    )
    out_summ = str(out_prefix) + ".peptide_residue_summary.tsv"
    summ.to_csv(out_summ, sep="\t", index=False)

    out_seq = str(out_prefix) + ".peptide_sequences.tsv"
    pepseq.to_csv(out_seq, sep="\t", index=False)

    # NEW report unknown residues
    out_unk = str(out_prefix) + ".unknown_residues.tsv"
    if unknown_rows:
        df_unk = pd.DataFrame(unknown_rows)
        df_unk = df_unk.sort_values(["pdb_id","peptide_chain","peptide_resnum"])
        df_unk.to_csv(out_unk, sep="\t", index=False)
        print(f"[INFO] unknown residue report: {out_unk} (rows={len(df_unk)})")
    else:
        # écrit un fichier vide pour stabilité pipeline
        pd.DataFrame(columns=["pdb_id","peptide_chain","peptide_resnum","peptide_icode","peptide_resname","reason"])\
          .to_csv(out_unk, sep="\t", index=False)

    print("[DONE]")
    print(f"  - pairs:   {out_pairs}   (rows={len(pairs)})")
    print(f"  - summary: {out_summ}    (rows={len(summ)})")
    print(f"  - seqs:    {out_seq}     (rows={len(pepseq)})")
    print(f"  - unknown: {out_unk}")


if __name__ == "__main__":
    main()
