#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
make_dataset_summary_table.py

USAGE 
python3 scripts/make_dataset_summary_table.py \
  --mapping_tsv run_out/gpcrdb_segments_pipeline/pdb_chain_to_uniprot_gpcrdb.tsv \
  --by_residue run_out/biophys_annotations/pocket_biophys_by_residue.with_gpcrdb_segments.tsv \
  --out_tsv run_out/dataset_summary_table.tsv
"""
import argparse
import pandas as pd
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapping_tsv", required=True,
                    help="run_out/gpcrdb_segments_pipeline/pdb_chain_to_uniprot_gpcrdb.tsv")
    ap.add_argument("--by_residue", required=True,
                    help="run_out/biophys_annotations/pocket_biophys_by_residue.with_gpcrdb_segments.tsv")
    ap.add_argument("--out_tsv", required=True,
                    help="Output summary table TSV")
    return ap.parse_args()


def main():
    args = parse_args()

    mapping = pd.read_csv(args.mapping_tsv, sep="\t", dtype=str)
    residue = pd.read_csv(args.by_residue, sep="\t", dtype=str)

    # Normalise PDB ID
    mapping["pdb_id"] = mapping["pdb_id"].str.lower()
    residue["pdb_id"] = residue["pdb_id"].str.lower()

    # Structures included in final analysis
    included_structures = set(residue["pdb_id"].unique())

    rows = []

    for _, r in mapping.iterrows():
        pdb_id = r["pdb_id"]
        uniprot = r.get("uniprot_acc", "NA")
        gpcrdb_entry = r.get("gpcrdb_entry", "NA")

        # Included?
        included = "Y" if pdb_id in included_structures else "N"

        # Segments mapped (only if included)
        if included == "Y":
            segs = (
                residue[residue["pdb_id"] == pdb_id]["gpcrdb_segment"]
                .dropna()
                .unique()
            )
            segs = sorted([s for s in segs if s != "NA"])
            segments_str = ", ".join(segs) if segs else "NA"
        else:
            segments_str = "NA"

        rows.append({
            "PDB_ID": pdb_id,
            "UniProt": uniprot,
            "GPCRdb_entry": gpcrdb_entry,
            "GPCRdb_segments_mapped": segments_str,
            "Included_in_analysis": included
        })

    summary = pd.DataFrame(rows)

    summary = summary.sort_values("PDB_ID")

    summary.to_csv(args.out_tsv, sep="\t", index=False)

    print(f"[DONE] Summary table written to: {args.out_tsv}")
    print(f"[INFO] Included structures: {summary['Included_in_analysis'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
