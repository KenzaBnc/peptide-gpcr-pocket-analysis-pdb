# =============================================================================
# Snakefile — Pipeline complet analyse poche peptide-GPCR (PDB)
#
# Usage:
#   snakemake --cores 4                    # pipeline complet
#   snakemake --cores 1 --dryrun           # aperçu des commandes
#   snakemake --cores 4 figures            # seulement les figures
#   snakemake -R detect_peptides --cores 1 # forcer re-run d'une règle
#
# Dépendances Python:
#   pip install snakemake gemmi requests pandas numpy matplotlib logomaker
#              biopython scikit-learn seaborn scipy
# Dépendances externes (optionnelles):
#   mkdssp  (pour rule peptides_dssp : sudo apt install dssp)
#   pymol   (pour rule pymol_consensus : génère les PNG)
# =============================================================================

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
PDB_BESTHIT_TSV = "data/gpcr_70_pdb_besthit.tsv"
EVIDENCE_TSV    = "data/gpcr_70_evidence_table.tsv"
CIF_CACHE       = "cif_cache"
WORKDIR         = "run_out"
OUTDIR          = "out"
SCRIPTS         = "scripts"
SCRIPTS_VAL     = "scripts_validation_consensus"
PDBS_TXT        = "pdbs.txt"

CUTOFF          = 5.0
THRESHOLD       = 0.50
THR_INT         = 50

CLASSES         = ["Class_A", "Class_B"]
VALIDATION_PDBS = [l.strip().upper() for l in open(PDBS_TXT) if l.strip()]

# Libellés complets passés aux scripts Python (espace, pas underscore)
CLASS_LABEL = {
    "Class_A": "Class A",
    "Class_B": "Class B",
}

# Structure de référence pour l'alignement (spatial variability + PyMOL)
CLASS_REF_PDB = {
    "Class_A": "7mby",
    "Class_B": "6niy",
}


# =============================================================================
# Règle terminale — toutes les sorties finales
# =============================================================================
rule all:
    input:
        # Tables principales
        f"{WORKDIR}/peptide_ligands_gpcr.tsv",
        f"{WORKDIR}/peptide_ligands_gpcr.contacts.gpcrdb.tsv",
        f"{WORKDIR}/peptide_ligands_gpcr.pockets.gpcrdb.tsv",
        f"{WORKDIR}/dataset_summary_table.tsv",
        f"{OUTDIR}/peptide_nature_from_cif.tsv",
        # Annotations biophysiques
        f"{WORKDIR}/biophys_annotations/pocket_biophys_by_residue.with_gpcrdb_segments.with_target_resnum.tsv",
        f"{WORKDIR}/biophys_annotations/peptide_contacts.contacts_pairs.tsv",
        f"{WORKDIR}/biophys_annotations/peptide_structure_features.tsv",
        # Validation GPCRdb vs Gemmi
        f"{OUTDIR}/gpcrdb_vs_gemmi.tsv",
        f"{OUTDIR}/gpcrdb_vs_gemmi_summary.png",
        # Consensus par classe
        expand(f"{OUTDIR}/consensus_validable/consensus_{{cls}}_thr{THR_INT}.validable.tsv", cls=CLASSES),
        # Validation LOO
        expand(f"{OUTDIR}/consensus_validation/{{cls}}/leave_one_out.tsv", cls=CLASSES),
        # Figures de validation
        f"{OUTDIR}/figures/peptide_length_barplot.png",
        expand(f"{OUTDIR}/consensus_validation/{{cls}}/figures/gpcrdb_position_frequencies_{{cls}}.png", cls=CLASSES),
        expand(f"{OUTDIR}/consensus_validation/{{cls}}/figures/loo_stability_clean.png", cls=CLASSES),
        # WebLogos
        expand(f"{OUTDIR}/pocket_weblogos/{{cls}}/{{cls}}.consensus_pocket_weblogo.png", cls=CLASSES),
        # Radars KD
        expand(f"{OUTDIR}/pocket_kd_radars/{{cls}}/{{cls}}.consensus_kd_radar_mean.png", cls=CLASSES),
        # Snakeplots & helixbox SVG
        expand(f"{OUTDIR}/consensus_svg/{{cls}}.consensus_snakeplot.svg", cls=CLASSES),
        expand(f"{OUTDIR}/consensus_svg/{{cls}}.consensus_helixbox.svg", cls=CLASSES),
        # Signatures d'interaction
        expand(f"{OUTDIR}/interaction_signatures/{{cls}}_interaction_signature_thr{THR_INT}.png", cls=CLASSES),
        # Cartes de contacts (sentinel)
        f"{OUTDIR}/peptide_biophys/maps/.done",
        # Variabilité spatiale (Class A et B explicites — répertoires lowercase)
        f"{OUTDIR}/spatial_variability_classA/Class_A.volet1_pca_poses.png",
        f"{OUTDIR}/spatial_variability_classB/Class_B.volet1_pca_poses.png",
        # Scripts PyMOL
        f"{OUTDIR}/pymol_consensus_classA/consensus_superposition_Class_A.pml",
        f"{OUTDIR}/pymol_consensus_classB/consensus_superposition_Class_B.pml",
        # Analyse positionnelle
        f"{OUTDIR}/positional_analysis/.done",
        # Figures individuelles par structure (pdbs.txt)
        expand(f"{OUTDIR}/structure_pocket_figures/{{pdb}}/.done", pdb=VALIDATION_PDBS),
        # Profil dual consensus
        expand(f"{OUTDIR}/consensus_dual_profile/{{cls}}.consensus_dual_profile.png", cls=CLASSES),
        # Radars polygones biophys par classe
        expand(f"{OUTDIR}/peptide_biophys/radar_class/{{cls}}/radar_kd_{{cls}}.png", cls=CLASSES),
        # Scripts PyMOL vue poche par récepteur
        f"{OUTDIR}/pymol_pocket_views/pocket_view_Class_A_{CLASS_REF_PDB['Class_A']}.pml",
        f"{OUTDIR}/pymol_pocket_views/pocket_view_Class_B_{CLASS_REF_PDB['Class_B']}.pml",


# =============================================================================
# PHASE 1 — Détection des peptides liés + contacts + poches
# =============================================================================
rule detect_peptides:
    """
    Télécharge les mmCIF depuis RCSB PDB (avec cache local),
    détecte les peptides liés (NeighborSearch, cutoff 5 Å),
    calcule les contacts et poches par résidu.
    """
    input:
        besthit = PDB_BESTHIT_TSV,
    output:
        tsv      = f"{WORKDIR}/peptide_ligands_gpcr.tsv",
        stats    = f"{WORKDIR}/peptide_ligands_gpcr.stats.tsv",
        contacts = f"{WORKDIR}/peptide_ligands_gpcr.contacts.tsv",
        pockets  = f"{WORKDIR}/peptide_ligands_gpcr.pockets.tsv",
        log      = f"{WORKDIR}/peptide_ligands_gpcr.log",
    params:
        cutoff         = CUTOFF,
        min_len        = 0,
        max_len        = 80,
        peptide_tm_max = 1,
        target_min_len = 150,
        timeout        = 180,
    shell:
        """
        python3 {SCRIPTS}/detect_peptide_ligands_from_pdb_besthit.py \
            --pdb_besthit_tsv {input.besthit} \
            --cif_cache       {CIF_CACHE} \
            --out_tsv         {output.tsv} \
            --out_stats       {output.stats} \
            --out_residue_contacts {output.contacts} \
            --out_pockets     {output.pockets} \
            --out_log         {output.log} \
            --cutoff          {params.cutoff} \
            --min_len         {params.min_len} \
            --max_len         {params.max_len} \
            --peptide_tm_max  {params.peptide_tm_max} \
            --target_min_len  {params.target_min_len} \
            --timeout         {params.timeout}
        """


# =============================================================================
# PHASE 2 — PDB cibles + numérotation GPCRdb
# =============================================================================
rule make_target_pdbs:
    """
    Extrait les PDB GPCR seul (sans peptide, G-protein, fusions, etc.)
    depuis le cache mmCIF. Sentinel .done créé à la fin.
    """
    input:
        tsv = f"{WORKDIR}/peptide_ligands_gpcr.tsv",
    output:
        done = f"{WORKDIR}/target_pdbs/.done",
    params:
        out_dir     = f"{WORKDIR}/target_pdbs",
        keep_altloc = "A",
    shell:
        """
        python3 {SCRIPTS}/make_target_only_pdbs.py \
            --in_tsv      {input.tsv} \
            --cif_cache   {CIF_CACHE} \
            --out_dir     {params.out_dir} \
            --keep_altloc {params.keep_altloc} \
            --overwrite
        touch {output.done}
        """


rule gpcrdb_numbering:
    """
    Envoie chaque PDB cible à l'API GPCRdb assign_generic_numbers.
    Génère un mapping (chain, resnum) → numéro générique (ex: 3x50).
    Appels réseau vers gpcrdb.org — ~5 min pour 20 structures.
    """
    input:
        tsv  = f"{WORKDIR}/peptide_ligands_gpcr.tsv",
        done = f"{WORKDIR}/target_pdbs/.done",
    output:
        mapping = f"{WORKDIR}/gpcrdb_numbering/gpcrdb_numbering.mapping.tsv",
        summary = f"{WORKDIR}/gpcrdb_numbering/gpcrdb_numbering.per_pdb_summary.tsv",
        stats   = f"{WORKDIR}/gpcrdb_numbering/gpcrdb_numbering.stats.tsv",
    params:
        target_pdb_dir = f"{WORKDIR}/target_pdbs",
        out_dir        = f"{WORKDIR}/gpcrdb_numbering",
        timeout        = 180,
        sleep_s        = 0.0,
        retries        = 2,
        retry_sleep    = 2.0,
    shell:
        """
        python3 {SCRIPTS}/gpcrdb_numbering_from_target_pdbs.py \
            --in_tsv          {input.tsv} \
            --target_pdb_dir  {params.target_pdb_dir} \
            --out_dir         {params.out_dir} \
            --timeout         {params.timeout} \
            --sleep_s         {params.sleep_s} \
            --retries         {params.retries} \
            --retry_sleep     {params.retry_sleep} \
            --overwrite
        """


rule annotate_contacts_gpcrdb:
    """Ajoute les numéros génériques GPCRdb à chaque ligne du tableau contacts."""
    input:
        contacts = f"{WORKDIR}/peptide_ligands_gpcr.contacts.tsv",
        mapping  = f"{WORKDIR}/gpcrdb_numbering/gpcrdb_numbering.mapping.tsv",
    output:
        f"{WORKDIR}/peptide_ligands_gpcr.contacts.gpcrdb.tsv",
    shell:
        """
        python3 {SCRIPTS}/annotate_contacts_with_gpcrdb.py \
            --contacts_tsv {input.contacts} \
            --mapping_tsv  {input.mapping} \
            --out_tsv      {output}
        """


rule annotate_pockets_gpcrdb:
    """Ajoute les numéros génériques GPCRdb à chaque ligne du tableau poches."""
    input:
        pockets = f"{WORKDIR}/peptide_ligands_gpcr.pockets.tsv",
        mapping = f"{WORKDIR}/gpcrdb_numbering/gpcrdb_numbering.mapping.tsv",
    output:
        f"{WORKDIR}/peptide_ligands_gpcr.pockets.gpcrdb.tsv",
    shell:
        """
        python3 {SCRIPTS}/annotate_pockets_with_gpcrdb.py \
            --pockets_tsv {input.pockets} \
            --mapping_tsv {input.mapping} \
            --out_tsv     {output}
        """


# =============================================================================
# PHASE 3 — Annotations biophysiques
# =============================================================================
rule biophys_with_class:
    """
    Calcule les propriétés biophysiques (Kyte–Doolittle, groupes AA Lehninger)
    des résidus de poche et ajoute la classe GPCR (A/B) depuis la table d'évidence.
    Produit : pocket_biophys_by_residue.tsv  +  pocket_biophys_by_pocket.tsv
    """
    input:
        pockets_gpcrdb = f"{WORKDIR}/peptide_ligands_gpcr.pockets.gpcrdb.tsv",
        evidence       = EVIDENCE_TSV,
    output:
        by_residue = f"{WORKDIR}/biophys_annotations/pocket_biophys_by_residue.tsv",
        by_pocket  = f"{WORKDIR}/biophys_annotations/pocket_biophys_by_pocket.tsv",
    params:
        outdir = f"{WORKDIR}/biophys_annotations",
    shell:
        """
        python3 {SCRIPTS}/annotate_pocket_biophys_gpcrdb_with_class.py \
            --input_tsv      {input.pockets_gpcrdb} \
            --evidence       {input.evidence} \
            --outdir_biophys {params.outdir}
        """


rule gpcrdb_segments:
    """
    RCSB GraphQL → UniProt → GPCRdb pour obtenir les segments TM/ECL/ICL.
    Fusionne les segments dans by_residue (ajoute gpcrdb_segment, _category, _display).
    Appels réseau RCSB + GPCRdb avec cache local — ~10 min au premier run.
    """
    input:
        pockets_gpcrdb = f"{WORKDIR}/peptide_ligands_gpcr.pockets.gpcrdb.tsv",
        by_residue     = f"{WORKDIR}/biophys_annotations/pocket_biophys_by_residue.tsv",
    output:
        chain_map      = f"{WORKDIR}/gpcrdb_segments_pipeline/pdb_chain_to_uniprot_gpcrdb.tsv",
        seg_ext        = f"{WORKDIR}/gpcrdb_segments_pipeline/gpcrdb_segments_extended.tsv",
        by_residue_seg = f"{WORKDIR}/biophys_annotations/pocket_biophys_by_residue.with_gpcrdb_segments.tsv",
    params:
        outdir            = f"{WORKDIR}/gpcrdb_segments_pipeline",
        missing_threshold = 0.15,
        timeout           = 180,
        retries           = 2,
        sleep_s           = 0.0,
    shell:
        """
        python3 {SCRIPTS}/pdb_chain_to_gpcrdb_segments.py \
            --pairs_tsv         {input.pockets_gpcrdb} \
            --outdir            {params.outdir} \
            --timeout           {params.timeout} \
            --retries           {params.retries} \
            --sleep_s           {params.sleep_s} \
            --by_residue        {input.by_residue} \
            --out_by_residue    {output.by_residue_seg} \
            --missing_threshold {params.missing_threshold}
        """


rule add_target_resnum:
    """
    Ajoute la colonne 'target_resnum' (= pocket_resi) requise par les scripts
    weblogos, radars, spatial_variability, PyMOL et validate_consensus.
    """
    input:
        f"{WORKDIR}/biophys_annotations/pocket_biophys_by_residue.with_gpcrdb_segments.tsv",
    output:
        f"{WORKDIR}/biophys_annotations/pocket_biophys_by_residue.with_gpcrdb_segments.with_target_resnum.tsv",
    run:
        import pandas as pd
        df = pd.read_csv(input[0], sep="\t", dtype=str)
        if "target_resnum" not in df.columns:
            for src in ["pocket_resi", "sequence_number"]:
                if src in df.columns:
                    df["target_resnum"] = df[src]
                    break
        df.to_csv(output[0], sep="\t", index=False)


rule peptide_contacts_biophys:
    """
    Contacts résidu–résidu peptide ↔ GPCR (min distance atomique).
    Annote les résidus peptidiques : classe biophys, AA canonique, index séquence.
    Produit : contacts_pairs.tsv, peptide_sequences.tsv, peptide_residue_summary.tsv
    """
    input:
        ligands    = f"{WORKDIR}/peptide_ligands_gpcr.tsv",
        target_ann = f"{WORKDIR}/biophys_annotations/pocket_biophys_by_residue.with_gpcrdb_segments.tsv",
    output:
        pairs   = f"{WORKDIR}/biophys_annotations/peptide_contacts.contacts_pairs.tsv",
        seqs    = f"{WORKDIR}/biophys_annotations/peptide_contacts.peptide_sequences.tsv",
        summary = f"{WORKDIR}/biophys_annotations/peptide_contacts.peptide_residue_summary.tsv",
        unknown = f"{WORKDIR}/biophys_annotations/peptide_contacts.unknown_residues.tsv",
    params:
        out_prefix = f"{WORKDIR}/biophys_annotations/peptide_contacts",
        cutoff     = CUTOFF,
    shell:
        """
        python3 {SCRIPTS}/annotate_peptide_contacts_biophys.py \
            --contacts_tsv     {input.ligands} \
            --cif_cache        {CIF_CACHE} \
            --out_prefix       {params.out_prefix} \
            --cutoff           {params.cutoff} \
            --target_annot_tsv {input.target_ann}
        """


rule peptides_dssp:
    """
    Structure secondaire des peptides via mkdssp (DSSP).
    Prérequis : mkdssp installé (conda install -c conda-forge dssp).
    """
    input:
        seqs = f"{WORKDIR}/biophys_annotations/peptide_contacts.peptide_sequences.tsv",
    output:
        f"{WORKDIR}/biophys_annotations/peptide_structure_features.tsv",
    shell:
        """
        python3 {SCRIPTS}/peptides_dssp.py \
            --input_tsv {input.seqs} \
            --pdb_dir   {CIF_CACHE} \
            --out_tsv   {output}
        """


rule dataset_summary:
    """Table récapitulative du jeu de données (structures, classes GPCR, poches)."""
    input:
        mapping    = f"{WORKDIR}/gpcrdb_segments_pipeline/pdb_chain_to_uniprot_gpcrdb.tsv",
        by_residue = f"{WORKDIR}/biophys_annotations/pocket_biophys_by_residue.with_gpcrdb_segments.with_target_resnum.tsv",
    output:
        f"{WORKDIR}/dataset_summary_table.tsv",
    shell:
        """
        python3 {SCRIPTS}/make_dataset_summary_table.py \
            --mapping_tsv {input.mapping} \
            --by_residue  {input.by_residue} \
            --out_tsv     {output}
        """


rule peptide_nature:
    """Extrait la nature des peptides depuis les mmCIF (cyclique, D-AA, PTM, etc.)."""
    input:
        ligands = f"{WORKDIR}/peptide_ligands_gpcr.tsv",
    output:
        f"{OUTDIR}/peptide_nature_from_cif.tsv",
    shell:
        """
        python3 {SCRIPTS}/extract_peptide_nature_from_cif.py \
            --inventory_tsv {input.ligands} \
            --pdb_dir       {CIF_CACHE} \
            --out_tsv       {output}
        """


# =============================================================================
# PHASE 4 — Validation GPCRdb vs Gemmi + consensus
# =============================================================================
rule gpcrdb_validate_interactions:
    """
    Compare les contacts Gemmi (NeighborSearch) avec GPCRdb (API HTML).
    Génère 4 tableaux : comparaison brute + signatures par position/segment/interaction.
    Appels réseau : 1 par PDB dans pdbs.txt avec délai 0.5 s.
    """
    input:
        pdbs       = PDBS_TXT,
        by_residue = f"{WORKDIR}/biophys_annotations/pocket_biophys_by_residue.with_gpcrdb_segments.tsv",
    output:
        tsv              = f"{OUTDIR}/gpcrdb_vs_gemmi.tsv",
        sig_pos          = f"{OUTDIR}/gpcrdb_vs_gemmi.signature_by_pos.tsv",
        sig_seg          = f"{OUTDIR}/gpcrdb_vs_gemmi.signature_by_segment.tsv",
        sig_seg_interact = f"{OUTDIR}/gpcrdb_vs_gemmi.signature_by_segment_and_interaction.tsv",
    shell:
        """
        python3 {SCRIPTS}/gpcrdb_validate_peptide_interactions_from_html.py \
            {input.pdbs} \
            {input.by_residue} \
            {output.tsv}
        """


rule consensus_pockets:
    """
    Poche consensus par classe GPCR (seuil >= {THR_INT}% des structures).
    Mode strict : positions confirmées par les deux sources (Gemmi + GPCRdb).
    """
    input:
        cmp_tsv   = f"{OUTDIR}/gpcrdb_vs_gemmi.tsv",
        by_pocket = f"{WORKDIR}/biophys_annotations/pocket_biophys_by_pocket.tsv",
    output:
        expand(
            f"{OUTDIR}/consensus_validable/consensus_{{cls}}_thr{THR_INT}.validable.tsv",
            cls=CLASSES
        ),
        expand(
            f"{OUTDIR}/consensus_validable/consensus_{{cls}}_thr{THR_INT}.validable.meta.tsv",
            cls=CLASSES
        ),
    params:
        outdir       = f"{OUTDIR}/consensus_validable",
        threshold    = THRESHOLD,
        classes      = "Class A,Class B",
        mode         = "strict",
        exclude_pdbs = "9MNI",
    shell:
        """
        python3 {SCRIPTS}/make_consensus_pockets_by_class_strict_validable.py \
            --cmp_tsv           {input.cmp_tsv} \
            --biophys_by_pocket {input.by_pocket} \
            --outdir            {params.outdir} \
            --threshold         {params.threshold} \
            --classes           "{params.classes}" \
            --mode              {params.mode} \
            --exclude_pdbs      {params.exclude_pdbs}
        """


rule validate_consensus:
    """
    Robustesse de la poche consensus :
    leave-one-out (jackknife) + mapping des segments TM/ECL.
    """
    input:
        contacts      = f"{WORKDIR}/biophys_annotations/pocket_biophys_by_residue.with_gpcrdb_segments.with_target_resnum.tsv",
        consensus_a   = f"{OUTDIR}/consensus_validable/consensus_Class_A_thr{THR_INT}.validable.tsv",
        consensus_b   = f"{OUTDIR}/consensus_validable/consensus_Class_B_thr{THR_INT}.validable.tsv",
    output:
        expand(f"{OUTDIR}/consensus_validation/{{cls}}/leave_one_out.tsv",    cls=CLASSES),
        expand(f"{OUTDIR}/consensus_validation/{{cls}}/mapping_segments.tsv", cls=CLASSES),
        expand(f"{OUTDIR}/consensus_validation/{{cls}}/segments_summary.tsv", cls=CLASSES),
    params:
        consensus_dir = f"{OUTDIR}/consensus_validable",
        outdir        = f"{OUTDIR}/consensus_validation",
        threshold     = THRESHOLD,
        classes       = "Class A,Class B",
    shell:
        """
        python3 {SCRIPTS_VAL}/validate_consensus_pocket.py \
            --contacts_tsv  {input.contacts} \
            --consensus_dir {params.consensus_dir} \
            --classes       "{params.classes}" \
            --threshold     {params.threshold} \
            --outdir        {params.outdir}
        """


# =============================================================================
# PHASE 5 — Figures de validation
# =============================================================================
rule plot_gpcrdb_vs_gemmi:
    """Barres empilées Gemmi vs GPCRdb par segment et type d'interaction."""
    input:
        seg          = f"{OUTDIR}/gpcrdb_vs_gemmi.signature_by_segment.tsv",
        seg_interact = f"{OUTDIR}/gpcrdb_vs_gemmi.signature_by_segment_and_interaction.tsv",
    output:
        f"{OUTDIR}/gpcrdb_vs_gemmi_summary.png",
    shell:
        """
        python3 {SCRIPTS_VAL}/plot_gpcrdb_vs_gemmi_summary.py \
            --segment_tsv             {input.seg} \
            --segment_interaction_tsv {input.seg_interact} \
            --out_png                 {output}
        """


rule plot_position_frequencies:
    """
    Fréquence des positions GPCRdb dans les poches par classe (barre + segments).
    Wildcard : {{cls}} ∈ {Class_A, Class_B}
    """
    input:
        contacts  = f"{WORKDIR}/biophys_annotations/pocket_biophys_by_residue.with_gpcrdb_segments.with_target_resnum.tsv",
        consensus = f"{OUTDIR}/consensus_validable/consensus_{{cls}}_thr{THR_INT}.validable.tsv",
    output:
        f"{OUTDIR}/consensus_validation/{{cls}}/figures/gpcrdb_position_frequencies_{{cls}}.png",
    params:
        class_label = lambda wc: CLASS_LABEL[wc.cls],
        outdir      = lambda wc: f"{OUTDIR}/consensus_validation/{wc.cls}/figures",
    shell:
        """
        python3 {SCRIPTS_VAL}/plot_gpcrdb_position_frequencies.py \
            --contacts_tsv  {input.contacts} \
            --consensus_tsv {input.consensus} \
            --class_label   "{params.class_label}" \
            --outdir        {params.outdir}
        """


rule plot_loo_stability:
    """
    Figure stabilité leave-one-out : taille poche + Jaccard par structure retirée.
    Wildcard : {{cls}} ∈ {Class_A, Class_B}
    """
    input:
        loo = f"{OUTDIR}/consensus_validation/{{cls}}/leave_one_out.tsv",
    output:
        f"{OUTDIR}/consensus_validation/{{cls}}/figures/loo_stability_clean.png",
    params:
        outdir = lambda wc: f"{OUTDIR}/consensus_validation/{wc.cls}/figures",
    shell:
        """
        python3 {SCRIPTS_VAL}/plot_leave_one_out_stability.py \
            --loo_tsv {input.loo} \
            --outdir  {params.outdir}
        """


rule peptide_length_barplot:
    """Barplot longueurs peptidiques coloré par classe, fractions helix/coil."""
    input:
        ligands  = f"{WORKDIR}/peptide_ligands_gpcr.tsv",
        features = f"{WORKDIR}/biophys_annotations/peptide_structure_features.tsv",
        biophys  = f"{WORKDIR}/biophys_annotations/pocket_biophys_by_pocket.tsv",
        nature   = f"{OUTDIR}/peptide_nature_from_cif.tsv",
    output:
        png = f"{OUTDIR}/figures/peptide_length_barplot.png",
        svg = f"{OUTDIR}/figures/peptide_length_barplot.svg",
    shell:
        """
        python3 {SCRIPTS}/make_peptide_length_barplot.py \
            --ligands  {input.ligands}  \
            --features {input.features} \
            --biophys  {input.biophys}  \
            --nature   {input.nature}   \
            --outdir   {OUTDIR}/figures
        """


# =============================================================================
# PHASE 6 — Visualisations des poches consensus
# =============================================================================
rule weblogos:
    """
    WebLogos (Class A + B) : pseudo-alignement des AA de poche aux positions GPCRdb,
    colorés par classe biophysique Lehninger.
    Produit PNG + PDF + TSV fréquences (utilisés par svg_snakeplots).
    """
    input:
        pocket_tsv  = f"{WORKDIR}/biophys_annotations/pocket_biophys_by_residue.with_gpcrdb_segments.with_target_resnum.tsv",
        consensus_a = f"{OUTDIR}/consensus_validable/consensus_Class_A_thr{THR_INT}.validable.tsv",
        consensus_b = f"{OUTDIR}/consensus_validable/consensus_Class_B_thr{THR_INT}.validable.tsv",
    output:
        png_a  = f"{OUTDIR}/pocket_weblogos/Class_A/Class_A.consensus_pocket_weblogo.png",
        pdf_a  = f"{OUTDIR}/pocket_weblogos/Class_A/Class_A.consensus_pocket_weblogo.pdf",
        freq_a = f"{OUTDIR}/pocket_weblogos/Class_A/Class_A.consensus_pocket_frequencies.tsv",
        png_b  = f"{OUTDIR}/pocket_weblogos/Class_B/Class_B.consensus_pocket_weblogo.png",
        pdf_b  = f"{OUTDIR}/pocket_weblogos/Class_B/Class_B.consensus_pocket_weblogo.pdf",
        freq_b = f"{OUTDIR}/pocket_weblogos/Class_B/Class_B.consensus_pocket_frequencies.tsv",
    params:
        outdir = f"{OUTDIR}/pocket_weblogos",
    shell:
        """
        python3 {SCRIPTS}/build_consensus_pocket_weblogos.py \
            --pocket_tsv  {input.pocket_tsv} \
            --consensus_a {input.consensus_a} \
            --consensus_b {input.consensus_b} \
            --outdir      {params.outdir}
        """


rule kd_radars:
    """
    Radars Kyte–Doolittle des poches consensus :
    1 polygone par structure + courbe moyenne, par classe.
    """
    input:
        pocket_tsv  = f"{WORKDIR}/biophys_annotations/pocket_biophys_by_residue.with_gpcrdb_segments.with_target_resnum.tsv",
        consensus_a = f"{OUTDIR}/consensus_validable/consensus_Class_A_thr{THR_INT}.validable.tsv",
        consensus_b = f"{OUTDIR}/consensus_validable/consensus_Class_B_thr{THR_INT}.validable.tsv",
    output:
        expand(f"{OUTDIR}/pocket_kd_radars/{{cls}}/{{cls}}.consensus_kd_radar_mean.png",          cls=CLASSES),
        expand(f"{OUTDIR}/pocket_kd_radars/{{cls}}/{{cls}}.consensus_kd_radar_per_structure.png", cls=CLASSES),
        expand(f"{OUTDIR}/pocket_kd_radars/{{cls}}/{{cls}}.consensus_kd_radar_mean.pdf",          cls=CLASSES),
    params:
        outdir = f"{OUTDIR}/pocket_kd_radars",
    shell:
        """
        python3 {SCRIPTS}/build_consensus_pocket_kd_radars.py \
            --pocket_tsv  {input.pocket_tsv} \
            --consensus_a {input.consensus_a} \
            --consensus_b {input.consensus_b} \
            --outdir      {params.outdir}
        """


rule svg_mapping:
    """
    Construit les fichiers mapping SVG ↔ gpcrdb_pos pour les templates
    snakeplot et helixbox (Class A et B). À relancer si les templates SVG changent.
    """
    input:
        annot_tsv     = f"{WORKDIR}/biophys_annotations/pocket_biophys_by_residue.with_gpcrdb_segments.with_target_resnum.tsv",
        numbering_tsv = f"{WORKDIR}/gpcrdb_numbering/gpcrdb_numbering.mapping.tsv",
        consensus_a   = f"{OUTDIR}/consensus_validable/consensus_Class_A_thr{THR_INT}.validable.tsv",
        consensus_b   = f"{OUTDIR}/consensus_validable/consensus_Class_B_thr{THR_INT}.validable.tsv",
        snakeplot_a   = "templates/classA_snakeplot.svg",
        helixbox_a    = "templates/classA_helixbox.svg",
        snakeplot_b   = "templates/classB_snakeplot.svg",
        helixbox_b    = "templates/classB_helixbox.svg",
    output:
        map_sa = "templates/classA_snakeplot_mapping.tsv",
        map_ha = "templates/classA_helixbox_mapping.tsv",
        map_sb = "templates/classB_snakeplot_mapping.tsv",
        map_hb = "templates/classB_helixbox_mapping.tsv",
    params:
        ref_a = CLASS_REF_PDB["Class_A"].upper(),
        ref_b = CLASS_REF_PDB["Class_B"].upper(),
    shell:
        """
        python3 {SCRIPTS}/build_svg_mapping_from_template_resnums.py \
            --svg {input.snakeplot_a} --positions_tsv {input.consensus_a} \
            --annot_tsv {input.annot_tsv} --template_pdb {params.ref_a} \
            --numbering_tsv {input.numbering_tsv} \
            --out_tsv {output.map_sa}

        python3 {SCRIPTS}/build_svg_mapping_from_template_resnums.py \
            --svg {input.helixbox_a} --positions_tsv {input.consensus_a} \
            --annot_tsv {input.annot_tsv} --template_pdb {params.ref_a} \
            --numbering_tsv {input.numbering_tsv} \
            --out_tsv {output.map_ha}

        python3 {SCRIPTS}/build_svg_mapping_from_template_resnums.py \
            --svg {input.snakeplot_b} --positions_tsv {input.consensus_b} \
            --annot_tsv {input.annot_tsv} --template_pdb {params.ref_b} \
            --numbering_tsv {input.numbering_tsv} \
            --out_tsv {output.map_sb}

        python3 {SCRIPTS}/build_svg_mapping_from_template_resnums.py \
            --svg {input.helixbox_b} --positions_tsv {input.consensus_b} \
            --annot_tsv {input.annot_tsv} --template_pdb {params.ref_b} \
            --numbering_tsv {input.numbering_tsv} \
            --out_tsv {output.map_hb}
        """


rule svg_snakeplots:
    """
    Colore les SVG snakeplot et helixbox GPCRdb avec le consensus :
    couleur par classe biophysique + opacité proportionnelle à la fréquence.
    Produit 4 SVG (snakeplot A/B + helixbox A/B).
    """
    input:
        freq_a      = f"{OUTDIR}/pocket_weblogos/Class_A/Class_A.consensus_pocket_frequencies.tsv",
        freq_b      = f"{OUTDIR}/pocket_weblogos/Class_B/Class_B.consensus_pocket_frequencies.tsv",
        pocket_tsv  = f"{WORKDIR}/biophys_annotations/pocket_biophys_by_residue.with_gpcrdb_segments.with_target_resnum.tsv",
        snakeplot_a = "templates/classA_snakeplot.svg",
        map_sa      = "templates/classA_snakeplot_mapping.tsv",
        helixbox_a  = "templates/classA_helixbox.svg",
        map_ha      = "templates/classA_helixbox_mapping.tsv",
        snakeplot_b = "templates/classB_snakeplot.svg",
        map_sb      = "templates/classB_snakeplot_mapping.tsv",
        helixbox_b  = "templates/classB_helixbox.svg",
        map_hb      = "templates/classB_helixbox_mapping.tsv",
    output:
        f"{OUTDIR}/consensus_svg/Class_A.consensus_snakeplot.svg",
        f"{OUTDIR}/consensus_svg/Class_A.consensus_helixbox.svg",
        f"{OUTDIR}/consensus_svg/Class_B.consensus_snakeplot.svg",
        f"{OUTDIR}/consensus_svg/Class_B.consensus_helixbox.svg",
    params:
        outdir = f"{OUTDIR}/consensus_svg",
    shell:
        """
        python3 {SCRIPTS}/build_consensus_gpcrdb_svgs.py \
            --consensus_a     {input.freq_a} \
            --consensus_b     {input.freq_b} \
            --snakeplot_a_svg {input.snakeplot_a} \
            --snakeplot_a_map {input.map_sa} \
            --helixbox_a_svg  {input.helixbox_a} \
            --helixbox_a_map  {input.map_ha} \
            --snakeplot_b_svg {input.snakeplot_b} \
            --snakeplot_b_map {input.map_sb} \
            --helixbox_b_svg  {input.helixbox_b} \
            --helixbox_b_map  {input.map_hb} \
            --outdir          {params.outdir} \
            --pocket_tsv      {input.pocket_tsv}
        """


rule interaction_signatures:
    """
    Barres empilées des types d'interaction (hydrophobic/polar/vdw/other)
    par position GPCRdb consensus — une figure par classe.
    """
    input:
        consensus_a = f"{OUTDIR}/consensus_validable/consensus_Class_A_thr{THR_INT}.validable.tsv",
        consensus_b = f"{OUTDIR}/consensus_validable/consensus_Class_B_thr{THR_INT}.validable.tsv",
    output:
        f"{OUTDIR}/interaction_signatures/Class_A_interaction_signature_thr{THR_INT}.png",
        f"{OUTDIR}/interaction_signatures/Class_B_interaction_signature_thr{THR_INT}.png",
    params:
        outdir    = f"{OUTDIR}/interaction_signatures",
        threshold = THR_INT,
    shell:
        """
        python3 {SCRIPTS}/plot_interaction_signature_from_consensus.py \
            --consensus_a {input.consensus_a} \
            --consensus_b {input.consensus_b} \
            --outdir      {params.outdir} \
            --threshold   {params.threshold}
        """


rule contact_maps:
    """
    Cartes de contacts peptide–GPCR par structure :
    heatmap AA peptide × position GPCRdb, avec segments + biophys + consensus.
    Sentinel .done créé à la fin (sorties dynamiques : 1 PNG par structure).
    """
    input:
        pairs       = f"{WORKDIR}/biophys_annotations/peptide_contacts.contacts_pairs.tsv",
        seqs        = f"{WORKDIR}/biophys_annotations/peptide_contacts.peptide_sequences.tsv",
        target_ann  = f"{WORKDIR}/biophys_annotations/pocket_biophys_by_residue.with_gpcrdb_segments.tsv",
        consensus_a = f"{OUTDIR}/consensus_validable/consensus_Class_A_thr{THR_INT}.validable.tsv",
        consensus_b = f"{OUTDIR}/consensus_validable/consensus_Class_B_thr{THR_INT}.validable.tsv",
    output:
        done = f"{OUTDIR}/peptide_biophys/maps/.done",
    params:
        consensus_dir = f"{OUTDIR}/consensus_validable",
        outdir        = f"{OUTDIR}/peptide_biophys/maps",
        cutoff        = CUTOFF,
        threshold     = THRESHOLD,
        classes       = "Class A,Class B",
    shell:
        """
        python3 {SCRIPTS}/plot_peptide_contact_maps_with_consensus.py \
            --pairs_tsv        {input.pairs} \
            --seqs_tsv         {input.seqs} \
            --target_annot_tsv {input.target_ann} \
            --consensus_dir    {params.consensus_dir} \
            --outdir           {params.outdir} \
            --mode             both \
            --classes          "{params.classes}" \
            --threshold        {params.threshold} \
            --cutoff           {params.cutoff} \
            --consensus_x_mode intersection
        touch {output.done}
        """


# =============================================================================
# PHASE 7 — Variabilité spatiale des peptides
# (deux règles explicites car les répertoires utilisent des noms lowercase)
# =============================================================================
rule spatial_variability_classA:
    """
    Variabilité spatiale des peptides Class A après alignement des récepteurs.
    Volet 1 : centres de masse (heatmap + PCA).
    Volet 2 : profondeur poche vs déviation spatiale (scatter + boxplot).
    """
    input:
        pocket_tsv = f"{WORKDIR}/biophys_annotations/pocket_biophys_by_residue.with_gpcrdb_segments.with_target_resnum.tsv",
        consensus  = f"{OUTDIR}/consensus_validable/consensus_Class_A_thr{THR_INT}.validable.tsv",
    output:
        pca   = f"{OUTDIR}/spatial_variability_classA/Class_A.volet1_pca_poses.png",
        cm    = f"{OUTDIR}/spatial_variability_classA/Class_A.volet1_cm_distances.png",
        depth = f"{OUTDIR}/spatial_variability_classA/Class_A.volet2_scatter_depth_spatial_deviation.png",
        box   = f"{OUTDIR}/spatial_variability_classA/Class_A.volet2_boxplot_depth_zones.png",
    params:
        outdir      = f"{OUTDIR}/spatial_variability_classA",
        class_label = CLASS_LABEL["Class_A"],
        ref_pdb     = CLASS_REF_PDB["Class_A"],
    shell:
        """
        python3 {SCRIPTS}/peptide_spatial_variability.py \
            --pocket_tsv    {input.pocket_tsv} \
            --consensus_tsv {input.consensus} \
            --pdb_dir       {CIF_CACHE} \
            --class_label   "{params.class_label}" \
            --reference_pdb {params.ref_pdb} \
            --outdir        {params.outdir}
        """


rule spatial_variability_classB:
    """Variabilité spatiale des peptides Class B (voir spatial_variability_classA)."""
    input:
        pocket_tsv = f"{WORKDIR}/biophys_annotations/pocket_biophys_by_residue.with_gpcrdb_segments.with_target_resnum.tsv",
        consensus  = f"{OUTDIR}/consensus_validable/consensus_Class_B_thr{THR_INT}.validable.tsv",
    output:
        pca   = f"{OUTDIR}/spatial_variability_classB/Class_B.volet1_pca_poses.png",
        cm    = f"{OUTDIR}/spatial_variability_classB/Class_B.volet1_cm_distances.png",
        depth = f"{OUTDIR}/spatial_variability_classB/Class_B.volet2_scatter_depth_spatial_deviation.png",
        box   = f"{OUTDIR}/spatial_variability_classB/Class_B.volet2_boxplot_depth_zones.png",
    params:
        outdir      = f"{OUTDIR}/spatial_variability_classB",
        class_label = CLASS_LABEL["Class_B"],
        ref_pdb     = CLASS_REF_PDB["Class_B"],
    shell:
        """
        python3 {SCRIPTS}/peptide_spatial_variability.py \
            --pocket_tsv    {input.pocket_tsv} \
            --consensus_tsv {input.consensus} \
            --pdb_dir       {CIF_CACHE} \
            --class_label   "{params.class_label}" \
            --reference_pdb {params.ref_pdb} \
            --outdir        {params.outdir}
        """


# =============================================================================
# PHASE 8 — Scripts PyMOL (superposition consensus)
# Ces règles génèrent le script .pml.
# Les PNG sont créés si PyMOL est accessible en mode non-interactif (pymol -cq).
# =============================================================================
rule pymol_consensus_classA:
    """
    Script PyMOL : superposition Class A — récepteur de référence + tous les peptides
    alignés, résidus consensus colorés par biophys.
    """
    input:
        by_residue = f"{WORKDIR}/biophys_annotations/pocket_biophys_by_residue.with_gpcrdb_segments.with_target_resnum.tsv",
        consensus  = f"{OUTDIR}/consensus_validable/consensus_Class_A_thr{THR_INT}.validable.tsv",
    output:
        pml = f"{OUTDIR}/pymol_consensus_classA/consensus_superposition_Class_A.pml",
    params:
        outdir      = f"{OUTDIR}/pymol_consensus_classA",
        class_label = CLASS_LABEL["Class_A"],
        ref_pdb     = CLASS_REF_PDB["Class_A"],
    shell:
        """
        python3 {SCRIPTS}/make_pymol_consensus_superposition_views.py \
            --by_residue    {input.by_residue} \
            --consensus_tsv {input.consensus} \
            --pdb_dir       {CIF_CACHE} \
            --class_label   "{params.class_label}" \
            --reference_pdb {params.ref_pdb} \
            --outdir        {params.outdir} \
            --label_consensus \
            --label_mode    gpcrdb
        """


rule pymol_consensus_classB:
    """Script PyMOL : superposition Class B (voir pymol_consensus_classA)."""
    input:
        by_residue = f"{WORKDIR}/biophys_annotations/pocket_biophys_by_residue.with_gpcrdb_segments.with_target_resnum.tsv",
        consensus  = f"{OUTDIR}/consensus_validable/consensus_Class_B_thr{THR_INT}.validable.tsv",
    output:
        pml = f"{OUTDIR}/pymol_consensus_classB/consensus_superposition_Class_B.pml",
    params:
        outdir      = f"{OUTDIR}/pymol_consensus_classB",
        class_label = CLASS_LABEL["Class_B"],
        ref_pdb     = CLASS_REF_PDB["Class_B"],
    shell:
        """
        python3 {SCRIPTS}/make_pymol_consensus_superposition_views.py \
            --by_residue    {input.by_residue} \
            --consensus_tsv {input.consensus} \
            --pdb_dir       {CIF_CACHE} \
            --class_label   "{params.class_label}" \
            --reference_pdb {params.ref_pdb} \
            --outdir        {params.outdir} \
            --label_consensus \
            --label_mode    gpcrdb
        """


# =============================================================================
# PHASE 9 — Analyse positionnelle des contacts peptide–GPCR
# =============================================================================
rule positional_analysis:
    """
    Analyse structurale receptor-centrique des contacts peptide–GPCR.
    Deux analyses valides (sans normalisation de position peptidique) :
      1) Profil biophysique receptor-centrique (segment × classe biophysique)
      2) Structure secondaire peptidique × segment GPCR (DSSP agrégé)
      3) Positions GPCRdb hotspots par classe
      4) Stratification intra-Classe A (coil / helix / 9IQV microprotéine)
    """
    input:
        contacts  = f"{WORKDIR}/biophys_annotations/peptide_contacts.contacts_pairs.tsv",
        sequences = f"{WORKDIR}/biophys_annotations/peptide_contacts.peptide_sequences.tsv",
        pocket    = f"{WORKDIR}/biophys_annotations/pocket_biophys_by_pocket.tsv",
        dssp      = f"{WORKDIR}/biophys_annotations/peptide_structure_features.tsv",
    output:
        done = f"{OUTDIR}/positional_analysis/.done",
    params:
        outdir = f"{OUTDIR}/positional_analysis",
    shell:
        """
        python3 {SCRIPTS}/peptide_positional_contact_analysis.py \
            --contacts_tsv  {input.contacts} \
            --sequences_tsv {input.sequences} \
            --pocket_tsv    {input.pocket} \
            --dssp_tsv      {input.dssp} \
            --outdir        {params.outdir}
        touch {output.done}
        """


# =============================================================================
# PHASE 6 (suite) — Figures complémentaires
# =============================================================================
rule consensus_dual_profile:
    """
    Profil dual consensus : biophysique récepteur + contacts peptide, par classe.
    Produit 2 PNG (Class A + B) dans out/consensus_dual_profile/.
    """
    input:
        pocket_tsv   = f"{WORKDIR}/biophys_annotations/pocket_biophys_by_residue.with_gpcrdb_segments.with_target_resnum.tsv",
        contacts_tsv = f"{WORKDIR}/biophys_annotations/peptide_contacts.contacts_pairs.tsv",
        consensus_a  = f"{OUTDIR}/consensus_validable/consensus_Class_A_thr{THR_INT}.validable.tsv",
        consensus_b  = f"{OUTDIR}/consensus_validable/consensus_Class_B_thr{THR_INT}.validable.tsv",
    output:
        expand(f"{OUTDIR}/consensus_dual_profile/{{cls}}.consensus_dual_profile.png", cls=CLASSES),
    params:
        outdir = f"{OUTDIR}/consensus_dual_profile",
    shell:
        """
        python3 {SCRIPTS}/plot_consensus_dual_profile.py \
            --pocket_tsv    {input.pocket_tsv} \
            --contacts_tsv  {input.contacts_tsv} \
            --consensus_a   {input.consensus_a} \
            --consensus_b   {input.consensus_b} \
            --outdir        {params.outdir}
        """


rule structure_pocket_figures:
    """
    Figures snakeplot, helixbox, radar KD et weblogo pour une structure individuelle.
    Wildcard : {pdb_id} ∈ VALIDATION_PDBS (pdbs.txt, en majuscules).
    Sentinel .done créé à la fin (sorties dynamiques : 5 fichiers par structure).
    """
    input:
        pocket_tsv    = f"{WORKDIR}/biophys_annotations/pocket_biophys_by_residue.with_gpcrdb_segments.with_target_resnum.tsv",
    output:
        done = f"{OUTDIR}/structure_pocket_figures/{{pdb_id}}/.done",
    params:
        outdir        = f"{OUTDIR}/structure_pocket_figures",
        templates_dir = "templates",
    shell:
        """
        python3 {SCRIPTS}/build_structure_pocket_figures.py \
            --pocket_tsv    {input.pocket_tsv} \
            --pdb_id        {wildcards.pdb_id} \
            --outdir        {params.outdir} \
            --templates_dir {params.templates_dir}
        touch {output.done}
        """


rule radar_polygons:
    """
    Radars polygones Kyte–Doolittle avec labels structuraux (TM/ECL),
    1 polygone par structure + axe consensus, par classe.
    """
    input:
        pocket_tsv  = f"{WORKDIR}/biophys_annotations/pocket_biophys_by_residue.with_gpcrdb_segments.with_target_resnum.tsv",
        consensus_a = f"{OUTDIR}/consensus_validable/consensus_Class_A_thr{THR_INT}.validable.tsv",
        consensus_b = f"{OUTDIR}/consensus_validable/consensus_Class_B_thr{THR_INT}.validable.tsv",
    output:
        expand(f"{OUTDIR}/peptide_biophys/radar_class/{{cls}}/radar_kd_{{cls}}.png", cls=CLASSES),
    params:
        outdir        = f"{OUTDIR}/peptide_biophys/radar_class",
        consensus_dir = f"{OUTDIR}/consensus_validable",
        threshold     = THRESHOLD,
        classes       = "Class A,Class B",
    shell:
        """
        python3 {SCRIPTS}/radar_by_class_polygons_with_structural_labels.py \
            --target_annot_tsv {input.pocket_tsv} \
            --outdir           {params.outdir} \
            --consensus_dir    {params.consensus_dir} \
            --threshold        {params.threshold} \
            --classes          "{params.classes}"
        """


# =============================================================================
# PHASE 8 (suite) — Scripts PyMOL vue poche par récepteur
# =============================================================================
rule pymol_pocket_views_classA:
    """
    Script PyMOL : superposition Class A avec vue poche par récepteur,
    résidus consensus colorés, peptide semi-transparent.
    """
    input:
        by_residue = f"{WORKDIR}/biophys_annotations/pocket_biophys_by_residue.with_gpcrdb_segments.with_target_resnum.tsv",
        consensus  = f"{OUTDIR}/consensus_validable/consensus_Class_A_thr{THR_INT}.validable.tsv",
    output:
        pml = f"{OUTDIR}/pymol_pocket_views/pocket_view_Class_A_{CLASS_REF_PDB['Class_A']}.pml",
    params:
        outdir      = f"{OUTDIR}/pymol_pocket_views",
        class_label = CLASS_LABEL["Class_A"],
        ref_pdb     = CLASS_REF_PDB["Class_A"],
    shell:
        """
        python3 {SCRIPTS}/make_pymol_per_receptor_pocket_view.py \
            --by_residue    {input.by_residue} \
            --consensus_tsv {input.consensus} \
            --cif_dir       {CIF_CACHE} \
            --class_label   "{params.class_label}" \
            --reference_pdb {params.ref_pdb} \
            --outdir        {params.outdir}
        """


rule pymol_pocket_views_classB:
    """Script PyMOL : superposition Class B (voir pymol_pocket_views_classA)."""
    input:
        by_residue = f"{WORKDIR}/biophys_annotations/pocket_biophys_by_residue.with_gpcrdb_segments.with_target_resnum.tsv",
        consensus  = f"{OUTDIR}/consensus_validable/consensus_Class_B_thr{THR_INT}.validable.tsv",
    output:
        pml = f"{OUTDIR}/pymol_pocket_views/pocket_view_Class_B_{CLASS_REF_PDB['Class_B']}.pml",
    params:
        outdir      = f"{OUTDIR}/pymol_pocket_views",
        class_label = CLASS_LABEL["Class_B"],
        ref_pdb     = CLASS_REF_PDB["Class_B"],
    shell:
        """
        python3 {SCRIPTS}/make_pymol_per_receptor_pocket_view.py \
            --by_residue    {input.by_residue} \
            --consensus_tsv {input.consensus} \
            --cif_dir       {CIF_CACHE} \
            --class_label   "{params.class_label}" \
            --reference_pdb {params.ref_pdb} \
            --outdir        {params.outdir}
        """


# =============================================================================
# Alias pratiques (cibles nommées)
# =============================================================================
rule detection:
    """Phase 1+2 uniquement : détection peptides + PDB cibles + numérotation GPCRdb."""
    input:
        f"{WORKDIR}/peptide_ligands_gpcr.contacts.gpcrdb.tsv",
        f"{WORKDIR}/peptide_ligands_gpcr.pockets.gpcrdb.tsv",


rule biophys:
    """Phase 3 uniquement : annotations biophysiques complètes."""
    input:
        f"{WORKDIR}/biophys_annotations/pocket_biophys_by_residue.with_gpcrdb_segments.with_target_resnum.tsv",
        f"{WORKDIR}/biophys_annotations/peptide_contacts.contacts_pairs.tsv",


rule consensus:
    """Phase 4 uniquement : validation GPCRdb + consensus par classe."""
    input:
        expand(f"{OUTDIR}/consensus_validable/consensus_{{cls}}_thr{THR_INT}.validable.tsv", cls=CLASSES),
        expand(f"{OUTDIR}/consensus_validation/{{cls}}/leave_one_out.tsv", cls=CLASSES),


rule figures:
    """Toutes les figures (suppose que les données sont déjà calculées)."""
    input:
        f"{OUTDIR}/gpcrdb_vs_gemmi_summary.png",
        f"{OUTDIR}/figures/peptide_length_barplot.png",
        expand(f"{OUTDIR}/consensus_validation/{{cls}}/figures/gpcrdb_position_frequencies_{{cls}}.png", cls=CLASSES),
        expand(f"{OUTDIR}/consensus_validation/{{cls}}/figures/loo_stability_clean.png", cls=CLASSES),
        expand(f"{OUTDIR}/pocket_weblogos/{{cls}}/{{cls}}.consensus_pocket_weblogo.png", cls=CLASSES),
        expand(f"{OUTDIR}/pocket_kd_radars/{{cls}}/{{cls}}.consensus_kd_radar_mean.png", cls=CLASSES),
        expand(f"{OUTDIR}/consensus_svg/{{cls}}.consensus_snakeplot.svg", cls=CLASSES),
        expand(f"{OUTDIR}/consensus_svg/{{cls}}.consensus_helixbox.svg",  cls=CLASSES),
        expand(f"{OUTDIR}/interaction_signatures/{{cls}}_interaction_signature_thr{THR_INT}.png", cls=CLASSES),
        f"{OUTDIR}/peptide_biophys/maps/.done",
        f"{OUTDIR}/spatial_variability_classA/Class_A.volet1_pca_poses.png",
        f"{OUTDIR}/spatial_variability_classB/Class_B.volet1_pca_poses.png",
        expand(f"{OUTDIR}/consensus_dual_profile/{{cls}}.consensus_dual_profile.png", cls=CLASSES),
        expand(f"{OUTDIR}/structure_pocket_figures/{{pdb}}/.done", pdb=VALIDATION_PDBS),
        expand(f"{OUTDIR}/peptide_biophys/radar_class/{{cls}}/radar_kd_{{cls}}.png", cls=CLASSES),
