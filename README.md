# Peptide–GPCR Pocket Analysis (PDB)

Pipeline Snakemake d'analyse structurale des poches de liaison peptide–GPCR à partir de structures PDB.
Il caractérise les interactions, les signatures biophysiques et les positions conservées pour les récepteurs
couplés aux protéines G (GPCR) de **Classe A** (rhodopsin-like) et **Classe B1** (secretin-like).

---

## Table des matières

1. [Contexte biologique](#1-contexte-biologique)
2. [Architecture du pipeline](#2-architecture-du-pipeline)
3. [Dépendances](#3-dépendances)
4. [Structure du projet](#4-structure-du-projet)
5. [Données d'entrée](#5-données-dentrée)
6. [Jeu de données](#6-jeu-de-données)
7. [Utilisation](#7-utilisation)
8. [Sorties par phase](#8-sorties-par-phase)
9. [Description des scripts](#9-description-des-scripts)
10. [Conventions biologiques et choix méthodologiques](#10-conventions-biologiques-et-choix-méthodologiques)
11. [Notes pour la comparaison avec les microprotéines](#11-notes-pour-la-comparaison-avec-les-microprotéines)

---

## 1. Contexte biologique

Les **GPCR** (G Protein-Coupled Receptors) sont la plus grande famille de récepteurs membranaires chez les mammifères.
Ils sont la cible de ~35 % des médicaments approuvés. Les peptides endogènes constituent une grande famille
de ligands GPCR : hormones, neuropeptides, cytokines, etc.

Ce pipeline extrait et analyse la **poche de liaison peptide–GPCR** à partir de co-structures cristallographiques
déposées dans la PDB. L'objectif est d'identifier :

- les **positions GPCRdb conservées** qui définissent la poche pour chaque classe de GPCR ;
- les **classes biophysiques** des résidus peptidiques en contact ;
- les **signatures d'interaction** différenciant la Classe A et la Classe B ;
- une **table de référence structurale** pour la future comparaison avec les microprotéines.

### Numérotation GPCRdb (Ballesteros–Weinstein étendue)

La numérotation générique GPCRdb encode la position d'un résidu dans la topologie GPCR sous la forme `X.YYzZZ` :
- `X` = numéro de l'hélice (1–7 pour TM1–TM7)
- `YY` = position relative au résidu le plus conservé de l'hélice (`x50` = résidu de référence)
- `zZZ` = index absolu dans la séquence UniProt

**Important** : un numéro `x < 50` n'est pas universellement du côté IC (intracellulaire) — la direction
varie selon l'hélice. Ce pipeline utilise la numérotation GPCRdb uniquement pour comparer des positions
entre structures, pas pour inférer leur orientation.

### Classes biophysiques des acides aminés (Lehninger)

| Classe | Acides aminés | Commentaire |
|--------|---------------|-------------|
| aromatic | F, Y, W | Interactions π-stacking, aromatics lock |
| hydrophobic_aliphatic | A, V, I, L, M | Classification Lehninger 8e éd. |
| polar_uncharged | S, T, N, Q | Liaisons H sans charge formelle |
| positive | K, R, H | Chargés + à pH physiologique |
| negative | D, E | Chargés − à pH physiologique |
| other | C, G, P | C : ponts disulfures ECL ; G/P : contraintes structurales |

> **Note sur G et P** : L'échelle Kyte–Doolittle donne G = −0.4 et P = −1.6 (hydrophiles).
> Ils sont donc exclus de la classe hydrophobe aliphatique, conformément à Lehninger.
> La valeur continue `kd_score` est également disponible pour l'analyse quantitative.

### Catégories d'interaction GPCRdb

Le pipeline conserve les catégories GPCRdb **sans simplification** :
`hydrophobic` · `aromatic` · `polar` · `vdw` · `ionic`

---

## 2. Architecture du pipeline

```
Phase 1 — Détection peptides + contacts + poches (Gemmi, cutoff 5 Å)
    ↓
Phase 2 — Extraction PDB cibles + numérotation générique GPCRdb (API)
    ↓
Phase 3 — Annotations biophysiques + segments GPCRdb + DSSP peptides
    ↓
Phase 4 — Validation GPCRdb vs Gemmi + consensus strict par classe
    ↓
Phase 5 — Figures de validation (fréquences, LOO, Gemmi vs GPCRdb)
    ↓
Phase 6 — Visualisations consensus (WebLogos, radars KD, SVG, signatures)
    ↓
Phase 7 — Variabilité spatiale des peptides (PCA, profondeur poche)
    ↓
Phase 8 — Scripts PyMOL (superposition consensus)
    ↓
Phase 9 — Analyse positionnelle receptor-centrique + DSSP
```

La gestion des dépendances, la mise en cache et la parallélisation sont assurées par **Snakemake**.
Chaque règle définit ses entrées, sorties et commande shell.

---

## 3. Dépendances

### Python (≥ 3.10)

```bash
pip install snakemake gemmi requests pandas numpy matplotlib \
            logomaker biopython scikit-learn seaborn scipy
```

### Dépendances externes

| Outil | Usage | Installation |
|-------|-------|--------------|
| `mkdssp` | Structure secondaire peptides (Phase 3) | `conda install -c conda-forge dssp` |
| PyMOL | Génération des PNG consensus (Phase 8) | `conda install -c conda-forge pymol-open-source` |

> PyMOL est **optionnel** : le pipeline génère les scripts `.pml` même sans PyMOL.
> Les PNG sont produits si PyMOL est accessible en mode non-interactif (`pymol -cq`).

### Accès réseau

Plusieurs règles effectuent des appels vers des APIs externes :
- **GPCRdb** (`gpcrdb.org`) : numérotation générique (Phase 2), validation interactions (Phase 4)
- **RCSB PDB** : téléchargement des mmCIF (Phase 1, avec cache local)
- **RCSB GraphQL + UniProt** : mapping chaînes → segments GPCRdb (Phase 3)

---

## 4. Structure du projet

```
peptide_gpcr_pocket_analysis_pdb/
│
├── Snakefile                          # Pipeline complet (9 phases)
├── README.md                          # Ce fichier
│
├── data/                              # Données d'entrée (fournies)
│   ├── gpcr_70_pdb_besthit.tsv        # Meilleurs hits PDB par séquence GPCR
│   ├── gpcr_70_evidence_table.tsv     # Classes GPCRdb (A/B) par séquence
│   └── gpcr_70.fasta                  # Séquences GPCR
│
├── pdbs.txt                           # Liste des PDB à traiter (validation GPCRdb)
├── cif_cache/                         # Cache mmCIF téléchargés (auto-généré)
│
├── templates/                         # SVG templates snakeplot + helixbox
│   ├── classA_snakeplot.svg
│   ├── classA_helixbox.svg
│   ├── classB_snakeplot.svg
│   └── classB_helixbox.svg
│
├── scripts/                           # Scripts du pipeline principal
├── scripts_validation_consensus/      # Scripts de validation et figures LOO
│
├── run_out/                           # Données intermédiaires (auto-généré)
│   ├── peptide_ligands_gpcr.tsv
│   ├── peptide_ligands_gpcr.contacts.gpcrdb.tsv
│   ├── peptide_ligands_gpcr.pockets.gpcrdb.tsv
│   ├── gpcrdb_numbering/
│   ├── gpcrdb_segments_pipeline/
│   └── biophys_annotations/
│
└── out/                               # Sorties finales (figures + tables)
    ├── gpcrdb_vs_gemmi*.tsv / *.png
    ├── consensus_validable/
    ├── consensus_validation/
    ├── pocket_weblogos/
    ├── pocket_kd_radars/
    ├── consensus_svg/
    ├── interaction_signatures/
    ├── peptide_biophys/
    ├── spatial_variability_classA/
    ├── spatial_variability_classB/
    ├── pymol_consensus_classA/
    ├── pymol_consensus_classB/
    └── positional_analysis/
```

---

## 5. Données d'entrée

### `data/gpcr_70_pdb_besthit.tsv`

Table des meilleurs hits PDB pour chaque séquence GPCR (BLASTP contre la PDB).
Colonnes clés : `qseqid`, `pdb_seqid`, `pident`, `qcovs`, `bitscore`, `confidence`, `stitle`.

```
qseqid       pdb_seqid    pident   qcovs   confidence
FBpp0070550  pdb|7xjl|F   37.33    73.0    HIGH
```

### `data/gpcr_70_evidence_table.tsv`

Table d'évidence bioinformatique par séquence GPCR.
Colonnes clés : `sequence_id`, `GPCRdb__class` (ex: `Class A (Rhodopsin)`), `PDB__id`.

### `pdbs.txt`

Liste des codes PDB (un par ligne) utilisés pour la validation GPCRdb vs Gemmi (Phase 4).

---

## 6. Jeu de données

**19 structures PDB** de complexes peptide–GPCR :

| Classe | N structures | Longueurs peptides | Structure secondaire |
|--------|-------------|-------------------|---------------------|
| **Class A** (Rhodopsin-like) | 15 | 3–65 aa | Majoritairement coil (11/15 helix_fraction ≤ 0.5) |
| **Class B1** (Secretin-like) | 3 actives + 1 référence séparée | 19–69 aa | Majoritairement hélicoïdal |

> **Attention** : 9MNI est inclus dans le jeu de données PDB mais **exclu du consensus Class B**
> (voir ci-dessous). Le nombre effectif de structures pour le consensus Class B est donc **3**.

### Structures Class A (PDB)

| PDB | Longueur | Ligand | GPCR | Note |
|-----|----------|--------|------|------|
| 7XJL | 14 aa | Spexin | Galanin receptor 2 (galr2_human) | Partiellement hélicoïdal |
| 7XW9 | 4 aa | TRH peptide | TRH receptor (trfr_human) | Très court, coil pur |
| 7X1T | 5 aa | Taltirelin | TRH receptor (trfr_human) | Analogue TRH synthétique, coil pur |
| 8WZ2 | 27 aa | QRFP (26RFa) | QRFP receptor (qrfpr_human) | Mixte |
| 7YOO | 36 aa | Neuropeptide Y | NPY2R (npy2r_human) | Partiellement hélicoïdal |
| 8IBV | 22 aa | Motilin | Motilin receptor (mtlr_human) | Hélicoïdal (0.58) |
| 8JBH | 11 aa | Substance P | Neurokinin 3 receptor (nk3r_human) | Coil pur |
| 7W57 | 34 aa | Neuromedin-S | Neuromedin-U receptor 2 (nmur2_human) | Coil pur |
| 9IQV | **65 aa** | Muscarinic toxin 3 | α-1A adrenoceptor (ada1a_human) | **Knottin microprotéine** (référence) |
| 9M1O | 9 aa | NPFF-B | NPFF receptor 2 (npff2_human) | Coil pur |
| 7T11 | 8 aa | Octreotide | SST receptor 2 (ssr2_human) | Analogue somatostatine cyclique, coil pur |
| 9U4Y | 11 aa | GnRH | β-2 adrenergic receptor (adrb2_human) | Coil pur |
| 7MBY | 9 aa | CCK-8 | CCK-A receptor (cckar_human) | Coil pur |
| 9BKK | 9 aa | CCK-8 | CCK-A receptor (cckar_human) | Coil pur |
| 7RYC | 10 aa | Oxytocin | Oxytocin receptor (oxyr_human) | Cyclique, coil pur |

### Structures Class B1 (PDB)

| PDB | Longueur | Ligand | GPCR | Statut consensus | Note |
|-----|----------|--------|------|-----------------|------|
| 6NIY | 32 aa | Calcitonin | Calcitonin receptor (calcr_human) | ✓ validable | Hélicoïdal (0.74) |
| 7TS0 | 40 aa | Urocortin | CRF receptor 2 (crfr2_human) | ✓ validable | Très hélicoïdal (0.78) |
| 9BUE | 38 aa | Cagrilintide | Calcitonin receptor (calcr_human) | ⚠ non validable GPCRdb | Mixte (0.48) — analogue amyline synthétique |
| 9MNI | 69 aa | dC2_050 minibinder | CGRP receptor (calrl_human + RAMP1) | ✗ exclu | Minibinder de novo, ECD-only |

> **9IQV** est une microprotéine knottin (65 aa, riche en cystéines, structure coil malgré sa taille)
> liée à un récepteur Class A. Elle est traitée **séparément** comme référence pour la future
> comparaison avec d'autres microprotéines.
>
> **9MNI** est un minibinder de novo (dC2_050, 69 aa) conçu pour cibler le complexe
> calrl_human/RAMP1. Il est **exclu du consensus Class B** car son mode de liaison est
> fondamentalement différent des peptides endogènes : 13 contacts sur 14 se trouvent dans le
> domaine N-terminal extracellulaire (NTD) du récepteur, et non dans le bundle transmembranaire.
> De plus, ses contacts avec RAMP1 (qui forme la moitié de l'interface de liaison) ne sont pas
> capturés par le pipeline. Il est conservé dans le jeu de données comme **référence séparée
> "ECD-interface binder"**, parallèlement à 9IQV côté Class A.
>
> **9BUE** ne possède aucun contact confirmé par GPCRdb (0 contacts "both", 35 "gemmi_only") :
> GPCRdb ne dispose pas de données d'interaction pour cette structure au moment de l'analyse.
> Elle est donc absente du dénominateur en mode strict.

---

## 7. Utilisation

### Pipeline complet

```bash
snakemake --cores 4
```

### Aperçu des règles sans exécution

```bash
snakemake --dryrun --quiet
```

### Cibles partielles (alias)

```bash
snakemake --cores 1 detection     # Phase 1+2 uniquement
snakemake --cores 1 biophys       # Phase 3 uniquement
snakemake --cores 1 consensus     # Phase 4 uniquement
snakemake --cores 4 figures       # Toutes les figures (données pré-calculées)
```

### Forcer la ré-exécution d'une règle

```bash
snakemake -R positional_analysis --cores 1
snakemake -R peptides_dssp --cores 1
```

### Paramètres configurables (en tête du Snakefile)

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `CUTOFF` | `5.0` Å | Distance maximale pour un contact résidu–résidu |
| `THRESHOLD` | `0.50` | Fréquence minimale pour inclusion dans le consensus |
| `THR_INT` | `50` | Idem en entier (= 50 %) pour les noms de fichiers |
| `CIF_CACHE` | `cif_cache/` | Répertoire de cache des mmCIF |
| `WORKDIR` | `run_out/` | Données intermédiaires |
| `OUTDIR` | `out/` | Sorties finales |

---

## 8. Sorties par phase

### Phase 1 — Détection peptides + contacts + poches

**Règle** : `detect_peptides`  
**Script** : `detect_peptide_ligands_from_pdb_besthit.py`

| Fichier | Description |
|---------|-------------|
| `run_out/peptide_ligands_gpcr.tsv` | Inventaire des peptides détectés (PDB, chaînes, longueur, séquence) |
| `run_out/peptide_ligands_gpcr.contacts.tsv` | Contacts résidu–résidu (distance atomique minimale) |
| `run_out/peptide_ligands_gpcr.pockets.tsv` | Résidus GPCR en contact avec le peptide |
| `run_out/peptide_ligands_gpcr.stats.tsv` | Statistiques par structure |

**Critères de détection** :
- Peptide : 0–80 résidus, ≤ 1 hélice TM prédite, cutoff 5 Å
- Récepteur : ≥ 150 résidus

---

### Phase 2 — PDB cibles + numérotation GPCRdb

**Règles** : `make_target_pdbs`, `gpcrdb_numbering`, `annotate_contacts_gpcrdb`, `annotate_pockets_gpcrdb`

| Fichier | Description |
|---------|-------------|
| `run_out/target_pdbs/*.pdb` | Structures GPCR seules (sans peptide, G-protein, fusions) |
| `run_out/gpcrdb_numbering/gpcrdb_numbering.mapping.tsv` | Mapping (pdb, chaîne, resnum) → position GPCRdb (ex: `3x50`) |
| `run_out/peptide_ligands_gpcr.contacts.gpcrdb.tsv` | Contacts annotés avec positions GPCRdb |
| `run_out/peptide_ligands_gpcr.pockets.gpcrdb.tsv` | Poches annotées avec positions GPCRdb |

> **Appels réseau** : ~5 min pour 20 structures. La numérotation est obtenue via l'API
> `gpcrdb.org/services/structure/assign_generic_numbers`.

---

### Phase 3 — Annotations biophysiques + DSSP

**Règles** : `biophys_with_class`, `gpcrdb_segments`, `add_target_resnum`, `peptide_contacts_biophys`, `peptides_dssp`, `dataset_summary`, `peptide_nature`

#### Fichiers principaux

| Fichier | Description |
|---------|-------------|
| `run_out/biophys_annotations/pocket_biophys_by_residue.with_gpcrdb_segments.with_target_resnum.tsv` | Résidus GPCR de poche : position GPCRdb, segment TM/ECL, kd_score, classe AA |
| `run_out/biophys_annotations/pocket_biophys_by_pocket.tsv` | Agrégat par structure : fractions biophysiques, kd moyen, classe GPCR |
| `run_out/biophys_annotations/peptide_contacts.contacts_pairs.tsv` | **Table centrale** : chaque contact résidu_peptide ↔ résidu_GPCR avec position GPCRdb, segment, classe biophysique peptidique |
| `run_out/biophys_annotations/peptide_contacts.peptide_sequences.tsv` | Séquences et longueurs des peptides |
| `run_out/biophys_annotations/peptide_structure_features.tsv` | Features DSSP par peptide : `helix_fraction`, `coil_fraction`, `end_to_end_distance` |
| `out/peptide_nature_from_cif.tsv` | Nature des peptides depuis mmCIF (cyclique, D-AA, PTM, etc.) |
| `run_out/dataset_summary_table.tsv` | Table récapitulative du jeu de données |

#### Colonnes clés de `peptide_contacts.contacts_pairs.tsv`

| Colonne | Description |
|---------|-------------|
| `pdb_id` | Code PDB (majuscules) |
| `peptide_chain` | Identifiant de chaîne du peptide |
| `peptide_pos_index` | Position du résidu dans le peptide (1-indexée, N-term = 1) |
| `peptide_aa1` | Code 1-lettre de l'acide aminé peptidique |
| `peptide_class` | Classe biophysique : `aromatic` / `hydrophobic` / `polar` / `positive` / `negative` / `special` / `other` |
| `gpcrdb_pos` | Position GPCRdb du résidu GPCR (ex: `3x32`, `45x50`) |
| `segment` | Segment GPCR : `TM1`–`TM7`, `ECL1`–`ECL3`, `N-term` |
| `min_dist` | Distance atomique minimale (Å) entre les deux résidus |

---

### Phase 4 — Validation GPCRdb vs Gemmi + consensus

**Règles** : `gpcrdb_validate_interactions`, `consensus_pockets`, `validate_consensus`

| Fichier | Description |
|---------|-------------|
| `out/gpcrdb_vs_gemmi.tsv` | Comparaison position par position : `both` / `gemmi_only` / `gpcrdb_only` |
| `out/gpcrdb_vs_gemmi.signature_by_pos.tsv` | Fréquence des positions GPCRdb par source |
| `out/gpcrdb_vs_gemmi.signature_by_segment.tsv` | Fréquence par segment et source |
| `out/gpcrdb_vs_gemmi.signature_by_segment_and_interaction.tsv` | Fréquence par segment × type interaction × source |
| `out/consensus_validable/consensus_Class_A_thr50.validable.tsv` | **Poche consensus Class A** (≥ 50 % structures, confirmé Gemmi + GPCRdb) |
| `out/consensus_validable/consensus_Class_B_thr50.validable.tsv` | **Poche consensus Class B** |
| `out/consensus_validation/Class_A/leave_one_out.tsv` | Robustesse LOO : taille poche + Jaccard par structure retirée |

#### Poches consensus (seuil 50 %, mode strict)

**Class A** — 16 positions : TM3 (4), ECL2 (3), TM6 (3), TM7 (3), TM2 (1), TM4 (1), TM5 (1)  
**Class B** — 23 positions : TM1 (5), TM7 (5), TM5 (4), TM2 (3), TM3 (3), TM6 (2), ECL2 (1)

> **Mode strict** : une position est retenue si elle apparaît dans ≥ 50 % des structures ET
> est confirmée à la fois par Gemmi (NeighborSearch) et GPCRdb (API HTML).

> **Limitations du consensus Class B** :
> - **9MNI est exclu** du calcul (paramètre `--exclude_pdbs 9MNI` dans `consensus_pockets`).
>   Son mode de liaison ECD-only n'est pas représentatif des peptides Class B1 endogènes.
> - **9BUE n'est pas validable** en mode strict (0 contacts confirmés par GPCRdb) et n'entre
>   pas dans le dénominateur.
> - En conséquence, le consensus Class B repose **effectivement sur 2 structures seulement**
>   (6NIY et 7TS0). Les 23 positions correspondent aux résidus présents dans ces deux structures
>   simultanément. Cette fragilité statistique doit être mentionnée lors de toute interprétation.

---

### Phase 5 — Figures de validation

**Règles** : `plot_gpcrdb_vs_gemmi`, `plot_position_frequencies`, `plot_loo_stability`

| Fichier | Description |
|---------|-------------|
| `out/gpcrdb_vs_gemmi_summary.png` | Barres empilées Gemmi vs GPCRdb par segment et type d'interaction |
| `out/consensus_validation/Class_A/figures/gpcrdb_position_frequencies_Class_A.png` | Fréquence des positions GPCRdb (Class A) |
| `out/consensus_validation/Class_B/figures/gpcrdb_position_frequencies_Class_B.png` | Fréquence des positions GPCRdb (Class B) |
| `out/consensus_validation/*/figures/loo_stability_clean.png` | Stabilité leave-one-out |

---

### Phase 6 — Visualisations des poches consensus

**Règles** : `weblogos`, `kd_radars`, `svg_mapping`, `svg_snakeplots`, `interaction_signatures`, `contact_maps`

| Fichier | Description |
|---------|-------------|
| `out/pocket_weblogos/Class_A/Class_A.consensus_pocket_weblogo.png` | WebLogo : fréquences AA aux positions consensus (Class A) |
| `out/pocket_weblogos/Class_B/Class_B.consensus_pocket_weblogo.png` | WebLogo Class B |
| `out/pocket_kd_radars/Class_A/Class_A.consensus_kd_radar_mean.png` | Radar Kyte–Doolittle des positions consensus (Class A) |
| `out/pocket_kd_radars/Class_B/Class_B.consensus_kd_radar_mean.png` | Radar Kyte–Doolittle (Class B) |
| `out/consensus_svg/Class_A.consensus_snakeplot.svg` | Snakeplot GPCRdb coloré par biophysique (Class A) |
| `out/consensus_svg/Class_A.consensus_helixbox.svg` | Helixbox GPCRdb coloré (Class A) |
| `out/interaction_signatures/Class_A_interaction_signature_thr50.png` | Types d'interaction GPCRdb par position consensus (Class A) |
| `out/peptide_biophys/maps/*.png` | Cartes de contacts peptide–GPCR par structure |

---

### Phase 7 — Variabilité spatiale

**Règles** : `spatial_variability_classA`, `spatial_variability_classB`

| Fichier | Description |
|---------|-------------|
| `out/spatial_variability_classA/Class_A.volet1_pca_poses.png` | PCA des centres de masse des peptides (Class A) |
| `out/spatial_variability_classA/Class_A.volet1_cm_distances.png` | Distances inter-centres de masse |
| `out/spatial_variability_classA/Class_A.volet2_scatter_depth_spatial_deviation.png` | Profondeur poche vs déviation spatiale |
| `out/spatial_variability_classA/Class_A.volet2_boxplot_depth_zones.png` | Distribution par zone de profondeur |
| *(idem pour Class_B)* | |

---

### Phase 8 — Scripts PyMOL

**Règles** : `pymol_consensus_classA`, `pymol_consensus_classB`

| Fichier | Description |
|---------|-------------|
| `out/pymol_consensus_classA/consensus_superposition_Class_A.pml` | Script PyMOL : superposition Class A avec résidus consensus colorés |
| `out/pymol_consensus_classB/consensus_superposition_Class_B.pml` | Script PyMOL Class B |

Exécution manuelle :
```bash
pymol -cq out/pymol_consensus_classA/consensus_superposition_Class_A.pml
```

---

### Phase 9 — Analyse positionnelle receptor-centrique

**Règle** : `positional_analysis`  
**Script** : `peptide_positional_contact_analysis.py`

> **Approche** : Deux analyses structuralement valides, **sans** normalisation de position peptidique
> (qui serait invalide pour comparer des peptides de 3 à 65 aa sur un axe commun).

#### Analyse 1 — Profil biophysique receptor-centrique

**Fichier** : `out/positional_analysis/contact_profile_classA_classB.png`

Pour chaque segment GPCR, distribution (%) des classes biophysiques des résidus peptidiques en contact.
Résultats biologiquement cohérents :

- **Class A** : dominance **aromatique** (33 %) → TM7 > ECL2 > TM6 > TM3. Pocket orthostérique aromatique caractéristique des récepteurs rhodopsin-like.
- **Class B** : équipartition **polaire + hydrophobe** (35 % chacun) → N-term > TM1. Interaction amphipatique typique des hormones peptidiques sur les récepteurs secretin-like.

#### Analyse 2 — Structure secondaire peptidique × segment GPCR

**Fichier** : `out/positional_analysis/dssp_segment_profile.png`

Pour chaque segment GPCR, fraction des contacts provenant de peptides hélicoïdaux
(helix_fraction > 0.5) ou en coil.

- TM1 Class B : majoritairement contacté par des **peptides hélicoïdaux** → cohérent avec l'insertion d'une hélice amphipatique
- TM3/TM7 Class A : principalement contacté par des **peptides en coil**

#### Analyse 3 — Positions GPCRdb hotspots

**Fichiers** : `gpcrdb_hotspots_Class_A.png`, `gpcrdb_hotspots_Class_B.png`, `gpcrdb_hotspots_9IQV_microprotein.png`

Top 20 (Class A) / 15 (Class B / 9IQV) positions GPCRdb les plus contactées,
colorées par la classe biophysique dominante du résidu peptidique.

#### Analyse 4 — Stratification intra-Classe A

**Fichier** : `out/positional_analysis/classA_stratification.png`

La Classe A est biologiquement hétérogène (peptides de 3 à 65 aa). Trois sous-groupes :

| Groupe | N structures | Critère | Profil dominant |
|--------|-------------|---------|-----------------|
| Coil-dominant | 11 | helix_fraction ≤ 0.5 | Aromatique, TM7/ECL2 |
| Helix-dominant | 3 | helix_fraction > 0.5 | Mixte, ECL2/TM6 |
| 9IQV microprotéine | 1 | Knottin 65 aa | Aromatique, ECL2/TM7 |

#### Tables de référence

| Fichier | Description |
|---------|-------------|
| `out/positional_analysis/receptor_reference_table.tsv` | Contacts par (classe, groupe DSSP, segment, classe biophysique) |
| `out/positional_analysis/gpcrdb_positions_reference_table.tsv` | Contacts par position GPCRdb exacte |
| `out/positional_analysis/analysis_summary.txt` | Résumé statistique textuel |

Ces tables sont prêtes pour la **comparaison directe avec de nouvelles microprotéines**.

---

## 9. Description des scripts

### Scripts du pipeline principal (`scripts/`)

| Script | Phase | Description |
|--------|-------|-------------|
| `detect_peptide_ligands_from_pdb_besthit.py` | 1 | Téléchargement mmCIF (RCSB), détection peptides (NeighborSearch Gemmi), calcul contacts et poches |
| `make_target_only_pdbs.py` | 2 | Extraction PDB récepteur seul (supprime peptide, G-protein, fusions) |
| `gpcrdb_numbering_from_target_pdbs.py` | 2 | API GPCRdb `assign_generic_numbers` → mapping resnum → position générique |
| `annotate_contacts_with_gpcrdb.py` | 2 | Jointure contacts + mapping GPCRdb |
| `annotate_pockets_with_gpcrdb.py` | 2 | Jointure poches + mapping GPCRdb |
| `annotate_pocket_biophys_gpcrdb_with_class.py` | 3 | Calcul kd_score (Kyte–Doolittle), classification AA Lehninger, ajout classe GPCR |
| `pdb_chain_to_gpcrdb_segments.py` | 3 | RCSB GraphQL → UniProt → GPCRdb → segments TM/ECL/ICL par chaîne |
| `annotate_peptide_contacts_biophys.py` | 3 | Contacts résidu–résidu peptide ↔ GPCR, annotation biophysique peptide |
| `peptides_dssp.py` | 3 | Structure secondaire peptides via `mkdssp` : helix_fraction, coil_fraction, end-to-end distance |
| `make_dataset_summary_table.py` | 3 | Table récapitulative (structures, classes, n_résidus) |
| `extract_peptide_nature_from_cif.py` | 3 | Détection D-AA, PTM, liaisons cycliques depuis mmCIF |
| `gpcrdb_validate_peptide_interactions_from_html.py` | 4 | Récupération interactions GPCRdb via HTML/JS, comparaison avec Gemmi |
| `make_consensus_pockets_by_class_strict_validable.py` | 4 | Poche consensus par classe (seuil %, mode strict bivalidé) |
| `build_consensus_pocket_weblogos.py` | 6 | WebLogos des poches consensus (logomaker, colorisation Lehninger) |
| `build_consensus_pocket_kd_radars.py` | 6 | Radars Kyte–Doolittle (un polygone par structure + courbe moyenne) |
| `build_svg_mapping_from_template_resnums.py` | 6 | Génère les fichiers mapping SVG ↔ positions GPCRdb |
| `build_consensus_gpcrdb_svgs.py` | 6 | Colore les SVG snakeplot/helixbox par fréquence + biophysique |
| `plot_interaction_signature_from_consensus.py` | 6 | Barres empilées types d'interaction GPCRdb par position consensus |
| `plot_peptide_contact_maps_with_consensus.py` | 6 | Cartes de contacts heatmap AA peptide × position GPCRdb |
| `peptide_spatial_variability.py` | 7 | PCA poses peptides, distances inter-CM, profondeur poche |
| `make_pymol_consensus_superposition_views.py` | 8 | Génère scripts PyMOL avec superposition + coloration consensus |
| `peptide_positional_contact_analysis.py` | 9 | **Analyse receptor-centrique + DSSP** (voir Phase 9 ci-dessus) |

### Scripts de validation (`scripts_validation_consensus/`)

| Script | Description |
|--------|-------------|
| `validate_consensus_pocket.py` | Leave-one-out jackknife, Jaccard, mapping segments |
| `plot_gpcrdb_position_frequencies.py` | Fréquence des positions GPCRdb par structure et par classe |
| `plot_leave_one_out_stability.py` | Figure stabilité LOO (taille poche + Jaccard) |
| `plot_gpcrdb_vs_gemmi_summary.py` | Comparaison Gemmi vs GPCRdb par segment × type interaction |

### Scripts standalone (hors pipeline Snakemake)

| Script | Description |
|--------|-------------|
| `scripts/compare_pockets_gpcrdb.py` | Comparaison exploratoire de poches GPCRdb |
| `scripts/build_peptide_feature_table.py` | Table de features peptidiques étendue |
| `scripts/radar_by_class_polygons_with_structural_labels.py` | Radars alternatifs avec annotations structurales |
| `scripts/plot_peptide_contact_maps_with_consensus_modified.py` | Version modifiée des cartes de contacts |
| `scripts/make_pymol_gallery_receptor_only.py` | Galerie PyMOL récepteurs seuls |
| `scripts/show_all_pdb_peptides_grid_by_class.pml` | Script PyMOL : grille de toutes les structures par classe |
| `scripts/show_all_pdb_peptides_only.pml` | Script PyMOL : tous les peptides superposés |

---

## 10. Conventions biologiques et choix méthodologiques

### Calcul des contacts (Gemmi NeighborSearch)

- **Cutoff** : 5.0 Å (distance atomique minimale entre atomes lourds)
- Les contacts incluent van der Waals, liaisons H directes, contacts hydrophobes
- Un résidu peptidique peut avoir plusieurs contacts avec plusieurs résidus GPCR

### Numérotation GPCRdb

- Mapping obtenu via l'API officielle `gpcrdb.org`
- Les positions sans numéro générique (boucles non conservées, termini) sont exclues de la poche consensus
- La numérotation encode la conservation fonctionnelle inter-espèces et inter-GPCR

### Consensus strict (bivalidé)

Une position est dans la poche consensus si et seulement si :
1. Elle apparaît dans ≥ 50 % des structures de sa classe (Gemmi)
2. Elle est confirmée par GPCRdb pour au moins une structure

Ce mode **strict** réduit les faux positifs liés aux artefacts cristallographiques.

### DSSP (structure secondaire peptides)

- **Outil** : `mkdssp` (DSSP 4.x, Hekkelman)
- **Classification** : H, G, I → hélice ; tout le reste (P, T, S, C, " ") → coil
- **Seuil** : `helix_fraction > 0.5` → peptide "hélicoïdal dominant" au niveau de l'analyse positionnelle
- Seule la fraction agrégée est utilisée (pas de DSSP résidu par résidu)

### Kyte–Doolittle (hydrophobicité)

Valeurs continues utilisées pour les radars et la colonne `kd_score` :

```
A=1.8  R=-4.5  N=-3.5  D=-3.5  C=2.5  Q=-3.5  E=-3.5  G=-0.4
H=-3.2  I=4.5  L=3.8  K=-3.9  M=1.9  F=2.8  P=-1.6  S=-0.8
T=-0.7  W=-0.9  Y=-1.3  V=4.2
```

G (−0.4) et P (−1.6) sont légèrement hydrophiles → cohérent avec leur exclusion de `AA_HYDROPHOBIC_ALIPHATIC`.

---

## 11. Notes pour la comparaison avec les microprotéines

La Phase 9 produit des tables de référence conçues pour comparer de nouvelles microprotéines avec le profil des peptides connus.

### Workflow de comparaison (futur)

1. Détecter les contacts microprotéine–GPCR avec le même pipeline (cutoff 5 Å, numérotation GPCRdb)
2. Calculer DSSP de la microprotéine
3. Joindre avec `receptor_reference_table.tsv` sur `(segment, peptide_class)`
4. Comparer avec `gpcrdb_positions_reference_table.tsv` sur `gpcrdb_pos`

### Cas 9IQV — Microprotéine knottin de référence (Class A)

- **Structure** : knottin (inhibitory cystine knot), 65 aa, pont disulfures multiples
- **Ligand** : Muscarinic toxin 3 (venin de serpent)
- **Récepteur** : α-1A adrenorécepteur (Class A, ada1a_human)
- **Profil** : ECL2 > TM7 > TM2 — similaire au profil Class A coil-dominant
- **Classe biophysique** : aromatique (26 %) + hydrophobe (25 %) + positif (18 %)
- **Intérêt** : démontre qu'une microprotéine peut occuper la poche orthostérique d'un GPCR Class A
  avec un profil d'interaction proche des peptides courts naturels

### Cas 9MNI — Minibinder de novo de référence (Class B, ECD-interface)

- **Structure** : minibinder de novo (dC2_050), 69 aa, hélice amphipatique
- **Récepteur** : récepteur CGRP = calrl_human + RAMP1 (Class B1, calrl_human)
- **Profil** : 13/14 contacts dans le NTD extracellulaire de calrl_human (positions 98x–113x), 1 résidu TM1
- **Exclu du consensus Class B** : mode de liaison ECD-only, non représentatif des peptides endogènes
- **Intérêt** : référence pour les microprotéines ciblant l'interface ECD/RAMP d'un GPCR Class B1 ;
  interface distincte de la poche TM orthostérique

### Métriques disponibles dans les tables de référence

| Colonne | Usage pour comparaison |
|---------|----------------------|
| `segment` | Identifier quels segments la microprotéine contacte |
| `peptide_class` | Comparer la composition biophysique des résidus en contact |
| `pct_in_segment` | % relatif dans le segment (normalise la taille de la poche) |
| `gpcrdb_pos` | Positions exactes — vérifier si la microprotéine touche les mêmes résidus |
| `n_contacts` | Volume d'interaction (corrélé à l'affinité) |

---

## 12. Résultats principaux

Cette section résume les résultats biologiques du pipeline, destinés à alimenter un rapport de recherche.

### 12.1 Caractérisation du jeu de données

Le jeu de données comprend **19 co-structures peptide–GPCR** couvrant **16 récepteurs humains distincts** (ada1a, adrb2, calcr, calrl, cckar, crfr2, galr2, mtlr, nk3r, nmur2, npff2, npy2r, oxyr, qrfpr, ssr2, trfr).

**Class A (15 structures, 14 actives + 9IQV référence microprotéine) :**
- Peptides de 3 à 65 aa (médiane ~9 aa hors 9IQV)
- Structure secondaire majoritairement coil : 11/14 structures ont une helix_fraction ≤ 0.5
- Ligands endogènes variés : neuropeptides (NPY, Substance P), hormones (Oxytocine, GnRH), analogues synthétiques (Octreotide, Taltirelin), knottin (9IQV)
- 841 contacts résidu–résidu analysés (hors 9IQV)

**Class B1 (4 structures, 3 actives + 9MNI référence ECD-interface) :**
- Peptides de 32 à 69 aa
- Structure secondaire majoritairement hélicoïdale : 3/3 structures actives ont une helix_fraction > 0.5 (0.74 à 0.78)
- Ligands : hormones peptidiques naturelles (Calcitonin, Urocortin) et analogue synthétique (Cagrilintide)
- 303 contacts résidu–résidu analysés

---

### 12.2 Poche consensus Class A (16 positions, seuil 50 %, mode strict)

Le consensus est calculé sur **12 structures validables** (contacts confirmés par Gemmi ET GPCRdb).

#### Distribution par segment

| Segment | Positions | Fréquence (range) |
|---------|-----------|-------------------|
| ECL2 | 45x50, **45x51**, 45x52 | 75 %–**92 %** |
| TM6 | 6x51, **6x55**, 6x58 | 75 %–**92 %** |
| TM7 | **7x34**, **7x38**, 7x42 | 58 %–**83 %** |
| TM3 | 3x29, 3x32, 3x33, 3x36 | 50 %–67 % |
| TM2 | 2x63 | 58 % |
| TM4 | 4x61 | 67 % |
| TM5 | 5x40 | 50 % |

Les positions **45x51 et 6x55** (92 %) sont les plus fréquentes, suivies de **7x34 et 7x38** (83 %).

#### Core ultra-robuste (analyse leave-one-out)

L'analyse LOO sur 15 structures montre un **overlap constant de 0.4375** quelle que soit la structure retirée — 7 positions sur 16 sont présentes dans 100 % des runs LOO :

> **TM3** : 3x29, 3x32, 3x33, 3x36 — **TM5** : 5x40 — **TM7** : 7x34, 7x38

Ces 7 positions constituent le **pharmacophore minimal** invariant de la poche Class A. Le Jaccard varie de 0.200 à 0.226 (taille LOO : 21–26 positions), indiquant une poche stable mais avec une périphérie variable selon les structures.

#### Interprétation biologique

La topologie ECL2/TM6/TM7 est la signature de la **poche orthostérique canonique** des GPCRs rhodopsin-like. La dominance aromatique des contacts peptidiques (33 % des résidus peptidiques en contact) est cohérente avec les interactions π-stacking et aromatic lock caractéristiques de cette classe. TM3 (cluster 3x29–3x36) forme la paroi "arrière" de la poche et contribue à la sélectivité.

---

### 12.3 Poche consensus Class B1 (23 positions, seuil 50 %, mode strict)

> **Avertissement** : ce consensus repose sur **2 structures validables** (6NIY — Calcitonin,
> 7TS0 — Urocortin). 9MNI est exclu (mode de liaison ECD-only) et 9BUE n'est pas validable
> par GPCRdb. Les résultats doivent être interprétés avec prudence.

#### Distribution par segment

| Segment | Positions | Fréquence |
|---------|-----------|-----------|
| TM1 | 1x27, 1x33, 1x36, 1x39, 1x40 | 1x33, 1x36 à **100 %** |
| TM7 | 7x37, 7x38, 7x41, 7x42, 7x45 | 7x37, 7x42, 7x45 à **100 %** |
| TM5 | 5x37, 5x38, 5x39, 5x40 | 5x40 à **100 %** |
| TM2 | 2x64, 2x67, 2x68 | 2x64 à **100 %** |
| TM3 | 3x36, 3x37, 3x44 | 50 % |
| TM6 | 6x53, 6x56 | 6x53 à **100 %** |
| ECL2 | 45x52 | **100 %** |

**9 positions à 100 %** : 1x33, 1x36, 2x64, 45x52, 5x40, 6x53, 7x37, 7x42, 7x45.

#### Interprétation biologique

La **dominance de TM1** (5 positions) est la signature structurale des GPCRs Class B1. Elle reflète le **mécanisme de liaison en deux domaines** : le N-terminal du peptide s'insère dans le bundle TM (contacts TM1/TM5/TM6/TM7), tandis que l'extrémité C-terminale est capturée par le domaine N-terminal extracellulaire (NTD) du récepteur. Ce mode d'ancrage par TM1 est absent de la poche Class A.

Le profil d'interaction est **amphipatique** (polar 35 % + hydrophobe 35 %), cohérent avec l'insertion d'une hélice amphipatique (face hydrophobe dans TM1/TM5, face polaire vers les boucles).

---

### 12.4 Profils biophysiques comparés

| Propriété | Class A | Class B | 9IQV knottin |
|-----------|---------|---------|-------------|
| **Résidus aromatiques** | **33 %** | 24 % | 26 % |
| **Résidus hydrophobes** | 17 % | **35 %** | 25 % |
| **Résidus polaires** | 16 % | **35 %** | 18 % |
| Résidus positifs | 16 % | 11 % | **18 %** |
| Segments principaux | TM7 > ECL2 > TM6 > TM2 > TM3 | TM1 > N-term > TM7 > ECL2 > TM5 | ECL2 > TM7 > TM2 > TM6 > TM5 |
| Structure secondaire | 11/14 coil | 3/3 hélicoïdal | coil (malgré 65 aa) |
| N contacts total | 841 | 303 | 76 |

La **dominance aromatique en Class A** (33 %) reflète la poche orthostérique hydrophobe-aromatique : les peptides courts en conformation étendue engagent des résidus F, W, Y du récepteur par des interactions π-stacking et van der Waals. Le résidu le plus fréquemment contacté en Class A est en ECL2/TM7 (aromatic lock).

La **double dominance polaire + hydrophobe en Class B** (35 % chacune) traduit le caractère amphipatique des hormones hélicoïdales : une face hydrophobe s'enfouit dans TM1/TM5, une face polaire forme des liaisons H avec les boucles extracellulaires et le NTD.

---

### 12.5 9IQV comme référence microprotéine — preuve de concept

La **Muscarinic toxin 3** (9IQV, knottin 65 aa, venin de serpent) présente sur l'α-1A adrenorécepteur un profil d'interaction **quasi-identique aux peptides Class A coil-dominant** :

- Segments contactés : ECL2 > TM7 > TM2 > TM6 > TM5 (même ordre que la médiane Class A)
- Composition biophysique : aromatique (26 %) + hydrophobe (25 %) + positif (18 %)
- 76 contacts — densité comparable aux peptides courts malgré la taille (65 aa)
- Structure coil dominante malgré les multiples ponts disulfures (rigidité interne)

Ce résultat démontre qu'une **microprotéine à architecture knottin peut occuper la poche orthostérique d'un GPCR Class A** avec les mêmes déterminants biophysiques qu'un peptide de 8–15 aa. C'est la preuve de concept centrale pour généraliser l'analyse à d'autres microprotéines.

---

### 12.6 Tables de référence pour comparaisons futures

La Phase 9 produit trois tables directement exploitables pour comparer une nouvelle microprotéine :

| Fichier | Contenu | Usage |
|---------|---------|-------|
| `out/positional_analysis/receptor_reference_table.tsv` | Contacts par (classe GPCR, groupe DSSP peptide, segment, classe biophysique) | Comparer le profil receptor-centrique global |
| `out/positional_analysis/gpcrdb_positions_reference_table.tsv` | Contacts par position GPCRdb exacte | Identifier les hotspots communs |
| `out/positional_analysis/analysis_summary.txt` | Résumé statistique textuel | Vue d'ensemble rapide |

**Workflow de comparaison :**
1. Détecter les contacts microprotéine–GPCR (même pipeline, cutoff 5 Å)
2. Calculer DSSP de la microprotéine → helix_fraction
3. Joindre avec `receptor_reference_table.tsv` sur `(segment, peptide_class)`
4. Comparer avec `gpcrdb_positions_reference_table.tsv` sur `gpcrdb_pos`
5. Calculer le score de similarité avec le profil 9IQV (aromatique + hydrophobe sur ECL2/TM7)

---

## 13. Corrections et limitations connues

### Corrections apportées aux scripts

| Script | Problème | Correction |
|--------|----------|------------|
| `scripts/extract_peptide_nature_from_cif.py` | Le mapping chaîne → entité cherchait en priorité dans `_struct_asym.label_asym_id` (identifiant interne mmCIF) alors que `peptide_chain` dans `peptide_ligands_gpcr.tsv` est un `auth_asym_id` (identifiant auteur). Résultat : 4 structures (7X1T, 8JBH, 8IBV, 9MNI) remontaient la mauvaise entité (ScFv16, NK3R-pFastbac1, Gq, RAMP1 au lieu de Taltirelin, Substance P, Motiline, dC2_050). | Priorité inversée : `atom_site.auth_asym_id` en premier, `_struct_asym.label_asym_id` en dernier recours. |
| `scripts_validation_consensus/plot_leave_one_out_stability.py` | Le script sauvegardait la figure sous `loo_stability_clean_improved.png` alors que le Snakefile attendait `loo_stability_clean.png`. | Nom de fichier corrigé. |
| `scripts/make_pymol_consensus_superposition_views.py` | La colonne `top_aa` était accédée sans vérification de présence, causant un `KeyError` quand le TSV consensus ne la contient pas. | Fallback ajouté (`top_aa = "X"` si colonne absente). |
| `scripts/make_consensus_pockets_by_class_strict_validable.py` | Pas de mécanisme d'exclusion explicite de structures. | Paramètre `--exclude_pdbs` ajouté. |

### Limitations du consensus Class B

Le consensus Class B (23 positions, seuil 50 %, mode strict) présente deux limitations majeures :

1. **9MNI exclu** : le minibinder de novo dC2_050 contacte quasi-exclusivement le NTD extracellulaire
   (pas le bundle TM). Son mode de liaison n'est pas représentatif des peptides endogènes Class B1.
   Exclu via `--exclude_pdbs 9MNI` dans la règle `consensus_pockets` du Snakefile.

2. **9BUE non validable** : aucune donnée d'interaction GPCRdb disponible pour Cagrilintide/9BUE
   au moment de l'analyse (0 contacts "both", 35 "gemmi_only"). Structure absente du dénominateur
   en mode strict.

**Conséquence** : le consensus Class B repose sur **2 structures seulement** (6NIY — Calcitonin, et
7TS0 — Urocortin). Les 23 positions reflètent l'intersection de ces deux structures. Ce résultat
doit être interprété avec prudence et devra être consolidé lorsque de nouvelles structures
GPCRdb-validées de Class B1 seront disponibles.

---

## Licence et contact

Projet de recherche — Analyse bioinformatique structurale GPCR.  
Pour toute question, se référer aux commentaires dans le Snakefile et les scripts individuels.
