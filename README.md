# Beyond Potency: Multi-Objective Analysis of Reporter-mRNA Lipid Nanoparticle Formulations

**Accepted at Next Generation Cancer Therapeutics 2026 (MD Anderson Cancer Center, October 6–8)**

## Summary

LNP formulation screens rank candidates by transfection potency, but potent formulations frequently fail on particle size, dispersity, or RNA encapsulation. This project systematically quantifies that gap using 464 reporter-mRNA formulations from 62 published studies.

### Key findings

- Among 105 formulations with complete measurements, **65% of potent formulations failed at least one physicochemical criterion** (44.7% after excluding the largest contributing study)
- **Encapsulation efficiency** accounted for 35 of 39 failures — the dominant bottleneck
- Potent+Developable formulations had **structurally simpler ionizable lipids** (fewer rings, fewer tertiary amines; p < 0.001, Bonferroni-corrected)
- Particle size was the only endpoint with **cross-publication predictive signal** (27% MAE reduction under leave-one-publication-out validation across 28 studies)

## Data

- **Primary:** [LNP Atlas](https://doi.org/10.5281/zenodo.17243732) — 1,092 formulations (Nature Scientific Data, 2025)
- **External validation:** 45 formulations manually curated from 4 independent papers
- **Potency validation:** [LNPDB](https://doi.org/10.1038/s41467-026-68818-1) — 19,528 formulations (Nature Communications, 2026)

## Pipeline

| Phase | Description | Status |
|-------|-------------|--------|
| A | Data acquisition and profiling | Complete |
| B | Cleaning, filtering, within-study normalization | Complete |
| C | Publication-specific ratio parsing + RDKit descriptors | Complete (411/464 high-confidence) |
| D | Single-objective SHAP models (4 endpoints) | Complete (only particle size passed) |
| E | Multi-objective classification, Pareto, structural comparison | Complete |
| F | Leave-one-publication-out validation | Complete for particle size |
| G | Abstract and figures | Submitted |

## Reproducibility

Built on JHU Rockfish HPC. Environment:



cd /scratch4/ladamo2/aisirihuv/neoantigen_state_robustness
cat > README.md << 'EOF'
# Cellular-State Escape Risk Among Clinically Prioritized Neoantigens in IDH-Wild-Type Glioblastoma

## Summary

Neoantigen vaccine design typically prioritizes MHC binding affinity and clonal prevalence but overlooks whether targeted antigens are expressed in transcriptionally stable or plastic tumor cell states. This project introduces the **State Robustness Index (SRI)** — a metric quantifying how consistently a neoantigen's source gene is expressed across GBM cellular states — and applies it to clinically selected vaccine candidates from three trials.

## Key datasets

- **Primary atlas:** Neftel et al. scRNA-seq (GSE131928) — IDH-wt GBM cellular states
- **Validation atlas:** Wang et al. scRNA-seq (GSE182109) — recurrent and newly diagnosed GBM
- **Clinical trials:** GNOS-PV01 (Johanns et al., *Nature Cancer* 2026), NeoVax (Keskin et al., *Nature* 2019), ZSNeo-DC (Zhang et al., *Nature Communications* 2026)

## Results

All result tables are in `results/`. Key files:

| File | Description |
|------|-------------|
| `state_robustness_index_full.csv` | SRI scores for 403 genes (Neftel atlas) |
| `wang_state_robustness_index.csv` | SRI scores computed on the Wang validation atlas |
| `sri_concordance.csv` | Cross-atlas SRI concordance |
| `gnos_pv01_final_candidates.csv` | GNOS-PV01 vaccine targets with SRI annotations |
| `immunogenicity_sri_merged.csv` | SRI vs immunogenicity analysis |
| `neovax_sri_results_full.csv` | NeoVax trial SRI analysis |
| `zsneo_sri_results_full.csv` | ZSNeo-DC trial SRI analysis |
| `ranking_comparison.csv` | How SRI re-ranks vaccine candidates vs standard prioritization |

## Data access

Raw scRNA-seq data (too large for GitHub) can be downloaded from GEO:
- Neftel et al.: [GSE131928](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE131928)
- Wang et al.: [GSE182109](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE182109)

## Note on analysis scripts

Analyses were conducted interactively through Claude Science and ChatGPT sessions on JHU Rockfish HPC. Standalone reproducible scripts are being reconstructed and will be added.

## Author

Maanas Rao — MS Biotechnology, Johns Hopkins University
