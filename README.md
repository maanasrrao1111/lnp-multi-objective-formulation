# LNP Multi-Objective Formulation Project — Rockfish Setup

## Status
Phase A (data acquisition) and Phase B (cleaning) are done — those outputs are
your three input files. Phase C (feature engineering) is written and tested
below. Phases D–G still need to be written; do those in follow-up sessions
once you've confirmed Phase C runs cleanly on Rockfish.

## Directory structure
```
lnp_project/
  environment.yml                      conda env spec
  data/
    atlas_cleaned.csv                  464 x 41, from Phase B
    lnpdb_cleaned.csv                  19,797 x 67, held out for potency validation
    external_validation_cleaned.csv    45 x 20, developability validation set
  scripts/
    phase_c_feature_engineering.py     compositional + RDKit descriptors
  slurm/
    submit_phase_c.slurm               SLURM batch script
  results/                             script outputs land here
  figures/                             plots land here
  logs/                                SLURM stdout/stderr (created on first run)
```

## One-time setup on Rockfish

```bash
# 1. Log in
ssh <YourJHED>@login.rockfish.jhu.edu

# 2. Make a project directory in your scratch space (NOT home — scratch has
#    much more quota; ask your PI which scratch allocation to use, e.g.
#    /scratch4/<lab>/<you>/ or /scratch16/<you>/)
mkdir -p /scratch4/<lab_or_your_dir>/lnp_project
cd /scratch4/<lab_or_your_dir>/lnp_project

# 3. Transfer the project folder from your laptop (run this from your LOCAL
#    machine, not on Rockfish):
scp -r lnp_project/ <YourJHED>@login.rockfish.jhu.edu:/scratch4/<lab_or_your_dir>/

# 4. Back on Rockfish: build the conda environment
module load anaconda
conda env create -f environment.yml
conda activate lnp-project

# 5. Sanity check RDKit installed correctly
python -c "from rdkit import Chem; print(Chem.MolFromSmiles('CCO'))"
# should print something like <rdkit.Chem.rdchem.Mol object at 0x...>
```

## Running Phase C

Edit `slurm/submit_phase_c.slurm` first:
- Replace `YOUR_JHED@jhu.edu` with your actual email
- Confirm `--partition=defq` is right for your allocation (check with your PI
  or `sinfo` — `defq` and `shared` are the common general-use partitions, but
  labs with dedicated allocations may have a different partition name)

Then, from the project root on Rockfish:
```bash
sbatch slurm/submit_phase_c.slurm
```

Check status:
```bash
squeue -u <YourJHED>
```

When it finishes (should take well under 5 minutes — this is a small job,
464 rows), check the log and output:
```bash
cat logs/phase_c_*.out
head results/atlas_features.csv
```

Expected output (confirmed by test run before handoff):
```
Loading data/atlas_cleaned.csv ...
  464 rows loaded
  Parsed molar ratio for 462/464 rows
  Computing descriptors for 175 unique ionizable lipid SMILES ...
  WARNING: 1 SMILES failed descriptor computation -> logged to results/phase_c_failed_smiles.csv
  390/464 rows have full molecular descriptors
Saved -> results/atlas_features.csv  (464 x 57)
```

## Important notes carried over from the pipeline design

- **Don't run this on the login node.** Rockfish policy prohibits
  compute-heavy work directly on login nodes — always go through `sbatch` or
  `interact`, even though this particular script is light enough it'd
  probably go unnoticed. Build the habit now for Phase D/F, which won't be
  light.
- **pKa is not computed** in Phase C — flagged as NaN in the descriptor
  columns. If you want it, that needs either a literature lookup per lipid
  or a trained pKa predictor; don't let a future session quietly fabricate
  estimates for this.
- **74/464 rows lack molecular descriptors** (missing or unparseable SMILES).
  These are kept in the dataset with NaN, not dropped. Decide explicitly in
  Phase D whether models drop these rows or you go back and manually resolve
  the missing SMILES (worth doing if any of the 74 turn out to be
  high-transfection or high-encapsulation formulations you don't want to
  lose from the SHAP analysis).
- **`ratio_type`/`ratio_value`** (N:P vs weight ratio from Phase B) are
  synthesis-recipe parameters, not physchem features — keep them out of the
  developability classification in Phase E, per the earlier scoping
  decision.

## What's next (Phase D onward — not yet written)

1. **Phase D**: single-objective SHAP models (transfection, size, PDI, EE)
   using `results/atlas_features.csv` as input. This is the next script to
   write.
2. **Phase E**: developability classification + Pareto frontier — the core
   novel-contribution analysis.
3. **Phase F**: leave-one-study-out CV within Atlas (developability
   generalization) + LNPDB potency-only external check + the 45-formulation
   external developability check.
4. **Phase G**: figures + abstract.

Bring `results/atlas_features.csv` back to a Claude session (or continue
locally) when you're ready for Phase D — that keeps the expensive iterative
/ exploratory work on your own compute, and uses Claude for the parts that
actually need judgment calls (model choice, threshold decisions, interpreting
SHAP output biologically) rather than burning tokens on execution.
