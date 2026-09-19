from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA = Path('data/atlas_cleaned.csv')
RESULTS = Path('results')
FIGURES = Path('figures')
ENDPOINTS = ['transfection_pctile', 'size_mean', 'pdi_mean', 'ee_mean']
ORDER = ['Potent + Developable', 'Potent only', 'Developable only', 'Neither']

RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

df = pd.read_csv(DATA)
required = ['lnp_id', 'paper_doi', 'paper_title'] + ENDPOINTS
missing = [c for c in required if c not in df.columns]
if missing:
    raise SystemExit(f'STOP: required columns missing: {missing}')

# Endpoint availability across all formulations.
completeness = pd.DataFrame({
    'endpoint': ENDPOINTS,
    'available_n': [df[c].notna().sum() for c in ENDPOINTS],
    'missing_n': [df[c].isna().sum() for c in ENDPOINTS],
    'available_pct': [100 * df[c].notna().mean() for c in ENDPOINTS],
})
completeness.to_csv(RESULTS / 'phase_e_endpoint_completeness.csv', index=False)

# Four-objective analysis uses only formulations with all four endpoints.
complete = df.dropna(subset=ENDPOINTS).copy()
complete['endpoint_range_flag'] = (
    ~complete['transfection_pctile'].between(0, 1)
    | (complete['size_mean'] <= 0)
    | (complete['pdi_mean'] < 0)
    | ~complete['ee_mean'].between(0, 100)
)

# Primary prespecified thresholds.
complete['potent_primary'] = complete['transfection_pctile'] > 0.50
complete['size_pass_primary'] = complete['size_mean'].between(60, 150)
complete['pdi_pass_primary'] = complete['pdi_mean'] < 0.30
complete['ee_pass_primary'] = complete['ee_mean'] > 80
complete['developable_primary'] = (
    complete['size_pass_primary']
    & complete['pdi_pass_primary']
    & complete['ee_pass_primary']
)

potent = complete['potent_primary']
developable = complete['developable_primary']
complete['category_primary'] = np.select(
    [potent & developable, potent & ~developable,
     ~potent & developable, ~potent & ~developable],
    ORDER,
    default='Unclassified',
)
complete.to_csv(RESULTS / 'phase_e_primary_classification.csv', index=False)

counts = complete['category_primary'].value_counts().reindex(ORDER, fill_value=0)
summary = pd.DataFrame({
    'category': ORDER,
    'n': [int(counts[x]) for x in ORDER],
    'pct_of_complete_cases': [100 * counts[x] / len(complete) for x in ORDER],
})
summary.to_csv(RESULTS / 'phase_e_primary_summary.csv', index=False)

# Threshold sensitivity analysis.
rows = []
for potency_cutoff in [0.50, 0.75]:
    for pdi_cutoff in [0.20, 0.30]:
        for ee_cutoff in [80, 90]:
            p = complete['transfection_pctile'] > potency_cutoff
            d = (complete['size_mean'].between(60, 150)
                 & (complete['pdi_mean'] < pdi_cutoff)
                 & (complete['ee_mean'] > ee_cutoff))
            n_potent = int(p.sum())
            n_fail = int((p & ~d).sum())
            rows.append({
                'potency_cutoff': potency_cutoff,
                'pdi_cutoff': pdi_cutoff,
                'ee_cutoff_pct': ee_cutoff,
                'n_complete_cases': len(complete),
                'n_potent': n_potent,
                'n_potent_and_developable': int((p & d).sum()),
                'n_potent_failing_physchem': n_fail,
                'pct_potent_failing_physchem': 100 * n_fail / n_potent,
            })
pd.DataFrame(rows).to_csv(RESULTS / 'phase_e_threshold_sensitivity.csv', index=False)

study_counts = (complete.groupby(['paper_doi', 'paper_title'], dropna=False)
                .size().reset_index(name='n_complete_formulations')
                .sort_values('n_complete_formulations', ascending=False))
study_counts.to_csv(RESULTS / 'phase_e_complete_cases_by_study.csv', index=False)

# One simple abstract-ready figure.
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(summary['category'], summary['n'])
ax.set_ylabel('Number of formulations')
ax.set_title('Potency and physicochemical developability')
ax.tick_params(axis='x', rotation=20)
for bar, n in zip(bars, summary['n']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.7,
            str(int(n)), ha='center')
fig.tight_layout()
fig.savefig(FIGURES / 'phase_e_primary_classification.png', dpi=300)
plt.close(fig)

n_potent = int(potent.sum())
n_fail = int((potent & ~developable).sum())
fail_pct = 100 * n_fail / n_potent
report = f'''PHASE E CORE ENDPOINT ANALYSIS
================================
Atlas formulations loaded: {len(df)}
Complete four-endpoint formulations: {len(complete)}
Studies represented: {complete['paper_doi'].nunique()}
Endpoint range flags: {int(complete['endpoint_range_flag'].sum())}

Primary thresholds:
  transfection percentile > 0.50
  size 60-150 nm
  PDI < 0.30
  encapsulation efficiency > 80%

Category counts:
{summary.to_string(index=False)}

Potent formulations: {n_potent}
Potent formulations failing at least one physicochemical criterion: {n_fail}
Failure percentage among potent formulations: {fail_pct:.1f}%
'''
(RESULTS / 'phase_e_core_report.txt').write_text(report)
print(report)
print('Saved Phase E CSV files, report, and figure successfully.')
