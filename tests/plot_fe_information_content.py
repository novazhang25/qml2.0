#!/usr/bin/env python3
"""Plot saved Run 1 FE spectra and energies; never invoke solvers or training."""
import csv
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/qml-fe-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/audits/fe_all_molecules'
OUT = ROOT / 'results/audits/fe_information_content'
MOLECULES = ('LiH', 'BeH2', 'H2O', 'NH3', 'N2', 'CO', 'HF', 'H2O2')


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def draw(ax, x, y, train, xlabel, ylabel, title):
    ax.plot(x, y, color='#9aa4af', lw=1.2, zorder=1, label='Geometry-order path')
    ax.scatter(x[train], y[train], color='#1764ab', s=42, marker='o', label='Train (15)', zorder=3)
    ax.scatter(x[~train], y[~train], color='#d45a16', s=48, marker='^', label='Test (15)', zorder=3)
    for k in (0, 29):
        ax.annotate(f'{k+1:03d}', (x[k], y[k]), xytext=(5, 7), textcoords='offset points', fontsize=8)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.ticklabel_format(useOffset=False, axis='both')
    ax.grid(alpha=.2)
    ax.margins(.12)
    ax.legend(fontsize=8)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    spectra = read_csv(SOURCE / 'fe_eigenvalues.csv')
    provenance = json.loads((SOURCE / 'provenance.json').read_text())
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in (SOURCE / 'fe_eigenvalues.csv', SOURCE / 'provenance.json')}
    stats, points, assessments = [], [], []
    overview, panels = plt.subplots(9, 2, figsize=(13, 36), layout='constrained')
    panel = 0
    for molecule in MOLECULES:
        rows = [r for r in spectra if r['molecule'] == molecule and r['view'] == 'Run1_exact']
        width = max(int(r['column_index']) for r in rows) + 1
        assert len(rows) == 30 * width
        X = np.full((30, width), np.nan)
        for r in rows:
            i, j = int(r['geometry_id'])-1, int(r['column_index'])
            assert np.isnan(X[i, j]), 'Duplicate spectrum entry'
            X[i, j] = float(r['eigenvalue'])
        assert np.isfinite(X).all()
        ids = np.arange(1, 31)
        train = ids % 2 == 1
        q, energy = [], []
        for i in ids:
            path = ROOT / f'results/descriptor/{molecule}/{i:03d}.npz'
            digest = sha(path)
            assert digest == provenance['input_sha256'][str(path)], f'Source changed since FE audit: {path}'
            hashes[str(path.relative_to(ROOT))] = digest
            with np.load(path, allow_pickle=False) as d:
                q.append(float(d['q_A']))
                energy.append(float(d['E_FCI']) - float(d['E_RHF_input']))
        q, energy = np.array(q), np.array(energy)
        assert np.isfinite(q).all() and np.isfinite(energy).all()
        assert np.all(np.diff(q) > 0)
        std = X[train].std(axis=0, ddof=0)
        np.testing.assert_allclose(std, provenance['actual_Run1_FE_constants'][molecule]['sigma'], rtol=1e-10, atol=1e-16)
        active = np.flatnonzero(std > 1e-12)
        assert len(active) == (2 if molecule == 'H2O2' else 1)
        for j in active:
            x = X[:, j]
            delta = np.diff(x)
            turns = int(np.sum(np.sign(delta[1:]) != np.sign(delta[:-1])))
            # Vertical spread between geometry-order linear segments over overlapping x ranges.
            spread = 0.
            for a in range(29):
                for b in range(a+2, 29):
                    low = max(min(x[a:a+2]), min(x[b:b+2]))
                    high = min(max(x[a:a+2]), max(x[b:b+2]))
                    if high > low:
                        at = np.array([low, high])
                        ya = energy[a] + (at-x[a]) * (energy[a+1]-energy[a]) / delta[a]
                        yb = energy[b] + (at-x[b]) * (energy[b+1]-energy[b]) / delta[b]
                        spread = max(spread, float(np.max(np.abs(ya-yb))))
            stats.append(dict(molecule=molecule, column_index=int(j), min=float(x.min()), max=float(x.max()), span=float(np.ptp(x)), training_std=float(std[j]), pearson_geometry_coordinate=float(np.corrcoef(q, x)[0,1]), pearson_geometry_index=float(np.corrcoef(ids, x)[0,1]), pearson_E_corr=float(np.corrcoef(energy, x)[0,1]), turning_points=turns, interpolated_branch_spread_Ha=spread))
            for k in range(30):
                points.append(dict(molecule=molecule, geometry_id=f'{k+1:03d}', split='train' if train[k] else 'test', scan_coordinate_A=q[k], column_index=int(j), raw_eigenvalue=x[k], E_corr_Ha=energy[k]))
            label = rf'Raw FE eigenvalue $\lambda_{{{j}}}$'
            specs = [(q, x, 'Scan coordinate q (Å)', label, 'active FE vs geometry', 'geometry'), (x, energy, label, r'$E_{FCI}-E_{RHF,input}$ (Ha)', 'correlation energy vs active FE', 'ecorr')]
            for col, (xx, yy, xl, yl, title, suffix) in enumerate(specs):
                title = f'{molecule} · column {j} · {title}'
                fig, ax = plt.subplots(figsize=(7.8, 5.3), layout='constrained')
                draw(ax, xx, yy, train, xl, yl, title)
                for ext in ('png', 'pdf'):
                    fig.savefig(OUT / f'{molecule}_lambda_{j}_{suffix}.{ext}', dpi=180)
                plt.close(fig)
                draw(panels[panel, col], xx, yy, train, xl, yl, title)
            panel += 1
            assessments.append(f'- **{molecule}, column {j}:** {turns} sampled turning point(s); maximum branch separation from geometry-order piecewise-linear interpolation = {spread*1000:.6g} mHa.')
    overview.savefig(OUT / 'overview.png', dpi=100)
    plt.close(overview)
    write_csv(OUT / 'active_eigenvalue_statistics.csv', stats)
    write_csv(OUT / 'plotted_points.csv', points)
    for name, digest in hashes.items():
        assert sha(ROOT / name) == digest
    (OUT / 'provenance.json').write_text(json.dumps(dict(input_sha256=hashes, activity_threshold=1e-12, training_std_ddof=0, split='Odd IDs train, even IDs test', energy='E_FCI - E_RHF_input', source='Existing Run1_exact spectra; source descriptor hashes verified against existing audit', excluded=['H2S']), indent=2)+'\n')
    lines = ['# Run 1 FE information content', '', 'All 30 geometries per molecule; H2S excluded. Raw, unstandardized eigenvalues. Columns are zero-based. Training std uses the 15 odd IDs and ddof=0; test uses the 15 even IDs. Activity: training std > 1e-12. Min/max/span and Pearson correlations use all 30 points. Geometry correlation uses saved q_A in Å (index correlation is also exported). Energy is the saved Run 1 target E_FCI − E_RHF_input in Hartree.', '', 'No RHF, FCI, descriptor generation, or training was run. Existing Run1_exact eigenvalues were read from the prior FE audit; all 240 descriptor hashes match that audit. Lines follow geometry order, not eigenvalue sorting. Endpoint annotations identify geometry IDs.', '', '| Molecule | Column | Min | Max | Span | Training std | r(q, λ) | r(Ecorr, λ) |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in stats:
        lines.append(f'| {r["molecule"]} | {r["column_index"]} | {r["min"]:.10g} | {r["max"]:.10g} | {r["span"]:.10g} | {r["training_std"]:.10g} | {r["pearson_geometry_coordinate"]:.7f} | {r["pearson_E_corr"]:.7f} |')
    lines += ['', '## Geometry folds and single-valuedness', '', 'Visual inspection: all nine active eigenvalue curves appear smooth at the sampled resolution. LiH, BeH2, H2O, N2, CO, HF, and both H2O2 columns are monotonic and support approximately single-valued energy relations over this scan. NH3 column 2 is smooth but nonmonotonic: its energy relation folds into two branches, with up to 26.23 mHa interpolated separation at the same eigenvalue. Smoothness therefore does not imply an informative one-to-one descriptor. These conclusions are restricted to the sampled scan.', '', *assessments, '', 'Branch separation measures the largest vertical energy difference where two nonadjacent geometry-order line segments overlap in eigenvalue. It is a sampled, piecewise-linear diagnostic, not a fitted physical model or a proof between geometries. Zero separation for a monotonic eigenvalue supports a single-valued relation on this scan. Pearson correlation alone does not establish smoothness or single-valuedness.', '', '## Plots', '']
    for r in stats:
        prefix = f'{r["molecule"]}_lambda_{r["column_index"]}'
        lines += [f'### {r["molecule"]}, column {r["column_index"]}', '', f'![Geometry]({prefix}_geometry.png)', '', f'![Correlation energy]({prefix}_ecorr.png)', '']
    (OUT / 'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(stats, indent=2))
    print(f'Saved 18 plots in PNG/PDF, overview, statistics, points and provenance to {OUT}')


if __name__ == '__main__':
    main()
