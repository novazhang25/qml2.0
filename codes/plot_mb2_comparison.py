"""Plot saved MB1/MB2/MB2-prime results; never train or modify source results."""
import argparse
import json
from pathlib import Path

import plot_run1_results as plots
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mb1-dir', type=Path, default=PROJECT / 'results/run1/20260924_115423_466018')
    parser.add_argument('--run2-dir', type=Path, default=PROJECT / 'results/run2/20260925_113307_117058')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--split', choices=['all', 'train', 'test'], default='test')
    parser.add_argument('--font-scale', type=float, default=1.60)
    parser.add_argument('--dpi', type=int, default=400)
    args = parser.parse_args()
    configs = [json.loads((p / 'configuration.json').read_text()) for p in (args.mb1_dir, args.run2_dir)]
    for key in ('molecules', 'seeds', 'epochs', 'batch_size', 'learning_rate', 'target', 'split_protocol', 'checkpoint_selection'):
        if configs[0][key] != configs[1][key]:
            raise ValueError(f'Reference configuration mismatch: {key}')
    if configs[0]['seeds'] != list(range(32)):
        raise ValueError('Expected the complete 32-seed Run 1 benchmark')
    molecules = configs[0]['molecules']
    methods = ['MB-1', 'MB2', 'MB2_prime']
    labels = ['MB1', 'MB2', 'MB2\u2032']
    plots.DISPLAY_RUN_LABELS.update(dict(zip(methods, labels)))
    plots.RUN_COLORS.update({'MB2': plots.COLORS['MB-2'], 'MB2_prime': plots.COLORS['MB-3']})
    records = (plots.discover_seed_runs(args.mb1_dir, molecules, ['MB-1'])
               + plots.discover_seed_runs(args.run2_dir, molecules, ['MB2', 'MB2_prime']))
    plots.validate_seed_runs(records, molecules, methods, 32)
    metrics = plots.load_metric_table(records)
    summary = plots.summarize_metrics(metrics, molecules, methods)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    metrics.to_csv(args.output_dir / 'per_seed_test_metrics.csv', index=False)
    summary.to_csv(args.output_dir / 'summary.csv', index=False)
    plots.setup_matplotlib(args.font_scale)
    for molecule in molecules:
        directory = args.output_dir / molecule
        directory.mkdir()
        plots.plot_metric_bars(metrics, summary, [molecule], methods, 'MAE',
                               directory / 'barplot_MAE.png', args.dpi,
                               run_label_prefix='run2', model_only_legend=True)
        curves = []
        for method, label in zip(methods, labels):
            stats, _ = plots.signed_error_stats(records, molecule, method, args.split)
            stats['se_signed_error_mHa'] = stats.std_signed_error_mHa / stats.n_seeds.pow(0.5)
            curves.append(stats.assign(model=label))
        pd.concat(curves, ignore_index=True).to_csv(
            directory / f'signed_error_{args.split}.csv', index=False)
    plots.plot_signed_error_curves(
        records, molecules, methods, args.output_dir, args.split, args.dpi,
        'Run 2', 'run2', 18.0 * args.font_scale, band='se', model_only_legend=True)
    (args.output_dir / 'sources.json').write_text(json.dumps(dict(
        MB1=str(args.mb1_dir.resolve()), MB2=str(args.run2_dir.resolve()),
        MB2_prime=str(args.run2_dir.resolve()),
        split=args.split, style='codes/plot_run1_results.py', units='mHa', seeds=list(range(32)),
        bar_error='SE = sample SD / sqrt(32)',
        signed_error_band='SE = sample SD (ddof=1) / sqrt(n_seeds) per geometry'), indent=2) + '\n')
    print(summary.to_string(index=False))
    print(f'Saved figures and data to {args.output_dir}')


if __name__ == '__main__':
    main()
