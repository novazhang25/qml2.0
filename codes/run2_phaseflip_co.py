#!/usr/bin/env python3
"""CO-only Run2 MB2 density-pair phase experiment; production is opt-in via CLI.

Reuse this repository's Run2 loader, MB2 circuit and train.py loop. Only the
selected 8x8 density matrix supplied to the pair encoder is rephased. The
original MB1 diagonal inputs and original Methods Lambda pairs are retained.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
from types import MappingProxyType

import numpy as np
import pandas as pd
import pennylane as qml
import torch

import run1
import run2
import train as training_api
from circuit import pair_addresses
from run2_model import PairEnergyModel

PROJECT = run2.PROJECT
OUTPUT_ROOT = PROJECT / 'results/run2_phaseflip_co'
REFERENCE_RUN2 = PROJECT / 'results/run2/20260925_113307_117058'
PHASES = MappingProxyType({
    'MB2':            (+1, +1, +1, +1, +1, +1, +1, +1),
    'MB2_flip_0':     (-1, +1, +1, +1, +1, +1, +1, +1),
    'MB2_flip_01':    (-1, -1, +1, +1, +1, +1, +1, +1),
    'MB2_flip_012':   (-1, -1, -1, +1, +1, +1, +1, +1),
    'MB2_flip_0123':  (-1, -1, -1, -1, +1, +1, +1, +1),
    'MB2_flip_01234': (-1, -1, -1, -1, -1, +1, +1, +1),
})
METHODS = tuple(PHASES)
PAIRS = pair_addresses(8)
PAIR_I, PAIR_J = np.asarray(PAIRS, dtype=int).T
ATOL, RTOL = 1e-12, 1e-10


def phase_sample(sample, phase):
    """Allocate D P D in current qubit order; never alter any other descriptor."""
    d = np.asarray(phase)
    if d.shape != (8,) or not np.isin(d, (-1, 1)).all():
        raise ValueError('A fixed vector of eight signs (+1 or -1) is required')
    if sample.molecule != 'CO' or sample.pij.shape != (8, 8):
        raise ValueError('CO P_valence.shape must equal (8, 8)')
    p_phase = d[:, None] * sample.pij * d[None, :]
    return replace(sample, pij=p_phase)


def ao_mapping(samples, metadata):
    """Read the actual loader mapping and verify it at every CO geometry."""
    mapping = None
    for sample in samples:
        if sample.molecule != 'CO' or sample.pij.shape != (8, 8):
            raise ValueError(f'{sample.sample_id}: P_valence.shape must equal (8, 8)')
        selected = metadata[sample.sample_id]['source_selected_ao_indices']
        with np.load(sample.source_file, allow_pickle=False) as raw:
            np.testing.assert_array_equal(
                sample.pij, raw['P_L'][np.ix_(selected, selected)])
            labels = [' '.join(str(raw['ao_labels'][i]).split()) for i in selected]
        current = dict(
            P_valence_shape=list(sample.pij.shape), index_base=0,
            full_ao_indices=list(selected), ao_labels=list(sample.active_ao_labels),
            source_ao_labels=labels,
            qubits=[dict(qubit=q, full_ao_index=i, ao_label=label)
                    for q, (i, label) in enumerate(zip(selected, labels, strict=True))])
        if mapping is not None and mapping != current:
            raise ValueError(f'{sample.sample_id}: AO order changes across geometries')
        mapping = current
    if mapping is None:
        raise ValueError('CO population is empty')
    return mapping


def prepare_runs(samples, manifest, metadata):
    """Fit MB1 once on original data, then change only the MB2 pair density."""
    prepared, audits = run2.prepare_runs(samples, manifest, metadata, methods=('MB2',))
    baseline, *original_groups = prepared['CO', 'MB2']
    for method in METHODS[1:]:
        model = PairEnergyModel(baseline.constants, method='MB2')
        # Always start from the original Run2 samples, never the previous flip.
        transformed = tuple(tuple(phase_sample(s, PHASES[method]) for s in group)
                            for group in original_groups)
        prepared['CO', method] = (model, *transformed)
    return prepared, audits


def validate_experiment(samples, prepared):
    """Check all geometries, including +/-I fixed-parameter feature identities."""
    expected_names = ('MB2', 'MB2_flip_0', 'MB2_flip_01', 'MB2_flip_012',
                      'MB2_flip_0123', 'MB2_flip_01234')
    if METHODS != expected_names:
        raise AssertionError('Exactly baseline and the five named cumulative variants are required')
    for k, method in enumerate(METHODS):
        expected_phase = np.ones(8, dtype=int)
        expected_phase[:k] = -1
        np.testing.assert_array_equal(PHASES[method], expected_phase)
    baseline, *baseline_groups = prepared['CO', 'MB2']
    training = baseline_groups[0]
    # Nonzero a2 AND b2 make the fixed-parameter checks informative.
    theta = torch.linspace(-.37, .43, baseline.num_parameters, dtype=torch.float64)
    unchanged = ('pii', 'fii', 'fij', 'lambda_pairs', 'active_ao_indices')
    snapshots = {s.sample_id: {k: getattr(s, k).copy() for k in ('pij', *unchanged)}
                 for s in samples}
    cases = dict(PHASES, all_minus_sanity=(-1,) * 8)
    max_symmetry_error = max_eigenvalue_error = max_prediction_error = 0.0
    with torch.no_grad():
        for sample in samples:
            original_args = baseline.pair_quantum_arguments(sample, theta)
            original_features = baseline.raw_readout_features(sample, theta)
            original_prediction = baseline.apply_readout(original_features, theta)
            for method, phase in cases.items():
                changed = phase_sample(sample, phase)
                p, d = changed.pij, np.asarray(phase)
                np.testing.assert_array_equal(p, np.outer(d, d) * sample.pij)
                np.testing.assert_allclose(p.T, p, atol=ATOL, rtol=RTOL)
                np.testing.assert_array_equal(np.diag(p), np.diag(sample.pij))
                eig_error = np.linalg.eigvalsh(p) - np.linalg.eigvalsh(sample.pij)
                np.testing.assert_allclose(np.linalg.eigvalsh(p),
                                           np.linalg.eigvalsh(sample.pij), atol=ATOL, rtol=RTOL)
                np.testing.assert_array_equal(p[PAIR_I, PAIR_J],
                    d[PAIR_I] * d[PAIR_J] * sample.pij[PAIR_I, PAIR_J])
                if np.shares_memory(p, sample.pij):
                    raise AssertionError('Phase density must not alias source density')
                for field in unchanged:
                    np.testing.assert_array_equal(getattr(changed, field), getattr(sample, field))
                args = baseline.pair_quantum_arguments(changed, theta)
                for index in (0, 1, 3, 4, 5, 6):
                    torch.testing.assert_close(args[index], original_args[index], rtol=0, atol=0)
                if method in ('MB2', 'all_minus_sanity'):
                    np.testing.assert_array_equal(p, sample.pij)
                    features = baseline.raw_readout_features(changed, theta)
                    prediction = baseline.apply_readout(features, theta)
                    torch.testing.assert_close(features, original_features, rtol=0, atol=ATOL)
                    torch.testing.assert_close(prediction, original_prediction, rtol=0, atol=ATOL)
                    max_prediction_error = max(max_prediction_error,
                                               abs(float(prediction - original_prediction)))
                max_symmetry_error = max(max_symmetry_error, float(np.max(np.abs(p.T - p))))
                max_eigenvalue_error = max(max_eigenvalue_error, float(np.max(np.abs(eig_error))))
    for method in METHODS:
        model, *groups = prepared['CO', method]
        if model.constants != baseline.constants or model.layout != baseline.layout:
            raise AssertionError(f'{method}: MB1 constants or MB2 parameter layout changed')
        if model.num_parameters != baseline.num_parameters:
            raise AssertionError(f'{method}: parameter count mismatch')
        originals = {s.sample_id: s for s in samples}
        for group, original_group in zip(groups, baseline_groups, strict=True):
            if tuple(s.sample_id for s in group) != tuple(s.sample_id for s in original_group):
                raise AssertionError(f'{method}: split population or ordering changed')
        for s in (s for group in groups for s in group):
            expected = phase_sample(originals[s.sample_id], PHASES[method])
            np.testing.assert_array_equal(s.pij, expected.pij)
            for field in unchanged:
                np.testing.assert_array_equal(getattr(s, field), getattr(expected, field))
            if s.target_energy != expected.target_energy or s.active_ao_labels != expected.active_ao_labels:
                raise AssertionError(f'{method}: target or AO labels changed')
        for seed in run2.SEEDS:
            torch.testing.assert_close(
                training_api.initialize_parameters(model, groups[0], seed=seed),
                training_api.initialize_parameters(baseline, training, seed=seed), rtol=0, atol=0)
    for sample in samples:
        for field, saved in snapshots[sample.sample_id].items():
            np.testing.assert_array_equal(getattr(sample, field), saved)
    return dict(status='PASS', geometries=len(samples), P_valence_shape=[8, 8],
                trained_models=list(METHODS), sanity_only=['all_minus_sanity'],
                parameter_count=baseline.num_parameters, atol=ATOL, rtol=RTOL,
                max_symmetry_error=max_symmetry_error, max_eigenvalue_error=max_eigenvalue_error,
                max_identity_prediction_error_Ha=max_prediction_error,
                exact_original_lambda_and_MB1_inputs=True, source_arrays_unchanged=True,
                cumulative_phase_vectors_verified=True, all_variants_start_from_original_P=True,
                all_32_seed_initializations_matched=True)


def verify_reference(reference, population, prepared, input_dir):
    """Verify the current saved CO benchmark; always train a fresh baseline."""
    reference = Path(reference).resolve()
    config = json.loads((reference / 'configuration.json').read_text())
    expected = dict(molecules=['CO'], seeds=list(run2.SEEDS), epochs=run2.EPOCHS,
                    batch_size=run2.BATCH_SIZE, learning_rate=run2.LEARNING_RATE,
                    optimizer='torch.optim.Adam', input_dir=str(Path(input_dir).resolve()),
                    target=run1.TARGET.formula, split_protocol='15-15',
                    checkpoint_selection='strictly improving train MAE; earliest epoch on ties',
                    pair_axis_convention=run2.AXIS_CONVENTION, pair_extraction=run2.PAIR_EXTRACTION)
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f'Current Run2 reference mismatch for {key}')
    if 'MB2' not in config['methods']:
        raise ValueError('Current Run2 reference must contain MB2')
    if json.loads((reference / 'population.json').read_text()) != population:
        raise ValueError('Current Run2 population/split/target/AO mapping mismatch')
    constants = asdict(prepared['CO', 'MB2'][0].constants)
    saved = json.loads((reference / 'preprocessing.json').read_text())['CO/MB2']
    if json.loads(json.dumps(constants)) != saved:
        raise ValueError('Current Run2 MB1 preprocessing mismatch')
    return config


def batch_schedule(samples, seed, epochs):
    """Materialize the existing independent SampleOrder stream for auditing."""
    order = training_api.SampleOrder(seed)
    return tuple(tuple(tuple(s.sample_id for s in batch) for batch in
                       training_api.iter_batches(samples, run2.BATCH_SIZE,
                                                  order.permutation(len(samples))))
                 for _ in range(epochs))


def train_one(output, method, seed, prepared, metadata, initial_theta, schedule, *, epochs):
    """Run unchanged train.py with observed batch verification and Run1 outputs.

    Each call constructs a fresh initialized vector and a fresh Adam optimizer.
    The checkpoint and final evaluation rules below mirror current Run1/Run2.
    """
    model, training, validation, testing = prepared['CO', method]
    candidate = training_api.initialize_parameters(model, training, seed=seed).detach().numpy()
    np.testing.assert_array_equal(candidate, initial_theta)
    directory = output / 'runs' / 'CO' / method / f'seed_{seed:02d}'
    directory.mkdir(parents=True, exist_ok=False)
    np.save(directory / 'initial_theta.npy', candidate, allow_pickle=False)
    observed = [[] for _ in range(epochs)]

    def progress(epoch, batch, loss, theta, optimizer):
        ids = tuple(s.sample_id for s in batch)
        if ids != schedule[epoch - 1][len(observed[epoch - 1])]:
            raise AssertionError(f'{method}/{seed}: actual minibatch order differs from baseline')
        observed[epoch - 1].append(ids)
        if epoch % 100 == 0 and len(observed[epoch - 1]) == 1:
            print(f'  CO/{method}/{seed}: epoch {epoch}', flush=True)

    print(f'Training CO / {method} / seed {seed}: {epochs} epochs', flush=True)
    result = training_api.train(model, training, validation, seed=seed, num_epochs=epochs,
        batch_size=run2.BATCH_SIZE, learning_rate=run2.LEARNING_RATE,
        checkpoint_selection='train', on_update=progress)
    if tuple(tuple(batches) for batches in observed) != schedule:
        raise AssertionError('Incomplete matched minibatch trajectory')
    run1.write_json(directory / 'batch_order.json', dict(seed=seed, epochs=observed))
    predictions, metrics = [], []
    for split, population in (('train', training), ('test', testing)):
        ev = training_api.evaluate(model, population, theta=result.theta, batch_size=run2.BATCH_SIZE)
        if not np.isfinite(list(ev.metrics.values())).all():
            raise FloatingPointError('Nonfinite final evaluation metrics')
        metrics.append(dict(molecule='CO', model=method, seed=seed, split=split,
            count=len(population), best_epoch=result.checkpoint.epoch, checkpoint_selection='train',
            best_train_mae_ha=result.checkpoint.train_mae_ha,
            final_test_evaluation_call_count=1, **ev.metrics))
        for sample, prediction in zip(population, ev.predictions, strict=True):
            meta = metadata[sample.sample_id]
            predictions.append(dict(molecule='CO', model=method, seed=seed, split=split,
                geometry_id=meta['geometry_id'], q_A=meta['q_A'], bond_length_A=meta['bond_length_A'],
                target_Ha=sample.target_energy, prediction_Ha=float(prediction),
                error_Ha=float(prediction - sample.target_energy)))
    np.save(directory / 'best_theta.npy', result.theta.detach().numpy(), allow_pickle=False)
    pd.DataFrame(predictions).to_csv(directory / 'predictions.csv', index=False)
    pd.DataFrame(metrics).to_csv(directory / 'metrics.csv', index=False)
    pd.DataFrame(result.history).to_csv(directory / 'training_history.csv', index=False)
    run1.write_json(directory / 'parameters.json', dict(model=method, underlying_model='MB2',
        seed=seed, phase_vector=PHASES[method], parameter_count=model.num_parameters,
        parameter_names=model.layout.names,
        best_parameters=dict(zip(model.layout.names, result.theta.tolist(), strict=True))))
    run1.write_json(directory / 'result.json', dict(best_epoch=result.checkpoint.epoch,
        checkpoint_selection='train', best_train_mae_ha=result.checkpoint.train_mae_ha,
        checkpoint_train_mae_ha=result.checkpoint.train_mae_ha, restored_train_metrics=metrics[0],
        final_test_metrics=metrics[1], final_test_evaluation_call_count=1,
        matched_initialization=True, matched_actual_minibatches=True))
    print('  Finished: ' + ', '.join(f"{m['split']} MAE={m['mae_mHa']:.8g}" for m in metrics)
          + ' mHa', flush=True)
    return metrics


def summarize(output, records, seeds):
    """Keep Run2 tables plus paired comparisons; sample SD uses ddof=1."""
    run2.summarize(output, records, ['CO'], splits=('train', 'test'), methods=METHODS, seeds=seeds)
    metrics = {(r['model'], r['seed'], r['split']): r['mae_mHa'] for r in records}
    rows = [dict(model=method, seed=seed, train_MAE=metrics[method, seed, 'train'],
                 test_MAE=metrics[method, seed, 'test'],
                 test_MAE_minus_matched_baseline=(metrics[method, seed, 'test']
                                                 - metrics['MB2', seed, 'test']))
            for method in METHODS for seed in seeds]
    frame = pd.DataFrame(rows)
    frame.to_csv(output / 'comparison.csv', index=False)
    summaries = []
    for method in METHODS:
        part = frame[frame.model == method]
        row = dict(model=method, n_seeds=len(part), energy_unit='mHa')
        for metric in ('train_MAE', 'test_MAE', 'test_MAE_minus_matched_baseline'):
            row[f'{metric}_mean'] = part[metric].mean()
            row[f'{metric}_sample_SD'] = part[metric].std(ddof=1)
        summaries.append(row)
    pd.DataFrame(summaries).to_csv(output / 'comparison_summary.csv', index=False)


def output_path(value=None):
    """Reject existing outputs and resolved escapes, including symlink escapes."""
    root = OUTPUT_ROOT.absolute()
    output = (Path(value) if value is not None else root /
              datetime.now().strftime('%Y%m%d_%H%M%S_%f')).resolve()
    if root.resolve() != root or output == root or not output.is_relative_to(root):
        raise ValueError(f'Output must be a new child directory under {root}')
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}')
    return output


def descriptor_hashes(samples):
    return {str(s.source_file): hashlib.sha256(s.source_file.read_bytes()).hexdigest()
            for s in samples}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=PROJECT / 'results/descriptor')
    parser.add_argument('--reference-run2', type=Path, default=REFERENCE_RUN2,
                        help='Read-only current CO Run2 configuration/population reference.')
    parser.add_argument('--output-dir', type=Path, help='New child of results/run2_phaseflip_co.')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--validate-only', action='store_true')
    modes.add_argument('--smoke-only', action='store_true', help='Exactly one epoch, seed 0, all six models.')
    modes.add_argument('--production', action='store_true', help='Explicitly train all current Run2 seeds/epochs.')
    args = parser.parse_args(argv)
    if not (args.validate_only or args.smoke_only or args.production):
        parser.error('Select --validate-only, --smoke-only, or --production')
    output = output_path(args.output_dir)
    samples, manifest, metadata = run2.load_population(
        args.input_dir, ('CO',), split_protocol='15-15', methods=('MB2',))
    source_hashes = descriptor_hashes(samples)
    mapping = ao_mapping(samples, metadata)
    prepared, audits = prepare_runs(samples, manifest, metadata)
    population = [dict(metadata[r.sample_id], split=r.split, retained_position=r.retained_position)
                  for r in manifest.rows]
    reference_config = verify_reference(args.reference_run2, population, prepared, args.input_dir)
    checks = validate_experiment(samples, prepared)
    seeds = (0,) if args.smoke_only else run2.SEEDS
    epochs = 1 if args.smoke_only else run2.EPOCHS
    config = dict(reference_config)
    config.update(methods=list(METHODS), display_labels={m: m for m in METHODS},
        trained_molecules=[] if args.validate_only else ['CO'], seeds=list(seeds), epochs=epochs,
        requested_epochs=run2.EPOCHS, smoke_only=args.smoke_only, validate_only=args.validate_only,
        input_dir=str(args.input_dir.resolve()), reference_run1=None,
        reference_run2=str(args.reference_run2.resolve()), baseline_reused=False,
        baseline_policy='fresh independently trained MB2 with identical initial vectors and minibatches',
        implementation='local run2_phaseflip_co.py; current run2.py + run2_model.py + run1.py + circuit.py + train.py',
        phase_vectors=dict(PHASES), phase_basis='current 8-AO valence Lowdin basis in unchanged Run2 qubit order',
        phase_construction='each cumulative prefix D is applied directly to original P; never to a previous phase result',
        phase_trainable=False, ao_mapping=mapping, pair_order=PAIRS,
        pair_full_ao_order=[[mapping['full_ao_indices'][i], mapping['full_ao_indices'][j]] for i, j in PAIRS],
        pair_preprocessing='P_phase = d[:,None]*P_valence*d[None,:]; original Lambda pairs unchanged; no new preprocessing',
        pair_encoders={m: '(pi/2)*tanh(a2*P_phase[i,j]+b2*Lambda_pair_original[i,j])' for m in METHODS},
        initialization='unchanged current MB2 initialize_parameters(model, train_samples, seed); fresh Adam per model',
        minibatch_order='unchanged train.SampleOrder(seed); all actual batches verified against saved common schedule',
        comparison_energy_unit='mHa', predictions_energy_unit='Hartree', summary_SD='sample SD (ddof=1); blank for one seed',
        sanity_only_phase_vectors={'all_minus': [-1] * 8},
        runtime=dict(python=platform.python_version(), numpy=np.__version__,
                     torch=torch.__version__, pennylane=qml.__version__, pandas=pd.__version__))
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (
        ('configuration', config), ('population', population), ('run2_preparation_audit', audits),
        ('validation', checks), ('source_descriptor_sha256', source_hashes),
        ('split_manifest', {'CO': {split: [metadata[r.sample_id]['geometry_id'] for r in manifest.rows
                                          if r.split == split] for split in ('train', 'test')}}),
        ('preprocessing', {f'CO/{m}': asdict(prepared['CO', m][0].constants) for m in METHODS}),
        ('parameter_counts', {m: prepared['CO', m][0].num_parameters for m in METHODS})):
        run1.write_json(output / f'{name}.json', value)
    for method in METHODS:
        directory = output / 'runs' / 'CO' / method
        directory.mkdir(parents=True)
        run1.write_json(directory / 'configuration.json', dict(config, model=method, phase_vector=PHASES[method]))
    print(f'CO phase-flip preparation passed: P_valence.shape=(8, 8). Output: {output}', flush=True)
    if args.validate_only:
        return output
    metrics = []
    for seed in seeds:
        baseline, training, _, _ = prepared['CO', 'MB2']
        initial = training_api.initialize_parameters(baseline, training, seed=seed).detach().numpy()
        schedule = batch_schedule(training, seed, epochs)
        protocol_dir = output / 'matched_protocol' / f'seed_{seed:02d}'
        protocol_dir.mkdir(parents=True)
        np.save(protocol_dir / 'initial_theta.npy', initial, allow_pickle=False)
        run1.write_json(protocol_dir / 'batch_order.json', dict(seed=seed, epochs=schedule))
        for method in METHODS:
            metrics.extend(train_one(output, method, seed, prepared, metadata, initial, schedule, epochs=epochs))
    summarize(output, metrics, seeds)
    if descriptor_hashes(samples) != source_hashes:
        raise AssertionError('Source descriptor files changed during the experiment')
    run1.write_json(output / ('smoke_result.json' if args.smoke_only else 'completion.json'), dict(
        status='PASS', methods=METHODS, seeds=seeds, epochs=epochs,
        matched_initializations=True, matched_actual_minibatches=True, source_files_unchanged=True,
        metrics=metrics))
    print(f'CO phase-flip experiment complete. Results: {output}', flush=True)
    return output


if __name__ == '__main__':
    main()
