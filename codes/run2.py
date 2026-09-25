#!/usr/bin/env python3
"""Run 2: original MB1 plus one trainable ZZ pair layer (MB2 and MB2_prime).

Load the current Run1 population and fit its unchanged MB1 preprocessing.
Use the current train.py workflow and Run1 result writer; never generate data.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import pennylane as qml
import torch

import run1
from circuit import EncodingConstants, MOLECULES, pair_addresses
from run2_model import DISPLAY_LABELS, METHODS, PairEnergyModel, PairSample
from train import compute_statistics, initialize_parameters

PROJECT = run1.PROJECT
SEEDS = run1.SEEDS
EPOCHS, BATCH_SIZE, LEARNING_RATE = run1.EPOCHS, run1.BATCH_SIZE, run1.LEARNING_RATE
AXIS_CONVENTION = 'G[p,q,r,s] = Lambda_Methods[p,r,q,s]'
PAIR_EXTRACTION = 'Lambda_Methods[mu,nu,mu,nu] = Lambda_L[source_mu,source_mu,source_nu,source_nu]'
RDM_CONVENTIONS = (
    'Gamma[p,q,r,s]=sum_spin <p^dagger r^dagger s q>; P[p,q]=sum_spin <q^dagger p>',
    'Gamma[p,q,r,s] = sum_spin <p^dagger r^dagger s q>; real spatial MOs',
)
FORMAL_CUMULANTS = (
    'Gamma_MP2_AO_consistent-Gamma0[P_HF]',
    'Lambda_HFref_AO=Gamma_MP2_AO_consistent-Gamma0_HF_AO',
)


def extract_lambda_pairs(raw, selected, path):
    """Verify the saved Methods convention, then address the saved rank-four G.

    These are read-only consistency checks, not a new cumulant construction.
    No T descriptor, correlated-density subtraction, centering or scale is used.
    Run1 has already verified the full-AO Lowdin transformation and selection.
    """
    def scalar(key):
        run1.require(key in raw and np.asarray(raw[key]).shape == (), path, key,
                     'required convention scalar is missing')
        return np.asarray(raw[key]).item()

    convention = scalar('rdm_convention')
    run1.require(convention in RDM_CONVENTIONS, path, 'rdm_convention',
                 'unverified rank-four axis convention')
    metadata = json.loads(scalar('rdm_metadata_json'))
    run1.require(metadata.get('formal_cumulant') in FORMAL_CUMULANTS, path,
                 'formal_cumulant', 'saved Methods HF-reference cumulant required')
    for key in ('index_ordering_AO', 'index_ordering_MO'):
        if key in metadata:
            run1.require(metadata[key] in RDM_CONVENTIONS, path, key,
                         'unverified rank-four axis convention')
    p = raw['P_AO']
    # The disconnected RHF product independently verifies the raw pqrs axes.
    gamma0 = np.einsum('pq,rs->pqrs', p, p) - .5 * np.einsum('ps,rq->pqrs', p, p)
    run1.check_record_array(raw['Gamma0_AO'], gamma0, path, 'raw-axis Gamma0[P_HF]')
    run1.check_record_array(raw['Lambda_AO'],
                           raw['Gamma_MP2_AO_consistent'] - raw['Gamma0_AO'],
                           path, 'saved Methods Lambda_AO subtraction')
    selected = np.asarray(selected, dtype=int)
    pairs = np.asarray(pair_addresses(len(selected)), dtype=int)
    i, j = selected[pairs[:, 0]], selected[pairs[:, 1]]
    values = raw['Lambda_L'][i, i, j, j]
    run1.require(np.isrealobj(values) and np.isfinite(values).all(), path,
                 'Lambda_Methods_abab', 'finite real pair descriptors required')
    return values.copy()


def load_population(input_dir, molecules=tuple(MOLECULES), *, split_manifest=None,
                    split_protocol='manifest', methods=METHODS):
    """Reuse Run1 validation, target, AO selection, geometry IDs and exact split."""
    if not methods or any(method not in METHODS for method in methods):
        raise ValueError('Select at least one supported Run2 method')
    samples, manifest, metadata = run1.load_population(
        input_dir, molecules, split_manifest=split_manifest, split_protocol=split_protocol)
    enriched = []
    for sample in samples:
        values = None
        if 'MB2' in methods:
            with np.load(sample.source_file, allow_pickle=False) as saved:
                values = extract_lambda_pairs(
                    saved, metadata[sample.sample_id]['source_selected_ao_indices'], sample.source_file)
        enriched.append(PairSample(**vars(sample), lambda_pairs=values))
    return tuple(enriched), manifest, metadata


def prepare_runs(samples, manifest, metadata, *, methods=METHODS):
    """Reuse MB1 preparation exactly, including its train-only fitted constants."""
    methods = tuple(dict.fromkeys(methods))
    if not methods or any(method not in METHODS for method in methods):
        raise ValueError('Select at least one supported Run2 method')
    baseline, audits = run1.prepare_runs(samples, manifest, metadata, methods=('MB-1',))
    prepared = {}
    for (molecule, _), (mb1, training, validation, testing) in baseline.items():
        for method in methods:
            model = PairEnergyModel(mb1.constants, method=method)
            populations = (training, validation, testing)
            if method == 'MB2_prime':
                # Even a joint experiment keeps cumulants out of prime samples.
                populations = tuple(tuple(replace(s, lambda_pairs=None) for s in group)
                                    for group in populations)
            else:
                for group in populations:
                    for sample in group:
                        if sample.lambda_pairs is None:
                            raise ValueError(f'{sample.sample_id}: MB2 requires saved Methods pair cumulants')
            expected = mb1.num_parameters + (2 if method == 'MB2' else 1)
            if model.num_parameters != expected:
                raise ValueError(f'{molecule}/{method}: pair parameter count mismatch')
            prepared[molecule, method] = (model, *populations)
    for audit in audits:
        molecule = audit['molecule']
        audit['methods'] = list(methods)
        audit['pair_order'] = [list(pair) for pair in pair_addresses(audit['MB1_qubits'])]
        audit['pair_preprocessing'] = 'raw entries; no centering or normalization; MB1 alpha unused'
        audit['pair_axis_convention'] = AXIS_CONVENTION
        audit['pair_extraction'] = PAIR_EXTRACTION
        audit['parameter_counts'] = {
            method: prepared[molecule, method][0].num_parameters for method in methods}
    return prepared, audits


def match_run1_reference(root, prepared, population, molecules, epochs, input_dir, split_protocol):
    """Check an explicitly selected current Run1 benchmark before creating output."""
    root = Path(root).resolve()
    config = json.loads((root / 'configuration.json').read_text())
    criterion = 'train' if split_protocol == '15-15' else 'validation'
    expected = dict(molecules=list(molecules), seeds=list(SEEDS), epochs=epochs,
                    batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
                    input_dir=str(Path(input_dir).resolve()), target=run1.TARGET.formula,
                    split_protocol=split_protocol,
                    implementation='local codes/run1.py + circuit.py + train.py',
                    full_AO_until_final_descriptor_selection=True,
                    checkpoint_selection=f'strictly improving {criterion} MAE; earliest epoch on ties')
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f'Run1 reference mismatch for {key}: {config.get(key)!r} != {value!r}')
    if 'MB-1' not in config.get('methods', []):
        raise ValueError('Run1 reference must include original MB-1')
    if json.loads((root / 'population.json').read_text()) != population:
        raise ValueError('Run1 reference population, split, AO mapping or target mismatch')
    scalers = json.loads((root / 'preprocessing.json').read_text())
    for (molecule, _), (model, *_) in prepared.items():
        if json.loads(json.dumps(asdict(model.constants))) != scalers.get(f'{molecule}/MB-1'):
            raise ValueError(f'{molecule}: Run1 MB1 preprocessing mismatch')
    metrics = pd.read_csv(root / 'per_seed_metrics.csv')
    for molecule in molecules:
        for split in run1.SPLIT_PROTOCOLS[split_protocol]:
            seeds = sorted(metrics[(metrics.molecule == molecule) & (metrics.model == 'MB-1') &
                                   (metrics.split == split)].seed.tolist())
            if seeds != list(SEEDS):
                raise ValueError(f'{molecule}/{split}: incomplete Run1 reference seeds')
    return str(root)


def train_one(output, molecule, method, seed, prepared, metadata, *, num_epochs=EPOCHS,
              checkpoint_selection='validation'):
    """Use the unmodified Run1 writer and train.py loop; save named pair weights."""
    records = run1.train_one(output, molecule, method, seed, prepared, metadata,
                            num_epochs=num_epochs, checkpoint_selection=checkpoint_selection)
    model, training, _, _ = prepared[molecule, method]
    directory = output / 'runs' / molecule / method / f'seed_{seed:02d}'
    theta = model.load_theta(directory / 'best_theta.npy')
    np.save(directory / 'initial_theta.npy',
            initialize_parameters(model, training, seed=seed).detach().numpy(), allow_pickle=False)
    run1.write_json(directory / 'parameters.json', dict(
        model=method, display_label=DISPLAY_LABELS[method], seed=seed,
        parameter_count=model.num_parameters, parameter_names=model.layout.names,
        best_parameters=dict(zip(model.layout.names, theta.tolist(), strict=True))))
    return records


def summarize(output, records, molecules, *, splits, methods=METHODS, seeds=SEEDS):
    """Write Run1's exact MAE tables, also allowing a labeled one-seed smoke run."""
    if tuple(seeds) == tuple(SEEDS):
        return run1.summarize(output, records, molecules, splits=splits, methods=methods)
    frame = pd.DataFrame(records)
    expected = {(m, method, seed, split) for m in molecules for method in methods
                for seed in seeds for split in splits}
    actual = list(zip(frame.molecule, frame.model, frame.seed, frame.split))
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('Summary requires complete and unique requested run coverage')
    summaries, overall, overall_runs = [], [], []
    for split in splits:
        part = frame[frame.split == split].rename(
            columns={'model': 'model_internal', 'mae_mHa': 'test_MAE_mHa'})
        summaries.append(compute_statistics(part).reset_index().rename(
            columns={'model_internal': 'model'}).assign(split=split))
        per_seed = part.groupby(['model_internal', 'seed'], as_index=False).test_MAE_mHa.mean()
        per_seed['molecule'] = 'ALL'
        overall_runs.append(per_seed.rename(columns={
            'model_internal': 'model', 'test_MAE_mHa': 'mae_mHa'}).assign(split=split))
        overall.append(compute_statistics(per_seed).reset_index().rename(
            columns={'model_internal': 'model'}).assign(split=split))
    frame.to_csv(output / 'per_seed_metrics.csv', index=False)
    pd.concat(summaries, ignore_index=True).to_csv(output / 'per_molecule_summary.csv', index=False)
    pd.concat(overall_runs, ignore_index=True).to_csv(output / 'overall_per_seed.csv', index=False)
    pd.concat(overall, ignore_index=True).to_csv(output / 'overall_summary.csv', index=False)


def load_prediction_run(output, molecule, method, seed):
    """Load saved scalers and all best weights, without fitting or training again.

    Returns (model, theta, split-to-samples). Uses the saved membership lists and
    verifies them against population.json before evaluating any predictions.
    """
    output = Path(output)
    config = json.loads((output / 'configuration.json').read_text())
    if molecule not in config['molecules'] or method not in config['methods'] or seed not in config['seeds']:
        raise ValueError('Requested molecule/model/seed is absent from saved configuration')
    protocol = config['split_protocol']
    saved_splits = run1.read_split_manifest(output / 'split_manifest.json')
    samples, _, metadata = load_population(
        config['input_dir'], [molecule], split_protocol=protocol, methods=(method,),
        split_manifest=output / 'split_manifest.json' if protocol == 'manifest' else None)
    manifest = run1.population_manifest(samples, {molecule: saved_splits[molecule]},
                                        source=output / 'split_manifest.json', split_protocol=protocol)
    actual = [dict(metadata[r.sample_id], split=r.split, retained_position=r.retained_position)
              for r in manifest.rows]
    expected = [row for row in json.loads((output / 'population.json').read_text())
                if row['molecule'] == molecule]
    if actual != expected:
        raise ValueError('Prediction population differs from saved geometry/target/split metadata')
    constants = EncodingConstants(**json.loads((output / 'preprocessing.json').read_text())[f'{molecule}/{method}'])
    model = PairEnergyModel(constants, method=method)
    by_id = {s.sample_id: s for s in samples}
    rows = sorted(manifest.rows, key=lambda row: row.retained_position)
    groups = {split: tuple(by_id[r.sample_id] for r in rows if r.split == split)
              for split in run1.SPLIT_PROTOCOLS[protocol]}
    if constants.training_sample_ids != tuple(s.sample_id for s in groups['train']):
        raise ValueError('Saved scaler training IDs disagree with saved split')
    directory = output / 'runs' / molecule / method / f'seed_{seed:02d}'
    parameters = json.loads((directory / 'parameters.json').read_text())
    if (parameters['model'] != method or parameters['seed'] != seed or
            parameters['parameter_names'] != list(model.layout.names) or
            parameters['parameter_count'] != model.num_parameters):
        raise ValueError('Saved parameter identity/layout mismatch')
    return model, model.load_theta(directory / 'best_theta.npy'), groups


def main(argv=None):
    """Audit current data, run a bounded smoke or the full matched seed sweep."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=PROJECT / 'results/descriptor')
    parser.add_argument('--molecules', nargs='+', choices=tuple(MOLECULES), default=list(MOLECULES))
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    parser.add_argument('--split-protocol', choices=tuple(run1.SPLIT_PROTOCOLS), default='manifest')
    parser.add_argument('--split-manifest', type=Path)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--output-dir', type=Path, help='New directory under results/run2; existing paths are rejected.')
    parser.add_argument('--match-run1', type=Path,
                        help='Verify matching Run1 configuration, population, MB1 scalers and seeds before training.')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--validate-only', action='store_true')
    modes.add_argument('--smoke-only', action='store_true', help='At most two epochs, seed 0, one selected molecule.')
    args = parser.parse_args(argv)
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    if args.split_protocol == 'manifest' and args.split_manifest is None:
        parser.error('--split-manifest is required for manifest mode; use --split-protocol 15-15 to explicitly select the current two-way Run1 protocol.')
    if args.split_protocol == '15-15' and args.split_manifest is not None:
        parser.error('--split-manifest cannot be combined with --split-protocol 15-15')
    methods = tuple(dict.fromkeys(args.methods))
    epochs = min(args.epochs, 2) if args.smoke_only else args.epochs
    output = (args.output_dir or PROJECT / 'results/run2' /
              datetime.now().strftime('%Y%m%d_%H%M%S_%f')).resolve()
    if not output.is_relative_to(PROJECT / 'results/run2'):
        parser.error('--output-dir must be under the current repository results/run2 directory')
    if output.exists():
        raise FileExistsError(f'Output already exists; choose a new directory: {output}')
    samples, manifest, metadata = load_population(
        args.input_dir.resolve(), args.molecules, split_manifest=args.split_manifest,
        split_protocol=args.split_protocol, methods=methods)
    prepared, audits = prepare_runs(samples, manifest, metadata, methods=methods)
    molecules = [a['molecule'] for a in audits]
    smoke_molecule = 'H2O' if 'H2O' in molecules else molecules[0]
    run_molecules = [smoke_molecule] if args.smoke_only else molecules
    seeds = (0,) if args.smoke_only else SEEDS
    selection = 'train' if args.split_protocol == '15-15' else 'validation'
    population = [dict(metadata[r.sample_id], split=r.split, retained_position=r.retained_position)
                  for r in manifest.rows]
    reference = None
    if args.match_run1 is not None:
        reference = match_run1_reference(args.match_run1, prepared, population, molecules,
                                        args.epochs, args.input_dir, args.split_protocol)
    output.mkdir(parents=True, exist_ok=False)
    run1.write_json(output / 'population.json', population)
    run1.write_json(output / 'audit.json', audits)
    run1.write_json(output / 'split_manifest.json', {
        m: {split: [metadata[r.sample_id]['geometry_id'] for r in manifest.rows
                    if r.molecule == m and r.split == split]
            for split in run1.SPLIT_PROTOCOLS[args.split_protocol]} for m in molecules})
    run1.write_json(output / 'preprocessing.json', {
        f'{m}/{method}': asdict(model.constants) for (m, method), (model, *_) in prepared.items()})
    run1.write_json(output / 'parameter_counts.json', {
        m: {'MB-1': audit['MB1_parameters'], **audit['parameter_counts']}
        for m, audit in zip(molecules, audits, strict=True)})
    run1.write_json(output / 'configuration.json', dict(
        molecules=molecules, trained_molecules=[] if args.validate_only else run_molecules,
        methods=methods, display_labels={m: DISPLAY_LABELS[m] for m in methods},
        seeds=seeds, epochs=epochs, requested_epochs=args.epochs,
        smoke_only=args.smoke_only, validate_only=args.validate_only,
        batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, optimizer='torch.optim.Adam',
        optimizer_defaults=dict(betas=[.9, .999], eps=1e-8, weight_decay=0., amsgrad=False),
        loss='mean squared error on unscaled Hartree correlation energy', target=run1.TARGET.formula,
        input_dir=str(args.input_dir.resolve()), implementation='local codes/run2.py + run2_model.py; current run1.py + circuit.py + train.py',
        reference_run1=reference, population=manifest.split_rule, split_protocol=args.split_protocol,
        split_manifest=str(args.split_manifest.resolve()) if args.split_manifest is not None else None,
        full_AO_until_final_descriptor_selection=True,
        pair_axis_convention=AXIS_CONVENTION, pair_extraction=PAIR_EXTRACTION,
        pair_preprocessing='raw P_munu and Methods Lambda_abab; no centering or normalization',
        one_body_encoder='original MB1: (pi/2)*tanh(a1*P_tilde_mu+b1*F_tilde_mu+c1)',
        pair_encoders={'MB2': '(pi/2)*tanh(a2*P_munu+b2*Lambda_Methods_munu_munu)',
                       'MB2_prime': '(pi/2)*tanh(a2*P_munu)'},
        pair_gate='IsingZZ(gamma) = exp(-1j*gamma*Z_mu*Z_nu/2); unique mu<nu, once',
        num_hea_layers=2, backend='default.qubit', shots=None, interface='torch',
        diff_method='backprop', dtype='float64', device='cpu',
        readout='unchanged Run1 padded Z plus three moments and scalar linear readout',
        initialization='original MB1 normal-vector draws first; append 0.05*N(0,1) a2,b2 draws; prime omits b2; train-mean bias',
        minibatch_order='unchanged train.SampleOrder(seed), independent of parameter draws',
        checkpoint_selection=f'strictly improving {selection} MAE; earliest epoch on ties',
        final_test_evaluation='once after restoring selected checkpoint; never used for selection',
        overall='equal-weight molecule MAE per seed, then canonical seed mean/SD/SE; all summary units mHa',
        runtime=dict(python=platform.python_version(), numpy=np.__version__,
                     torch=torch.__version__, pennylane=qml.__version__, pandas=pd.__version__)))
    print(f'Run2 preparation passed. Output: {output}', flush=True)
    if args.validate_only:
        return output
    metrics, completed = [], set()
    for method in methods:
        metrics.extend(train_one(output, smoke_molecule, method, 0, prepared, metadata,
                                 num_epochs=epochs, checkpoint_selection=selection))
        completed.add((smoke_molecule, method, 0))
    run1.write_json(output / 'smoke_result.json', dict(
        status='PASS', molecule=smoke_molecule, seed=0, epochs=epochs, methods=methods,
        smoke_only=args.smoke_only, finite_predictions_and_metrics=True, metrics=metrics))
    if not args.smoke_only:
        for molecule in molecules:
            for method in methods:
                for seed in SEEDS:
                    if (molecule, method, seed) not in completed:
                        metrics.extend(train_one(output, molecule, method, seed, prepared, metadata,
                                                 num_epochs=epochs, checkpoint_selection=selection))
    summarize(output, metrics, run_molecules, splits=run1.SPLIT_PROTOCOLS[args.split_protocol],
              methods=methods, seeds=seeds)
    print(f'Run2 {"smoke" if args.smoke_only else "experiment"} complete. Results: {output}', flush=True)
    return output


if __name__ == '__main__':
    main()
