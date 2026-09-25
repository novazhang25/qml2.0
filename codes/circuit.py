"""Run 1 descriptors, train-only scalers, circuits, and energy models.

Scientific functions are copied from the original geometry, fingerprint,
constants, circuit, and model sources. There are no canonical_pipeline imports.
FG uses geometric features; FE/FE_prime raw spectra are selected in run1.py;
MB-1 uses P/F diagonals and the shared two-layer ansatz, without pair gates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pennylane as qml
import torch

# Sample records and molecule/AO definitions

@dataclass(frozen=True)
class TargetDefinition:
    method: str = 'FCI'
    reference: str = 'UHF'
    formula: str = 'fci_total_energy - uhf_total_energy'
    unit: str = 'hartree'
    normalization: str = 'none'
    orbital_reference: str = 'RHF'
    active_space: str = 'all orbitals after frozen core'

def immutable_array(value: np.ndarray) -> np.ndarray:
    """Preserve values/dtype/shape; immutable bytes prevent re-enabling writes."""
    a = np.asarray(value)
    if a.dtype.hasobject:
        raise ValueError('Object arrays are not scientific inputs')
    return np.frombuffer(a.tobytes(order='C'), dtype=a.dtype).reshape(a.shape)

@dataclass(frozen=True, eq=False)
class ProcessedSample:
    sample_id: str
    dataset_id: str
    molecule: str
    geometry_index: int
    scan_coordinate: float
    scan_coordinate_name: str
    scan_coordinate_unit: str
    basis: str
    active_ao_indices: np.ndarray
    active_ao_labels: tuple[str, ...] | None
    active_ao_order_source: str
    pii: np.ndarray
    fii: np.ndarray
    pij: np.ndarray
    t_lowdin: np.ndarray | None
    uhf_total_energy: float
    fci_total_energy: float
    fci_corr_target: float
    target_energy: float
    target_definition: TargetDefinition
    ump2_total_energy: float | None
    ump2_corr_energy: float | None
    source_file: Path
    source_file_id: str
    source_sha256: str
    adapter_profile: str
    field_provenance: tuple[tuple[str, str], ...]
    fij: np.ndarray | None = field(default=None, kw_only=True)

    def __post_init__(self):
        """Validate the record and preserve immutable descriptors or model settings."""
        for name in ('active_ao_indices', 'pii', 'fii', 'pij', 'fij', 't_lowdin'):
            a = getattr(self, name)
            if a is not None:
                object.__setattr__(self, name, immutable_array(a))
        if self.active_ao_labels is not None:
            object.__setattr__(self, 'active_ao_labels', tuple(self.active_ao_labels))
        object.__setattr__(self, 'field_provenance', tuple((tuple(x) for x in self.field_provenance)))
REGISTRY_VERSION = 'paper-molecules-sto3g-v1'

@dataclass(frozen=True)
class Molecule:
    molecule_id: str
    geometry_kind: str
    electronic_atom_order: tuple[str, ...]
    scan_coordinate_name: str
    fixed_geometry_parameters: tuple[tuple[str, float], ...]
    frozen_ao_labels: tuple[str, ...]
    expected_core_count: int
    expected_active_ao_indices: tuple[int, ...]
    expected_active_ao_labels: tuple[str, ...] | None
    fg_geometry_adapter: str
    fg_feature_kind: str
    basis: str = 'sto-3g'
    charge: int = 0
    spin: int = 0

    @property
    def expected_active_ao_count(self):
        """Return the number of valence AOs, which sets the FE and MB-1 width."""
        return len(self.expected_active_ao_indices)

def valence(atom, symbol):
    """Build ordered second-shell AO labels for one atom."""
    return tuple((f'{atom} {symbol} {orb}' for orb in ('2s', '2px', '2py', '2pz')))
MOLECULES = MappingProxyType({
    "LiH": Molecule("LiH", "diatomic", ("Li", "H"), "Li-H distance", (),
        ("Li 1s",), 1, (1, 2, 3, 4, 5), valence(0, "Li") + ("1 H 1s",), "identity", "diatomic_2"),
    "BeH2": Molecule("BeH2", "symmetric_linear", ("H", "Be", "H"), "symmetric Be-H distance",
        (("HBeH_angle_deg", 180.0),), ("Be 1s",), 1, (0, 2, 3, 4, 5, 6),
        ("0 H 1s",) + valence(1, "Be") + ("2 H 1s",), "source_1_2_0", "xh2_6"),
    "H2O": Molecule("H2O", "symmetric_bent", ("O", "H", "H"), "symmetric O-H distance",
        (("HOH_angle_deg", 104.5),), ("O 1s",), 1, (1, 2, 3, 4, 5, 6),
        valence(0, "O") + ("1 H 1s", "2 H 1s"), "identity", "xh2_6"),
    "NH3": Molecule("NH3", "symmetric_pyramidal", ("N", "H", "H", "H"), "symmetric N-H distance",
        (("HNH_angle_deg", 106.7),), ("N 1s",), 1, (1, 2, 3, 4, 5, 6, 7),
        valence(0, "N") + ("1 H 1s", "2 H 1s", "3 H 1s"), "identity", "nh3_7"),
    "N2": Molecule("N2", "diatomic", ("N", "N"), "N-N distance", (), ("N 1s",), 2,
        (1, 2, 3, 4, 6, 7, 8, 9), valence(0, "N") + valence(1, "N"), "identity", "diatomic_2"),
    "CO": Molecule("CO", "diatomic", ("C", "O"), "C-O distance", (), ("C 1s", "O 1s"), 2,
        (1, 2, 3, 4, 6, 7, 8, 9), valence(0, "C") + valence(1, "O"), "identity", "diatomic_2"),
    "HF": Molecule("HF", "diatomic", ("H", "F"), "H-F distance", (), ("F 1s",), 1,
        (0, 2, 3, 4, 5), ("0 H 1s",) + valence(1, "F"), "identity", "diatomic_2"),
    # PySCF STO-3G groups angular momenta: S 3s precedes S 2p.
    # The H2S loader uses the saved Ne-core-excluded six-AO descriptor space.
    # This post-Lowdin AO selection does not change MP2 occupied-MO freezing.
    "H2S": Molecule("H2S", "symmetric_bent", ("S", "H", "H"), "symmetric S-H distance",
        (("HSH_angle_deg", 92.1),), ("S 1s", "S 2s", "S 2p"), 5,
        (2, 6, 7, 8, 9, 10),
        ("0 S 3s", "0 S 3px", "0 S 3py", "0 S 3pz", "1 H 1s", "2 H 1s"), "identity", "xh2_6"),
    "H2O2": Molecule("H2O2", "peroxide", ("O", "O", "H", "H"), "O-O distance",
        (("OH_distance_angstrom", .95), ("OOH_angle_deg", 100.0), ("torsion_parameter_deg", 111.5)),
        ("O 1s",), 2, (1, 2, 3, 4, 6, 7, 8, 9, 10, 11),
        valence(0, "O") + valence(1, "O") + ("2 H 1s", "3 H 1s"), "identity", "h2o2_10"),
})

# FG geometry features and atom ordering

def distance(a, b):
    """Compute an interatomic distance in Angstrom from stored Cartesian coordinates."""
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))

def angle(a, center, b):
    """Compute a bond angle in radians using its central atom."""
    v1, v2 = (np.asarray(a) - np.asarray(center), np.asarray(b) - np.asarray(center))
    n1, n2 = (float(np.linalg.norm(v1)), float(np.linalg.norm(v2)))
    if n1 <= 0 or n2 <= 0:
        raise ValueError('Cannot define angle from zero-length bond')
    return float(np.arccos(np.clip(float(np.dot(v1, v2) / (n1 * n2)), -1.0, 1.0)))

def dihedral(p0, p1, p2, p3):
    """Compute the signed four-atom torsion angle in radians."""
    b0, b1, b2 = (-(p1 - p0), p2 - p1, p3 - p2)
    norm = float(np.linalg.norm(b1))
    if norm < 1e-12:
        raise ValueError('Zero central dihedral bond')
    hat = b1 / norm
    v = b0 - float(np.dot(b0, hat)) * hat
    w = b2 - float(np.dot(b2, hat)) * hat
    if np.linalg.norm(v) < 1e-12 or np.linalg.norm(w) < 1e-12:
        raise ValueError('Collinear dihedral')
    return float(np.arctan2(float(np.dot(np.cross(hat, v), w)), float(np.dot(v, w))))

def _diatomic_features(c):
    """Return bond length and its reciprocal, in that order."""
    d = distance(*c)
    return _distances_with_reciprocals([d])

def _distances_with_reciprocals(distances):
    """Keep included distances in order, followed by exactly their reciprocals."""
    ds = np.asarray(distances, dtype=float)
    if not np.isfinite(ds).all() or np.any(ds <= 0):
        raise ValueError('FG distances must be finite and strictly positive')
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        reciprocals = 1.0 / ds
    if not np.isfinite(reciprocals).all():
        raise ValueError('FG reciprocal distances must be finite')
    return np.concatenate((ds, reciprocals))

def _xh2_features(c):
    """Return three distances, the two X-H reciprocals, and the H-X-H angle."""
    x, h1, h2 = c
    d1, d2, dhh = (distance(x, h1), distance(x, h2), distance(h1, h2))
    if min(d1, d2) < 1e-12:
        raise ValueError("X-H distances must be strictly positive")
    ds = np.asarray([d1, d2, dhh, 1.0 / d1, 1.0 / d2])
    return np.concatenate((ds, [angle(h1, x, h2)]))

def _pyramidal_features(c):
    """Return three N-H distances, their reciprocals, and the H1-N-H2 angle."""
    x, h1, h2, h3 = c
    ds = np.asarray([distance(x, h) for h in (h1, h2, h3)])
    return np.concatenate((_distances_with_reciprocals(ds), [angle(h1, x, h2)]))

def _peroxide_features(c):
    """Return six distances, the H-H reciprocal, two angles, and torsion."""
    o1, o2, h1, h2 = c
    dhh = distance(h1, h2)
    if dhh < 1e-12:
        raise ValueError('Peroxide H-H distance too small')
    ds = np.asarray([distance(o1, o2), distance(o1, h1), distance(o2, h2), distance(o1, h2), distance(o2, h1), dhh, 1.0 / dhh])
    return np.concatenate((ds, [angle(o2, o1, h1), angle(o1, o2, h2), dihedral(h1, o1, o2, h2)]))
FEATURES = {'diatomic_2': _diatomic_features, 'xh2_6': _xh2_features, 'nh3_7': _pyramidal_features, 'h2o2_10': _peroxide_features}

def fg_feature_names(molecule):
    """Name the original FG features in their encoding order."""
    kind = MOLECULES[molecule].fg_feature_kind
    distances, other = {
        'diatomic_2': (('r_1_2',), ()),
        'xh2_6': (('r_X_H1', 'r_X_H2', 'r_H1_H2'), ('angle_H1_X_H2',)),
        'nh3_7': (('r_N_H1', 'r_N_H2', 'r_N_H3'), ('angle_H1_N_H2',)),
        'h2o2_10': (('r_O1_O2', 'r_O1_H1', 'r_O2_H2', 'r_O1_H2', 'r_O2_H1', 'r_H1_H2'),
                       ('angle_O2_O1_H1', 'angle_O1_O2_H2', 'dihedral_H1_O1_O2_H2')),
    }[kind]
    reciprocal_distances = distances[:2] if kind == 'xh2_6' else distances[-1:] if kind == 'h2o2_10' else distances
    return distances + tuple(f'1/{name}' for name in reciprocal_distances) + other

def fg_features(molecule, coordinates):
    """Compute the original molecule-specific FG feature vector."""
    kind = MOLECULES[molecule].fg_feature_kind
    if kind not in FEATURES:
        raise ValueError(f'{molecule}: unsupported FG feature contract')
    result = FEATURES[kind](coordinates)
    if result.shape != (len(fg_feature_names(molecule)),) or not np.isfinite(result).all():
        raise ValueError(f'{molecule}: invalid FG feature shape or nonfinite values')
    return immutable_array(result)

def geometry_contract(molecule):
    """Return only the atom-order mapping needed for stored-coordinate FG input."""
    spec = MOLECULES[molecule]
    mapping = {'identity': tuple(range(len(spec.electronic_atom_order))), 'source_1_2_0': (1, 2, 0)}
    return {'source_to_fg': mapping[spec.fg_geometry_adapter]}

# MB-1 train-only descriptor statistics

SCHEMA_VERSION = 'mb-encoding-constants-v1'
PAIR_CONTRACT = 'active-i-lt-j-row-major-v1'
TRIPLE_CONTRACT = 'active-i-lt-j-lt-k-nested-v1'
PERCENTILE_CONTRACT = 'numpy-float64-abs-percentile-95-linear-v1'
MB1_PRIME = 'MB1_PRIME'
ONE_BODY_MODELS = ('MB-1', MB1_PRIME)
MODEL_LABELS = ('MB-1', MB1_PRIME, 'MB-2', 'MB-3', 'lambda_abab_only_no_zzz', 'pij_plus_lambda_abab_no_zzz')

def pair_addresses(n: int) -> tuple[tuple[int, int], ...]:
    """List distinct AO pairs in lexicographic order."""
    return tuple(((i, j) for i in range(n) for j in range(i + 1, n)))

def triple_addresses(n: int) -> tuple[tuple[int, int, int], ...]:
    """List distinct AO triples in lexicographic order."""
    return tuple(((i, j, k) for i in range(n) for j in range(i + 1, n) for k in range(j + 1, n)))

def upper_pair_values(Pij: np.ndarray) -> np.ndarray:
    """Same np.triu_indices traversal as historical data.upper_pair_values."""
    a = np.asarray(Pij, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError('Pij must be a square matrix')
    i, j = np.triu_indices(a.shape[0], k=1)
    return immutable_array(np.asarray(a[i, j], dtype=float))

def strict_upper_triple_values(T: np.ndarray) -> np.ndarray:
    """Extract ordered triple entries; unused by the MB-1 run."""
    a = np.asarray(T, dtype=float)
    if a.ndim != 3 or len(set(a.shape)) != 1:
        raise ValueError('T must be cubic')
    return immutable_array(np.asarray([a[i, j, k] for i, j, k in triple_addresses(a.shape[0])], dtype=float))

def _concat_nonempty(arrays, *, name):
    """Flatten and combine finite training descriptors before fitting scalar statistics."""
    flat = []
    for x in arrays:
        a = np.asarray(x)
        if a.size == 0:
            continue
        a = np.asarray(a, dtype=float).reshape(-1)
        if not np.all(np.isfinite(a)):
            raise ValueError(f'{name} contains non-finite entries')
        flat.append(a)
    if not flat:
        raise ValueError(f'No training entries for {name}')
    return np.concatenate(flat)

def _p95_abs(x, *, name):
    """Fit the 95th percentile of absolute training pair values with linear interpolation."""
    a = np.asarray(x, dtype=float).reshape(-1)
    if not a.size or not np.all(np.isfinite(a)):
        raise ValueError(f'Invalid entries for {name}')
    value = float(np.percentile(np.abs(a), 95, method='linear'))
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f'Invalid p95(abs({name})) = {value}')
    return value

@dataclass(frozen=True)
class EncodingConstants:
    manifest_id: str
    molecule: str
    model_label: str
    training_sample_ids: tuple[str, ...]
    n_active: int
    Pii_mean: float
    Pii_std: float
    Fii_mean: float
    Fii_std: float
    Pij_abs95: float
    alpha: float
    T_abs95: float | None
    beta: float | None
    schema_version: str = SCHEMA_VERSION
    pair_contract: str = PAIR_CONTRACT
    triple_contract: str | None = None
    percentile_contract: str = PERCENTILE_CONTRACT

    def __post_init__(self):
        """Validate the record and preserve immutable descriptors or model settings."""
        object.__setattr__(self, 'training_sample_ids', tuple(self.training_sample_ids))
        if self.model_label not in MODEL_LABELS or self.molecule not in MOLECULES:
            raise ValueError('Canonical molecule/model required')
        if not self.training_sample_ids or len(set(self.training_sample_ids)) != len(self.training_sample_ids):
            raise ValueError('Nonempty unique training identities required')
        if self.n_active != MOLECULES[self.molecule].expected_active_ao_count:
            raise ValueError('Active dimension mismatch')
        if not all((np.isfinite(v) for v in (self.Pii_mean, self.Fii_mean))):
            raise ValueError('Non-finite mean')
        scales = [self.Pii_std, self.Fii_std, self.Pij_abs95, self.alpha]
        if self.model_label == 'MB-3':
            if self.T_abs95 is None or self.beta is None or self.triple_contract != TRIPLE_CONTRACT:
                raise ValueError('MB-3 requires T scale and addressing contract')
            scales += [self.T_abs95, self.beta]
        elif any((v is not None for v in (self.T_abs95, self.beta, self.triple_contract))):
            raise ValueError('MB-1/MB-2 must not have T preprocessing')
        if not all((np.isfinite(v) and v > 0 for v in scales)):
            raise ValueError('Encoding scales must be finite and positive')

    @property
    def preprocessing_id(self):
        """Use the immutable fitted record as a circuit-cache key; no checksum is computed."""
        return repr(self)

    @property
    def pair_count_per_sample(self):
        """Return the number of distinct AO pairs."""
        return self.n_active * (self.n_active - 1) // 2

    @property
    def triple_count_per_sample(self):
        """Return zero for MB-1, which does not encode triple descriptors."""
        return self.n_active * (self.n_active - 1) * (self.n_active - 2) // 6 if self.model_label == 'MB-3' else 0

def compute_encoding_constants(samples, *, manifest: SimpleNamespace, molecule: str, model_label: str) -> EncodingConstants:
    """Fit solely manifest-selected training rows, in retained-position order.

    Supply the complete manifest inventory as accepted ProcessedSample records.
    Source identities are checked against the manifest; absolute paths are not
    used. T need not be loaded for MB-1/MB-2. No descriptor is modified.
    """
    if model_label not in MODEL_LABELS:
        raise ValueError('Use canonical MB-1 / MB-2 / MB-3 labels')
    if molecule not in MOLECULES:
        raise ValueError('Unknown paper molecule')
    samples = tuple(samples)
    if not all((isinstance(s, ProcessedSample) for s in samples)):
        raise TypeError('Accepted ProcessedSample records required')
    by_id = {s.sample_id: s for s in samples}
    if len(by_id) != len(samples) or set(by_id) != {r.sample_id for r in manifest.rows}:
        raise ValueError('Samples must exactly match declared manifest inventory')
    for row in manifest.rows:
        s = by_id[row.sample_id]
        if s.dataset_id != manifest.dataset_id or s.molecule != row.molecule or s.geometry_index != row.geometry_index or (s.scan_coordinate != row.scan_coordinate) or (s.scan_coordinate_unit != row.scan_coordinate_unit) or (s.source_file_id != row.source_file_id) or (s.source_sha256 != row.source_sha256) or (s.target_definition != manifest.target_definition):
            raise ValueError(f'Sample identity contradicts manifest: {row.sample_id}')
    rows = sorted((r for r in manifest.rows if r.molecule == molecule and r.included and (r.split == 'train')), key=lambda r: r.retained_position)
    if not rows:
        raise ValueError('No training samples for molecule')
    train = [by_id[r.sample_id] for r in rows]
    n = MOLECULES[molecule].expected_active_ao_count
    include_t = model_label == 'MB-3'
    for s in train:
        if s.basis != MOLECULES[molecule].basis or tuple(s.active_ao_indices) != MOLECULES[molecule].expected_active_ao_indices:
            raise ValueError('Training AO identity mismatch')
        arrays = [('Pii', s.pii, (n,)), ('Fii', s.fii, (n,)), ('Pij', s.pij, (n, n))]
        if include_t:
            arrays.append(('T', s.t_lowdin, (n, n, n)))
        for name, a, shape in arrays:
            if a is None or a.shape != shape or a.dtype.kind != 'f' or (not np.isfinite(a).all()) or a.flags.writeable:
                raise ValueError(f'Missing/invalid immutable {name}: {s.sample_id}')
    Pii = _concat_nonempty([s.pii for s in train], name='Pii')
    Fii = _concat_nonempty([s.fii for s in train], name='Fii')
    Pij = _concat_nonempty([upper_pair_values(s.pij) for s in train], name='Pij')
    pair95 = _p95_abs(Pij, name='Pij')
    Pii_mean, Pii_std = (float(np.mean(Pii)), float(np.std(Pii, ddof=0)))
    Fii_mean, Fii_std = (float(np.mean(Fii)), float(np.std(Fii, ddof=0)))
    triple95 = None
    if include_t:
        T = _concat_nonempty([strict_upper_triple_values(s.t_lowdin) for s in train], name='T')
        triple95 = _p95_abs(T, name='T')
    atanh09 = float(np.arctanh(0.9))
    return EncodingConstants(manifest.manifest_id, molecule, model_label, tuple((r.sample_id for r in rows)), n, Pii_mean, Pii_std, Fii_mean, Fii_std, pair95, float(atanh09 / pair95), triple95, float(atanh09 / triple95) if include_t else None, triple_contract=TRIPLE_CONTRACT if include_t else None)

def standardize_diagonal(diagonal: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Separate immutable float64 output: historical (diagonal - mean) / std."""
    a = np.asarray(diagonal, dtype=float)
    if a.ndim != 1 or not np.isfinite(a).all() or (not np.isfinite(mean)) or (not np.isfinite(std)) or (std <= 0):
        raise ValueError('Finite diagonal/mean and positive finite std required')
    return immutable_array((a - mean) / std)

# MB-1 encoder, HEA, and linear readout

@dataclass(frozen=True)
class CircuitIdentity:
    model_label: str
    n_active: int
    num_hea_layers: int
    include_layer3: bool
    preprocessing_id: str
    backend: str = 'default.qubit'
    shots: None = None
    interface: str = 'torch'
    diff_method: str = 'backprop'
    device_seed: int = 0

def shared_mb1_encoder(n, p_diag_std, f_diag_std, abc):
    """Queue the original MB-1 one-body RY gates with shared a, b, c."""
    a, b, c = abc
    for i in range(n):
        angle = 0.5 * math.pi * qml.math.tanh(a * p_diag_std[i] + b * f_diag_std[i] + c)
        qml.RY(angle, wires=i)


def shared_ansatz(n, phi, *, num_layers=2):
    """Queue the existing shared RY layers and directed CNOT ring unchanged."""
    for layer in range(num_layers):
        for q in range(n):
            qml.RY(phi[layer], wires=q)
        if n > 1:
            for q in range(n):
                target = (q + 1) % n
                qml.CNOT(wires=[q, target])

class CachedCircuitBank:
    """Instance-owned cache, with immutable, complete circuit identity keys.

    Backend, interface and analytic differentiation are deliberately fixed in
    Phase 1. A local device seed avoids NumPy-global RNG consumption at creation;
    analytic execution does not use sampling. No circuit optimizations are added.
    """

    def __init__(self):
        """Initialize the instance-owned cache, random stream, or checkpoint state."""
        self._qnode_cache = {}

    @staticmethod
    def identity(constants: EncodingConstants, *, num_hea_layers: int=2):
        """Describe the analytic circuit and fitted constants used for cache lookup."""
        if not isinstance(constants, EncodingConstants):
            raise TypeError('Accepted EncodingConstants required')
        if type(num_hea_layers) is not int or num_hea_layers < 1:
            raise ValueError('num_hea_layers must be a positive integer')
        return CircuitIdentity(constants.model_label, constants.n_active, num_hea_layers, constants.model_label == 'MB-3', constants.preprocessing_id)

    @property
    def cached_identities(self):
        """List the circuit configurations already constructed in this instance."""
        return tuple(self._qnode_cache)

    def get_qnode(self, constants: EncodingConstants, *, num_hea_layers: int=2):
        """Build or reuse the analytic PennyLane circuit without caching trainable parameters."""
        key = self.identity(constants, num_hea_layers=num_hea_layers)
        if key in self._qnode_cache:
            return self._qnode_cache[key]
        n = key.n_active
        alpha, beta = (constants.alpha, constants.beta)
        device = qml.device(key.backend, wires=n, shots=key.shots, seed=key.device_seed)

        def queue_circuit(p_diag_std, f_diag_std, p_matrix, abc, phi, t_tensor=None, lambda_pairs=None):
            """Queue the physical encoder, pair gates, two HEA blocks, and Z measurements."""
            if key.model_label == MB1_PRIME:
                a, c = abc
                for i in range(n):
                    angle = 0.5 * math.pi * qml.math.tanh(a * p_diag_std[i] + c)
                    qml.RY(angle, wires=i)
            else:
                shared_mb1_encoder(n, p_diag_std, f_diag_std, abc)
            if key.model_label not in ONE_BODY_MODELS:
                pair_position = 0
                for i in range(n):
                    for j in range(i + 1, n):
                        angle = 0.5 * math.pi * qml.math.tanh(alpha * p_matrix[i, j])
                        qml.IsingZZ(angle, wires=[i, j])
                        pair_position += 1
            if key.include_layer3:
                if beta is None or t_tensor is None:
                    raise ValueError('MB-3 requires beta and T')
                for i in range(n):
                    for j in range(i + 1, n):
                        for k in range(j + 1, n):
                            angle = 0.5 * math.pi * qml.math.tanh(beta * t_tensor[i, j, k])
                            qml.MultiRZ(angle, wires=[i, j, k])
            shared_ansatz(n, phi, num_layers=key.num_hea_layers)
            return tuple((qml.expval(qml.PauliZ(q)) for q in range(n)))
        if key.include_layer3:

            @qml.qnode(device, interface=key.interface, diff_method=key.diff_method)
            def circuit(p_diag_std, f_diag_std, p_matrix, t_tensor, abc, phi):
                """Execute the queued gates using Torch backpropagation on default.qubit."""
                return queue_circuit(p_diag_std, f_diag_std, p_matrix, abc, phi, t_tensor)
        else:

            @qml.qnode(device, interface=key.interface, diff_method=key.diff_method)
            def circuit(p_diag_std, f_diag_std, p_matrix, abc, phi):
                """Execute the queued gates using Torch backpropagation on default.qubit."""
                return queue_circuit(p_diag_std, f_diag_std, p_matrix, abc, phi)
        self._qnode_cache[key] = circuit
        return circuit

    def __call__(self, constants, *, p_diag_std, f_diag_std, p_matrix, t_tensor, abc, phi, num_hea_layers=2, lambda_pairs=None):
        """Evaluate the configured circuit or model without changing its parameters."""
        key = self.identity(constants, num_hea_layers=num_hea_layers)
        n = key.n_active
        prime = key.model_label == MB1_PRIME
        entries = [(p_diag_std, (n,)), (abc, (2 if prime else 3,)), (phi, (num_hea_layers,))]
        if not prime:
            entries.append((f_diag_std, (n,)))
        if key.model_label not in ONE_BODY_MODELS:
            entries.append((p_matrix, (n, n)))
        elif p_matrix is not None:
            raise ValueError('MB-1 circuit inputs must exclude pair descriptors')
        if key.include_layer3:
            entries.append((t_tensor, (n, n, n)))
        elif t_tensor is not None:
            raise ValueError('MB-1/MB-2 circuit inputs must exclude T')
        if lambda_pairs is not None:
            raise ValueError('Lambda supplied to a non-Lambda configuration')
        if any((x is None or tuple(x.shape) != shape for x, shape in entries)):
            raise ValueError('Circuit argument shapes disagree with identity')
        qnode = self.get_qnode(constants, num_hea_layers=num_hea_layers)
        args = (p_diag_std, f_diag_std, p_matrix)
        if key.include_layer3:
            args += (t_tensor,)
        return qml.math.stack(qnode(*args, abc, phi))

@dataclass(frozen=True)
class ParameterLayout:
    n_active: int
    num_hea_layers: int = 2
    include_fock: bool = True

    def __post_init__(self):
        """Validate the record and preserve immutable descriptors or model settings."""
        if type(self.n_active) is not int or self.n_active < 1:
            raise ValueError('n_active must be a positive integer')
        if type(self.num_hea_layers) is not int or self.num_hea_layers < 1:
            raise ValueError('num_hea_layers must be a positive integer')

    @property
    def context_width(self):
        """Return the padded Z readout width, at least eight."""
        return max(8, self.n_active)

    @property
    def readout_input_dim(self):
        """Count padded Z expectations plus the three invariant moments."""
        return self.context_width + 3

    @property
    def encoder_names(self):
        """MB-1' removes b entirely; all remaining parameters keep their names."""
        return ('a', 'b', 'c') if self.include_fock else ('a', 'c')

    @property
    def bias_index(self):
        """Locate the readout bias in the flat trainable parameter vector."""
        return len(self.encoder_names) + self.num_hea_layers

    @property
    def num_parameters(self):
        """Count all trainable circuit, bias, and readout parameters."""
        return self.bias_index + 1 + self.readout_input_dim

    @property
    def names(self):
        """Describe the order of entries in the flat parameter vector."""
        return self.encoder_names + tuple((f'phi{i}' for i in range(self.num_hea_layers))) + ('bias',) + tuple((f'Z{i}_weight' for i in range(self.context_width))) + ('mean_Z_weight', 'mean_Z2_weight', 'mean_Z3_weight')

    def validate(self, theta):
        """Reject parameters with the wrong shape, precision, or nonfinite values."""
        if not isinstance(theta, torch.Tensor) or theta.dtype != torch.float64:
            raise TypeError('theta must be an explicit float64 Torch tensor')
        if tuple(theta.shape) != (self.num_parameters,):
            raise ValueError(f'theta must have shape ({self.num_parameters},)')
        if not bool(torch.isfinite(theta).all()):
            raise ValueError('theta must be finite')

def pad_z_to_length(z, max_z_dim=8):
    """Right-zero-pad real expectations; never truncate or add quantum wires."""
    if z.ndim != 1 or not 0 < z.numel() <= max_z_dim:
        raise ValueError('Nonempty one-dimensional Z must fit the context')
    if z.dtype != torch.float64 or not bool(torch.isfinite(z).all()):
        raise ValueError('Finite float64 Z required')
    return torch.cat((z, z.new_zeros(max_z_dim - z.numel())))

def invariant_moments(z):
    """Compute the means of Z, Z squared, and Z cubed over real qubits."""
    if z.ndim != 1 or z.numel() == 0:
        raise ValueError('Nonempty one-dimensional real Z required')
    m1 = torch.mean(z)
    m2 = torch.mean(z ** 2)
    m3 = torch.mean(z ** 3)
    return torch.stack((m1, m2, m3))

@dataclass(frozen=True)
class CorrelationEnergyModel:
    constants: EncodingConstants
    num_hea_layers: int = 2
    circuit_bank: CachedCircuitBank = field(default_factory=CachedCircuitBank, repr=False, compare=False)

    def __post_init__(self):
        """Validate the record and preserve immutable descriptors or model settings."""
        self.circuit_bank.identity(self.constants, num_hea_layers=self.num_hea_layers)

    @property
    def layout(self):
        """Return the parameter layout for this model and qubit count."""
        return ParameterLayout(self.constants.n_active, self.num_hea_layers,
                               include_fock=self.constants.model_label != MB1_PRIME)

    @property
    def num_parameters(self):
        """Count all trainable circuit, bias, and readout parameters."""
        return self.layout.num_parameters

    def split_parameters(self, theta):
        """Separate encoder coefficients, shared HEA angles, and readout weights."""
        self.layout.validate(theta)
        width = len(self.layout.encoder_names)
        return (theta[:width], theta[width:self.layout.bias_index], theta[self.layout.bias_index:])

    def _quantum_arguments(self, sample: ProcessedSample, theta):
        """Standardize P/F diagonals; MB-1 excludes pair and triple circuit inputs."""
        abc, phi, _ = self.split_parameters(theta)
        c = self.constants
        if not isinstance(sample, ProcessedSample):
            raise TypeError('Accepted ProcessedSample required')
        if sample.molecule != c.molecule or len(sample.pii) != c.n_active:
            raise ValueError('Sample molecule/width disagrees with fitted constants')

        def descriptor(value):
            """Copy one immutable NumPy descriptor into a Torch tensor with the parameter precision."""
            return torch.tensor(value, dtype=theta.dtype, device=theta.device)
        Pii = descriptor(sample.pii)
        if c.model_label == MB1_PRIME:
            # F is never read, standardized, or sent to the MB-1' encoder.
            Pii_std = (Pii - float(c.Pii_mean)) / float(c.Pii_std)
            return (Pii_std, None, None, None, abc, phi)
        Fii = descriptor(sample.fii)
        Pij = None if c.model_label == 'MB-1' else descriptor(sample.pij)
        T = None
        if c.model_label == 'MB-3':
            if sample.t_lowdin is None:
                raise ValueError('MB-3 requires the accepted physical T')
            T = descriptor(sample.t_lowdin)
        Pii_std = (Pii - float(c.Pii_mean)) / float(c.Pii_std)
        Fii_std = (Fii - float(c.Fii_mean)) / float(c.Fii_std)
        return (Pii_std, Fii_std, Pij, T, abc, phi)

    def quantum_features(self, sample, theta):
        """Run the MB-1 circuit and return one Z expectation per active AO."""
        Pii_std, Fii_std, Pij, T, abc, phi = self._quantum_arguments(sample, theta)
        extra = {}
        z = self.circuit_bank(self.constants, p_diag_std=Pii_std, f_diag_std=Fii_std, p_matrix=Pij, t_tensor=T, abc=abc, phi=phi, num_hea_layers=self.num_hea_layers, **extra)
        return z.reshape(-1)

    def readout_features(self, z):
        """Append three Z moments to the padded vector of measured Z expectations."""
        if tuple(z.shape) != (self.constants.n_active,):
            raise ValueError('Readout moments require exactly the real Z vector')
        padded_z = pad_z_to_length(z, max_z_dim=self.layout.context_width)
        moments3 = invariant_moments(z)
        return torch.cat((padded_z, moments3))

    def raw_readout_features(self, sample, theta):
        """Build the circuit-derived features consumed by the linear readout."""
        return self.readout_features(self.quantum_features(sample, theta))

    def apply_readout(self, features, theta):
        """Apply the shared linear readout to one feature vector or a batch."""
        _, _, weights = self.split_parameters(theta)
        if features.ndim not in (1, 2) or features.shape[-1] != self.layout.readout_input_dim:
            raise ValueError('Readout feature shape disagrees with model layout')
        if features.dtype != torch.float64 or not bool(torch.isfinite(features).all()):
            raise ValueError('Finite float64 features required')
        return weights[0] + features @ weights[1:]

    def forward(self, sample, theta):
        """Predict the correlation energy for one geometry."""
        return self.apply_readout(self.raw_readout_features(sample, theta), theta)

    def __call__(self, sample, theta):
        """Evaluate the configured circuit or model without changing its parameters."""
        return self.forward(sample, theta)

    def predict_batch(self, samples, theta):
        """Pure fixed-theta forward: historical feature-first matrix readout.

        This is not an epoch/minibatch scheduler; order is supplied by the caller.
        Stacking scalar predictions instead can alter floating-point gradients.
        """
        samples = tuple(samples)
        if not samples:
            raise ValueError('Cannot predict an empty sample sequence')
        feature_rows = [self.raw_readout_features(s, theta) for s in samples]
        features = torch.stack(feature_rows, dim=0)
        return self.apply_readout(features, theta)

# FG/FE/FE_prime train-only scaling, circuit, and linear readout

VERSION = 'canonical-fingerprint-v1'

@dataclass(frozen=True, eq=False)
class FingerprintSample(ProcessedSample):
    fingerprint: np.ndarray

    def __post_init__(self):
        """Validate the record and preserve immutable descriptors or model settings."""
        super().__post_init__()
        object.__setattr__(self, 'fingerprint', immutable_array(self.fingerprint))

@dataclass(frozen=True)
class FingerprintConstants:
    manifest_id: str
    molecule: str
    model_label: str
    training_sample_ids: tuple
    n_active: int
    mu: tuple
    sigma: tuple
    sigma_safe: tuple
    constant_mask: tuple
    schema_version: str = VERSION
    epsilon: float = 1e-12

    def __post_init__(self):
        """Validate the record and preserve immutable descriptors or model settings."""
        for name in ('training_sample_ids', 'mu', 'sigma', 'sigma_safe', 'constant_mask'):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if self.schema_version != VERSION or self.model_label not in ('FG', 'FE', 'FE_prime') or self.molecule not in MOLECULES:
            raise ValueError('Unsupported fingerprint constants')
        if self.n_active != feature_width(self.molecule, self.model_label) or self.epsilon != 1e-12:
            raise ValueError('Fingerprint width/scaler contract mismatch')
        if not self.training_sample_ids or len(set(self.training_sample_ids)) != len(self.training_sample_ids):
            raise ValueError('Unique fingerprint training identities required')
        arrays = [np.asarray(getattr(self, k)) for k in ('mu', 'sigma', 'sigma_safe', 'constant_mask')]
        if any((x.shape != (self.n_active,) for x in arrays)) or not all((np.isfinite(x).all() for x in arrays)):
            raise ValueError('Invalid fingerprint column statistics')
        mu, sigma, safe, mask = arrays
        if np.any(sigma < 0) or tuple(mask) != tuple(sigma < self.epsilon) or (not np.array_equal(safe, np.where(mask, 1.0, sigma))):
            raise ValueError('Fingerprint constant-column policy mismatch')

    @property
    def preprocessing_id(self):
        """Use the immutable fitted record as a circuit-cache key; no checksum is computed."""
        return repr(self)

def feature_width(molecule, method):
    """Return the raw FG width or the common valence-spectrum width for FE/FE_prime."""
    spec = MOLECULES[molecule]
    if method in ('FE', 'FE_prime'):
        return spec.expected_active_ao_count
    if method != 'FG':
        raise ValueError('Fingerprint method required')
    return len(fg_feature_names(molecule))

def fit_constants(samples, *, manifest, molecule, model_label):
    """Fit per-feature means and population standard deviations on training rows only."""
    by_id = {s.sample_id: s for s in samples}
    rows = sorted((r for r in manifest.rows if r.included and r.molecule == molecule and (r.split == 'train')), key=lambda r: r.retained_position)
    matrix = np.vstack([by_id[r.sample_id].fingerprint for r in rows])
    if matrix.shape != (len(rows), feature_width(molecule, model_label)) or not np.isfinite(matrix).all():
        raise ValueError('Invalid training fingerprint matrix')
    mu = np.mean(matrix, axis=0)
    sigma = np.std(matrix, axis=0, ddof=0)
    mask = sigma < 1e-12
    safe = sigma.copy()
    safe[mask] = 1.0
    return FingerprintConstants(manifest.manifest_id, molecule, model_label, tuple((r.sample_id for r in rows)), matrix.shape[1], tuple(mu.tolist()), tuple(sigma.tolist()), tuple(safe.tolist()), tuple(mask.tolist()))

def encode(features, constants):
    """Standardize fingerprint columns, scale by pi/2, zero constant columns, and clip to pi."""
    a = np.asarray(features, dtype=float)
    values = math.pi / 2.0 * (a - np.asarray(constants.mu)) / np.asarray(constants.sigma_safe)
    values[..., np.asarray(constants.constant_mask, dtype=bool)] = 0.0
    values = np.clip(values, -math.pi, math.pi)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite fingerprint angles')
    return immutable_array(values)

@dataclass(frozen=True)
class FingerprintLayout:
    n_active: int
    num_hea_layers: int = 2

    @property
    def context_width(self):
        """Return the padded Z readout width, at least eight."""
        return max(8, self.n_active)

    @property
    def readout_input_dim(self):
        """Count padded Z expectations plus the three invariant moments."""
        return self.context_width + 3

    @property
    def bias_index(self):
        """Locate the readout bias in the flat trainable parameter vector."""
        return self.num_hea_layers

    @property
    def num_parameters(self):
        """Count all trainable circuit, bias, and readout parameters."""
        return self.num_hea_layers + 1 + self.readout_input_dim

    @property
    def names(self):
        """Describe the order of entries in the flat parameter vector."""
        return tuple((f'phi{i}' for i in range(self.num_hea_layers))) + ('bias',) + tuple((f'Z{i}_weight' for i in range(self.context_width))) + ('mean_Z_weight', 'mean_Z2_weight', 'mean_Z3_weight')

    def validate(self, theta):
        """Reject parameters with the wrong shape, precision, or nonfinite values."""
        if not isinstance(theta, torch.Tensor) or theta.dtype != torch.float64 or tuple(theta.shape) != (self.num_parameters,) or (not bool(torch.isfinite(theta).all())):
            raise ValueError('Finite explicit float64 fingerprint theta with exact layout required')

class FingerprintCircuit:

    def __init__(self):
        """Initialize the instance-owned cache, random stream, or checkpoint state."""
        self._nodes = {}

    def __call__(self, n, angles, phi):
        """Evaluate the configured circuit or model without changing its parameters."""
        import pennylane as qml
        if n not in self._nodes:
            device = qml.device('default.qubit', wires=n)

            @qml.qnode(device, interface='torch', diff_method='backprop')
            def circuit(angles, phi):
                """Execute the queued gates using Torch backpropagation on default.qubit."""
                for wire in range(n):
                    qml.RY(angles[wire], wires=wire)
                shared_ansatz(n, phi, num_layers=2)
                return tuple((qml.expval(qml.PauliZ(wire)) for wire in range(n)))
            self._nodes[n] = circuit
        return torch.stack(self._nodes[n](angles, phi))

@dataclass(frozen=True)
class FingerprintModel:
    constants: FingerprintConstants
    num_hea_layers: int = 2
    circuit_bank: FingerprintCircuit = field(default_factory=FingerprintCircuit, repr=False, compare=False)

    def __post_init__(self):
        """Validate the record and preserve immutable descriptors or model settings."""
        if type(self.constants) is not FingerprintConstants or self.num_hea_layers != 2:
            raise ValueError('Accepted fingerprint constants/depth required')

    @property
    def layout(self):
        """Return the parameter layout for this model and qubit count."""
        return FingerprintLayout(self.constants.n_active, self.num_hea_layers)

    @property
    def num_parameters(self):
        """Count all trainable circuit, bias, and readout parameters."""
        return self.layout.num_parameters

    def initialize_theta(self, train_samples, seed):
        """Preserve fingerprint parameter draw order and initialize bias to the training target mean."""
        generator = torch.Generator(device='cpu').manual_seed(seed)
        parts = [0.05 * torch.randn(shape, dtype=torch.float64, generator=generator) for shape in ((self.num_hea_layers,), (), (self.layout.readout_input_dim,))]
        parts[1].fill_(float(np.mean([s.target_energy for s in train_samples])))
        return torch.cat([x.reshape(-1) for x in parts]).requires_grad_()

    def make_optimizer(self, theta, learning_rate):
        """Create Adam using the original model-specific parameter blocks and learning rate."""
        self.layout.validate(theta)
        index = self.layout.bias_index
        blocks = [theta[:index].detach().requires_grad_(), theta[index].detach().requires_grad_(), theta[index + 1:].detach().requires_grad_()]
        optimizer = torch.optim.Adam(blocks, lr=learning_rate)
        optimizer._fingerprint_theta = theta
        return optimizer

    def prepare_optimizer_step(self, theta, optimizer):
        """Clear the flat fingerprint gradient before computing the next batch loss."""
        if getattr(optimizer, '_fingerprint_theta', None) is not theta:
            raise ValueError('Fingerprint optimizer must bind the actual flat parameter storage')
        theta.grad = None

    def bind_optimizer_gradients(self, theta, optimizer):
        """Attach flat-vector gradients to the original separate Adam parameter blocks."""
        self.validate_gradients(theta)
        if theta.grad is None:
            raise ValueError('Missing fingerprint gradient')
        index = self.layout.bias_index
        gradients = (theta.grad[:index], theta.grad[index], theta.grad[index + 1:])
        for parameter, gradient in zip(optimizer.param_groups[0]['params'], gradients, strict=True):
            parameter.grad = gradient.detach()

    def raw_readout_features(self, sample, theta):
        """Build the circuit-derived features consumed by the linear readout."""
        self.layout.validate(theta)
        return self._raw_features(sample, theta[:self.layout.bias_index])

    def _raw_features(self, sample, phi):
        """Encode a fingerprint, measure Z, and append padded readout features and moments."""
        if not isinstance(sample, FingerprintSample) or sample.molecule != self.constants.molecule or sample.fingerprint.shape != (self.constants.n_active,):
            raise ValueError('Fingerprint sample molecule/width mismatch')
        angles = torch.tensor(encode(sample.fingerprint, self.constants), dtype=torch.float64)
        z = self.circuit_bank(self.constants.n_active, angles, phi)
        return torch.cat((pad_z_to_length(z, self.layout.context_width), invariant_moments(z)))

    def forward(self, sample, theta):
        """Predict the correlation energy for one geometry."""
        return theta[self.layout.bias_index] + torch.dot(theta[self.layout.bias_index + 1:], self.raw_readout_features(sample, theta))

    def predict_batch(self, samples, theta):
        """Predict an ordered batch while preserving this model's original readout arithmetic."""
        if not samples:
            raise ValueError('Nonempty fingerprint batch required')
        self.layout.validate(theta)
        phi = theta[:self.layout.bias_index]
        bias = theta[self.layout.bias_index]
        weights = theta[self.layout.bias_index + 1:]
        return torch.stack([bias + torch.dot(weights, self._raw_features(s, phi)) for s in samples])

    def training_metadata(self, theta):
        """Record shared HEA angles for the per-epoch training history."""
        return {f'phi{i}': theta[i].item() for i in range(self.num_hea_layers)}

    def validate_gradients(self, theta):
        """Stop if a fingerprint gradient contains nonfinite values."""
        if theta.grad is not None and (not bool(torch.isfinite(theta.grad).all())):
            raise FloatingPointError('Nonfinite fingerprint gradient')

# Model selection

def construct_model(constants, depth):
    """Select the shared fingerprint model for FG/FE/FE_prime or the MB-1 model."""
    if constants.model_label in ('FG', 'FE', 'FE_prime'):
        return FingerprintModel(constants, num_hea_layers=depth)
    return CorrelationEnergyModel(constants, num_hea_layers=depth)
