"""Run 2 pair encoders around the unchanged Run 1 MB-1 model.

The common flat parameter vector, train-only diagonal preprocessing, two-layer
HEA, and scalar readout are inherited from MB-1. Pair coefficients are appended
after that vector so additional draws cannot shift any common initialization.
The original layout names a/b/c denote a1/b1/c1 in the Run 2 equations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path

import numpy as np
import pennylane as qml
import torch

from circuit import (
    CorrelationEnergyModel, ParameterLayout, ProcessedSample, immutable_array,
    pair_addresses, shared_ansatz, shared_mb1_encoder,
)


METHODS = ('MB2', 'MB2_prime')
DISPLAY_LABELS = {'MB2': 'MB2', 'MB2_prime': 'MB2\u2032'}


@dataclass(frozen=True, eq=False)
class PairSample(ProcessedSample):
    """Current Run 1 sample plus raw Methods Lambda_abab in upper-pair order."""

    lambda_pairs: np.ndarray | None = field(default=None, kw_only=True)

    def __post_init__(self):
        super().__post_init__()
        if self.lambda_pairs is not None:
            values = np.asarray(self.lambda_pairs)
            count = len(self.pii) * (len(self.pii) - 1) // 2
            if (values.shape != (count,) or values.dtype.kind != 'f'
                    or not np.isfinite(values).all()):
                raise ValueError('Lambda pairs must be finite floating values in upper-pair order')
            object.__setattr__(self, 'lambda_pairs', immutable_array(values))


@dataclass(frozen=True)
class PairParameterLayout(ParameterLayout):
    """Preserve every MB-1 index and append only the requested pair weights."""

    method: str = 'MB2'

    def __post_init__(self):
        super().__post_init__()
        if self.method not in METHODS:
            raise ValueError(f'Unknown Run 2 method: {self.method}')
        if self.num_hea_layers != 2 or not self.include_fock:
            raise ValueError('Run 2 requires the original MB-1 encoder and two-layer HEA')

    @property
    def common_num_parameters(self):
        return super().num_parameters

    @property
    def pair_names(self):
        return ('a2', 'b2') if self.method == 'MB2' else ('a2',)

    @property
    def num_parameters(self):
        return self.common_num_parameters + len(self.pair_names)

    @property
    def names(self):
        return super().names + self.pair_names


def pair_encoder(n, p_matrix, lambda_pairs, coefficients, *, method):
    """Encode raw pair entries exactly once, using exp(-i gamma ZZ / 2)."""
    if method not in METHODS:
        raise ValueError(f'Unknown Run 2 method: {method}')
    a2 = coefficients[0]
    for position, (mu, nu) in enumerate(pair_addresses(n)):
        argument = a2 * p_matrix[mu, nu]
        if method == 'MB2':
            argument = argument + coefficients[1] * lambda_pairs[position]
        gamma = 0.5 * math.pi * qml.math.tanh(argument)
        qml.IsingZZ(gamma, wires=[mu, nu])


@dataclass(frozen=True)
class PairEnergyModel(CorrelationEnergyModel):
    """Original MB-1 -> one raw ZZ pair layer -> original HEA and readout."""

    method: str = 'MB2'
    _qnode: object = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        super().__post_init__()
        if self.constants.model_label != 'MB-1':
            raise ValueError('Run 2 must reuse constants fitted for original MB-1')
        # Validate the fixed HEA and full parameter layout at construction.
        self.layout
        n = self.constants.n_active
        method = self.method
        device = qml.device('default.qubit', wires=n, shots=None, seed=0)

        @qml.qnode(device, interface='torch', diff_method='backprop')
        def circuit(p_diag_std, f_diag_std, p_matrix, lambda_pairs, abc, phi, coefficients):
            shared_mb1_encoder(n, p_diag_std, f_diag_std, abc)
            pair_encoder(n, p_matrix, lambda_pairs, coefficients, method=method)
            shared_ansatz(n, phi, num_layers=2)
            return tuple(qml.expval(qml.PauliZ(wire)) for wire in range(n))

        object.__setattr__(self, '_qnode', circuit)

    @property
    def layout(self):
        return PairParameterLayout(self.constants.n_active, self.num_hea_layers,
                                   include_fock=True, method=self.method)

    @property
    def qnode(self):
        """Expose the circuit for tape-level architecture and gate verification."""
        return self._qnode

    def split_parameters(self, theta):
        """Give the inherited readout exactly its original weights, without a2/b2."""
        self.layout.validate(theta)
        return (theta[:3], theta[3:self.layout.bias_index],
                theta[self.layout.bias_index:self.layout.common_num_parameters])

    def pair_quantum_arguments(self, sample, theta):
        """Reuse MB-1 standardization; pass raw P/Lambda pairs without rescaling."""
        p_diag_std, f_diag_std, _, _, abc, phi = self._quantum_arguments(sample, theta)
        n = self.constants.n_active
        p_matrix = np.asarray(sample.pij)
        if (p_matrix.shape != (n, n) or p_matrix.dtype.kind != 'f'
                or not np.isfinite(p_matrix).all()):
            raise ValueError('Pair density must be a finite matrix in the current MB-1 space')
        p_matrix = torch.tensor(p_matrix, dtype=theta.dtype, device=theta.device)
        lambda_tensor = None
        if self.method == 'MB2':
            # MB2_prime never accesses, transforms, or passes cumulant data.
            values = getattr(sample, 'lambda_pairs', None)
            if values is None:
                raise ValueError('MB2 requires the saved Methods Lambda_abab pairs')
            values = np.asarray(values)
            if (values.shape != (len(pair_addresses(n)),) or values.dtype.kind != 'f'
                    or not np.isfinite(values).all()):
                raise ValueError('Invalid saved Methods Lambda_abab pairs')
            lambda_tensor = torch.tensor(values, dtype=theta.dtype, device=theta.device)
        coefficients = theta[self.layout.common_num_parameters:]
        return (p_diag_std, f_diag_std, p_matrix, lambda_tensor, abc, phi, coefficients)

    def quantum_features(self, sample, theta):
        return qml.math.stack(self.qnode(*self.pair_quantum_arguments(sample, theta))).reshape(-1)

    def initialize_theta(self, train_samples, seed):
        """Keep exact MB-1 common draws and append shared-seed 0.05-normal weights."""
        if not train_samples:
            raise ValueError('Training set is empty')
        generator = torch.Generator(device='cpu').manual_seed(seed)
        common = 0.05 * torch.randn(self.layout.common_num_parameters,
                                   dtype=torch.float64, device='cpu', generator=generator)
        common[self.layout.bias_index] = np.mean([float(s.target_energy) for s in train_samples])
        # Separate scalar draws guarantee a2 matches even when b2 is absent.
        pair = [0.05 * torch.randn(1, dtype=torch.float64, device='cpu', generator=generator)
                for _ in self.layout.pair_names]
        theta = torch.cat((common, *pair)).requires_grad_()
        self.layout.validate(theta)
        return theta

    def validate_gradients(self, theta):
        """The generic Run 1 Adam optimizer owns the full leaf, including pair weights."""
        if theta.grad is None or tuple(theta.grad.shape) != (self.num_parameters,):
            raise ValueError('Run 2 requires a gradient for the complete parameter vector')
        if not bool(torch.isfinite(theta.grad).all()):
            raise FloatingPointError('Nonfinite Run 2 gradient')

    def load_theta(self, path):
        """Restore a saved full vector for prediction, rejecting incompatible layouts."""
        path = Path(path)
        if path.is_dir():
            path = path / 'best_theta.npy'
        values = np.load(path, allow_pickle=False)
        if values.dtype != np.float64:
            raise TypeError('Saved Run 2 parameters must be float64')
        theta = torch.tensor(values, dtype=torch.float64)
        self.layout.validate(theta)
        return theta
