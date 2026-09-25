"""Run 1 initialization, optimization, checkpointing, evaluation, and aggregation.

The best checkpoint uses validation MAE by default, or training MAE for the
explicit historical two-way protocol. Training never receives the test
population. This module does not load data or launch runs when imported.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

# Initialization, minibatches, Adam, and the training loop

def initialize_parameters(model, train_samples, *, seed):
    """One full normal vector, then overwrite bias with ordered train mean.

    Keep the overwritten bias draw. CPU Generator uses the same Torch algorithm
    as historical manual_seed + randn_like, without changing global RNG state.
    """
    if not train_samples:
        raise ValueError('Training set is empty')
    if hasattr(model, 'initialize_theta'):
        return model.initialize_theta(train_samples, seed)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    prime = getattr(getattr(model, 'constants', None), 'model_label', None) == 'MB1_PRIME'
    theta = 0.05 * torch.randn(model.num_parameters + int(prime), dtype=torch.float64, device='cpu', generator=generator)
    if prime:
        # Preserve MB-1's exact seeded draws for every surviving parameter.
        # The discarded b draw is never included in the trainable tensor.
        theta = torch.cat((theta[:1], theta[2:]))
    mean_target = np.mean([float(s.target_energy) for s in train_samples])
    theta[model.layout.bias_index] = mean_target
    return theta.requires_grad_()

class SampleOrder:
    """Independent, persistent per-run shuffle stream; fresh randperm per epoch."""

    def __init__(self, seed):
        """Initialize the instance-owned cache, random stream, or checkpoint state."""
        self.generator = torch.Generator(device='cpu').manual_seed(seed)

    def permutation(self, count):
        """Draw the next epoch order from the persistent, seed-local shuffle generator."""
        return torch.randperm(count, generator=self.generator).tolist()

def iter_batches(samples, batch_size, indices=None):
    """Yield minibatches in the supplied order, including the final partial batch."""
    if batch_size <= 0:
        raise ValueError('batch_size must be positive')
    if indices is None:
        indices = list(range(len(samples)))
    for start in range(0, len(indices), batch_size):
        yield [samples[index] for index in indices[start:start + batch_size]]

def make_optimizer(theta, *, learning_rate=0.02, model=None):
    """One independent theta group, actual historical Adam constructor arguments.

    Effective defaults are runtime-qualified in STEP6_CONTRACT.md and compared
    in tests. No scheduler, clipping or separate inactive-parameter groups.
    """
    if theta.dtype != torch.float64 or theta.device.type != 'cpu' or (not theta.is_leaf) or (not theta.requires_grad):
        raise ValueError('A caller-owned, trainable CPU float64 leaf theta is required')
    if model is not None and hasattr(model, 'make_optimizer'):
        return model.make_optimizer(theta, learning_rate)
    return torch.optim.Adam([theta], lr=learning_rate)

def targets(samples, theta):
    """Build the batch target tensor in the same precision and device as theta."""
    return torch.stack([torch.as_tensor(s.target_energy, dtype=theta.dtype, device=theta.device).reshape(()) for s in samples])

def mse_loss(model, samples, theta):
    """Compute the mean squared correlation-energy error for one minibatch."""
    predictions = model.predict_batch(samples, theta)
    return torch.mean((predictions - targets(samples, theta)) ** 2)

def optimizer_step(model, batch, theta, optimizer, *, epoch):
    """Perform one unchanged Adam update and reject nonfinite loss or gradients."""
    optimizer.zero_grad(set_to_none=True)
    if hasattr(model, 'prepare_optimizer_step'):
        model.prepare_optimizer_step(theta, optimizer)
    loss = mse_loss(model, batch, theta)
    if not torch.isfinite(loss):
        raise FloatingPointError(f'Non-finite loss detected at epoch {epoch}: {loss.item()}')
    loss.backward()
    if hasattr(model, 'validate_gradients'):
        model.validate_gradients(theta)
    if hasattr(model, 'bind_optimizer_gradients'):
        model.bind_optimizer_gradients(theta, optimizer)
    optimizer.step()
    if hasattr(model, 'validate_gradients'):
        model.layout.validate(theta)
    return loss.detach()

@torch.no_grad()
def predict_many(model, samples, theta, *, batch_size):
    """Evaluate ordered minibatches and require finite, real predictions."""
    if not samples:
        raise ValueError('Cannot predict an empty sample sequence')
    chunks = [model.predict_batch(batch, theta).detach().cpu().numpy().reshape(-1) for batch in iter_batches(samples, batch_size)]
    result = _finite_real_array(np.concatenate(chunks), name='predictions')
    if result.shape != (len(samples),):
        raise ValueError('Unexpected prediction shape')
    return result

def _finite_real_array(values, *, name):
    """Validate numerical outputs before conversion can discard imaginary parts."""
    values = np.asarray(values)
    if not np.isrealobj(values):
        raise ValueError(f'{name} must contain real values')
    result = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError(f'Non-finite {name}')
    return result

def mae(y_true, y_pred):
    """Compute mean absolute error in the input energy units."""
    y = _finite_real_array(y_true, name='MAE targets')
    p = _finite_real_array(y_pred, name='MAE predictions')
    if y.ndim != 1 or p.shape != y.shape or not y.size:
        raise ValueError('Matching nonempty one-dimensional arrays required')
    value = float(np.mean(np.abs(y - p)))
    _finite_real_array(value, name='MAE')
    return value

@dataclass(frozen=True)
class Checkpoint:
    epoch: int
    train_mae_ha: float
    validation_mae_ha: float | None
    theta: torch.Tensor
    optimizer_state: dict

class BestCheckpoint:
    """Strict improvement of the selected MAE; preserve the earliest tied epoch."""

    def __init__(self, *, checkpoint_selection='validation'):
        """Initialize the instance-owned cache, random stream, or checkpoint state."""
        if checkpoint_selection not in ('validation', 'train'):
            raise ValueError("checkpoint_selection must be 'validation' or 'train'")
        self.checkpoint_selection = checkpoint_selection
        self.best_mae = float('inf')
        self.best_validation_mae = float('inf') if checkpoint_selection == 'validation' else None
        self.best_epoch = 0
        self.checkpoint = None

    def consider(self, *, epoch, train_mae, validation_mae=None, theta, optimizer):
        """Save parameters and state only when the selected MAE strictly improves."""
        metrics = [('train MAE', train_mae)]
        if self.checkpoint_selection == 'validation':
            if validation_mae is None:
                raise ValueError('Validation MAE is required for validation checkpoint selection')
            metrics.append(('validation MAE', validation_mae))
        elif validation_mae is not None:
            raise ValueError('Training checkpoint selection requires validation_mae=None')
        for name, value in metrics:
            if _finite_real_array(value, name=name).ndim != 0:
                raise ValueError(f'{name} must be a scalar')
        selected_mae = validation_mae if self.checkpoint_selection == 'validation' else train_mae
        if selected_mae < self.best_mae:
            self.best_mae = selected_mae
            if self.checkpoint_selection == 'validation':
                self.best_validation_mae = validation_mae
            self.best_epoch = epoch
            self.checkpoint = Checkpoint(epoch, train_mae, validation_mae, theta.detach().cpu().clone(), copy.deepcopy(optimizer.state_dict()))
            return True
        return False

    def restore(self, theta):
        """Restore the selected best parameters without restoring the optimizer."""
        if self.checkpoint is None:
            raise FileNotFoundError('Best checkpoint was not created')
        with torch.no_grad():
            theta.copy_(self.checkpoint.theta)
        return self.checkpoint

@dataclass(frozen=True)
class TrainingResult:
    theta: torch.Tensor
    checkpoint: Checkpoint
    history: tuple[dict, ...]
    final_optimizer_state: dict
    update_count: int

def train(model, train_samples, validation_samples, *, seed, num_epochs, batch_size=8, learning_rate=0.02, on_update=None, update_limit=None, checkpoint_selection='validation'):
    """Update on training samples and select checkpoints by the requested MAE.

    Validation selection requires validation samples. Explicit training selection
    requires an empty validation population and produces no validation metrics.
    No default epoch duration or output writer. Optional update_limit preflights
    a bounded verification run; it never silently truncates an epoch. on_update
    is an observation hook, not a scheduler or checkpoint-selection input.
    """
    if not train_samples:
        raise ValueError('Training set is empty')
    best = BestCheckpoint(checkpoint_selection=checkpoint_selection)
    if checkpoint_selection == 'validation' and not validation_samples:
        raise ValueError('Validation set is empty')
    if checkpoint_selection == 'train' and validation_samples:
        raise ValueError('Training checkpoint selection requires an empty validation set')
    if isinstance(num_epochs, bool) or not isinstance(num_epochs, (int, np.integer)) or num_epochs <= 0:
        raise ValueError('num_epochs must be a positive integer')
    if batch_size <= 0:
        raise ValueError('batch_size must be positive')
    planned = num_epochs * ((len(train_samples) + batch_size - 1) // batch_size)
    if update_limit is not None and planned > update_limit:
        raise ValueError('Requested trajectory exceeds explicit update_limit')
    theta = initialize_parameters(model, train_samples, seed=seed)
    optimizer = make_optimizer(theta, learning_rate=learning_rate, model=model)
    order = SampleOrder(seed)
    history = []
    count = 0
    for epoch in range(1, num_epochs + 1):
        indices = order.permutation(len(train_samples))
        for batch in iter_batches(train_samples, batch_size, indices):
            loss = optimizer_step(model, batch, theta, optimizer, epoch=epoch)
            count += 1
            if on_update is not None:
                on_update(epoch, batch, loss, theta, optimizer)
        train_predictions = predict_many(model, train_samples, theta, batch_size=batch_size)
        train_targets = np.asarray([s.target_energy for s in train_samples])
        train_mae = mae(train_targets, train_predictions)
        train_mae_mha = float(_finite_real_array(1000.0 * train_mae, name='train MAE in mHa'))
        row = dict(epoch=epoch, train_mae_ha=train_mae, train_mae_mha=train_mae_mha)
        validation_mae = None
        if checkpoint_selection == 'validation':
            validation_predictions = predict_many(model, validation_samples, theta, batch_size=batch_size)
            validation_targets = np.asarray([s.target_energy for s in validation_samples])
            validation_mae = mae(validation_targets, validation_predictions)
            validation_mae_mha = float(_finite_real_array(1000.0 * validation_mae, name='validation MAE in mHa'))
            row.update(validation_mae_ha=validation_mae, validation_mae_mha=validation_mae_mha)
        best.consider(epoch=epoch, train_mae=train_mae, validation_mae=validation_mae, theta=theta, optimizer=optimizer)
        history.append(row)
    checkpoint = best.restore(theta)
    return TrainingResult(theta, checkpoint, tuple(history), copy.deepcopy(optimizer.state_dict()), count)

# Evaluation and statistics across seeds

@dataclass(frozen=True)
class Evaluation:
    sample_ids: tuple
    predictions: np.ndarray
    targets: np.ndarray
    metrics: dict

def error_metrics(targets, predictions):
    """Historical one-dimensional Ha metrics. Reject empty/invalid inputs early."""
    y = _finite_real_array(targets, name='evaluation targets')
    p = _finite_real_array(predictions, name='evaluation predictions')
    if y.ndim != 1 or p.shape != y.shape or (not y.size):
        raise ValueError('Matching nonempty one-dimensional arrays required')
    e = p - y
    values = {'mae': float(np.mean(np.abs(y - p))), 'rmse': float(np.sqrt(np.mean((y - p) ** 2))), 'max_abs_error': float(np.max(np.abs(e))), 'signed_error_amplitude': float(np.max(e) - np.min(e))}
    metrics = {key + suffix: scale * value for key, value in values.items() for suffix, scale in (('_hartree', 1.0), ('_mHa', 1000.0))}
    _finite_real_array(list(metrics.values()), name='evaluation metrics')
    return metrics

def evaluate(model, samples, theta=None, *, checkpoint=None, batch_size=8):
    """Use exactly one supplied theta/checkpoint, preserving supplied row order."""
    if (theta is None) == (checkpoint is None):
        raise ValueError('Supply exactly one theta or checkpoint')
    chosen = checkpoint.theta if checkpoint is not None else theta
    p = predict_many(model, samples, chosen, batch_size=batch_size)
    y = _finite_real_array([s.target_energy for s in samples], name='evaluation targets')
    metrics = error_metrics(y, p)
    p.setflags(write=False)
    y.setflags(write=False)
    return Evaluation(tuple((s.sample_id for s in samples)), p, y, metrics)

def compute_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate seed MAEs into mean, sample SD, count, median, and standard error."""
    stats = df.groupby(['molecule', 'model_internal'])['test_MAE_mHa'].agg(mean='mean', sd='std', n='count', median='median')
    stats['se'] = stats['sd'] / np.sqrt(stats['n'])
    return stats
