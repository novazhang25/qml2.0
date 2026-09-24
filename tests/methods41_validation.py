"""Independent density contractions and exact fixed-core checks (no solvers)."""
from __future__ import annotations

import numpy as np

RDM_TOL = 1e-8


def maxabs(x):
    return float(np.max(np.abs(x), initial=0))


def check(out, key, residual, tolerance=RDM_TOL):
    value = maxabs(residual)
    out['checks'][key] = {'value': value, 'limit': tolerance, 'passed': value <= tolerance}


def contraction_loop(Gamma, S):
    """Literal sum_rs Gamma[p,q,r,s] S[r,s], independent of einsum."""
    n = len(S)
    result = np.zeros((n, n))
    for p in range(n):
        for q in range(n):
            for r in range(n):
                for s in range(n):
                    result[p, q] += Gamma[p, q, r, s] * S[r, s]
    return result


def pair_diagnostics(D, Gamma, S, N, norm_coefficient=1.0):
    n = len(S)
    if D.shape != (n,n) or Gamma.shape != (n,)*4:
        raise ValueError('RDM shape does not match the metric')
    if not all(np.isrealobj(x) and np.isfinite(x).all() for x in (D, Gamma, S)):
        raise ValueError('RDM/metric contains nonfinite or nonreal values')
    contracted = np.einsum('pqrs,rs->pq', Gamma, S)
    loop = contraction_loop(Gamma, S)
    rhs = (N - 1) * D.T
    electron_count = float(np.einsum('pq,qp', D, S))
    pair_count = float(np.einsum('pq,qp', contracted, S))
    electron_loop = sum(D[p,q] * S[q,p] for p in range(n) for q in range(n))
    pair_loop = sum(loop[p,q] * S[q,p] for p in range(n) for q in range(n))
    out = {'checks': {}, 'electron_number': electron_count, 'pair_number': pair_count,
           'expected_electron_number': N * norm_coefficient,
           'expected_pair_number': N * (N - 1) * norm_coefficient,
           'P_shape': list(D.shape), 'Gamma_shape': list(Gamma.shape),
           'P_frobenius_norm': float(np.linalg.norm(D)),
           'Gamma_frobenius_norm': float(np.linalg.norm(Gamma))}
    check(out, 'particle_number', electron_count - N * norm_coefficient)
    check(out, 'pair_number', pair_count - N * (N - 1) * norm_coefficient)
    check(out, 'contraction', contracted - rhs)
    check(out, 'contraction_explicit_loop', loop - rhs)
    check(out, 'contraction_loop_vs_einsum', loop - contracted)
    check(out, 'particle_number_loop_vs_einsum', electron_loop - electron_count)
    check(out, 'pair_number_loop_vs_einsum', pair_loop - pair_count)
    check(out, 'P_hermiticity', D - D.T)
    check(out, 'Gamma_hermiticity', Gamma - Gamma.transpose(1,0,3,2))
    check(out, 'Gamma_particle_exchange', Gamma - Gamma.transpose(2,3,0,1))
    out['passed'] = all(item['passed'] for item in out['checks'].values())
    out['failed_checks'] = [key for key, item in out['checks'].items() if not item['passed']]
    terms = {'lhs_einsum': contracted, 'lhs_loop': loop, 'rhs': rhs,
             'residual': contracted - rhs,
             'summands_pqrs': Gamma * S[None,None,:,:]}
    return out, terms


def frozen_core_diagnostics(D, Gamma, ncore):
    """Gamma = Gamma_active + Gamma_core + core/active direct-exchange terms.

    Closed-shell core is a determinant, so its expectation factors from the
    active wavefunction at every retained order. All arrays here use MO indices.
    """
    n = len(D)
    core = np.zeros_like(D)
    core[np.arange(ncore), np.arange(ncore)] = 2
    active = D.copy()
    active[:ncore,:] = 0
    active[:,:ncore] = 0
    gamma_core = np.einsum('pq,rs->pqrs', core, core) - .5*np.einsum('ps,rq->pqrs', core, core)
    mixed = (np.einsum('pq,rs->pqrs', core, active) + np.einsum('pq,rs->pqrs', active, core)
             - .5*np.einsum('ps,rq->pqrs', core, active) - .5*np.einsum('ps,rq->pqrs', active, core))
    # Entries with at least one core index are entirely known from D_active.
    mask = np.zeros((n,)*4, dtype=bool)
    mask[:ncore,:,:,:] = True
    mask[:,:ncore,:,:] = True
    mask[:,:,:ncore,:] = True
    mask[:,:,:,:ncore] = True
    expected = gamma_core + mixed
    loop_error = 0.
    for p,q,r,s in np.ndindex(Gamma.shape):
        if mask[p,q,r,s]:
            value = (core[p,q]*core[r,s] - .5*core[p,s]*core[r,q]
                     + core[p,q]*active[r,s] + active[p,q]*core[r,s]
                     - .5*core[p,s]*active[r,q] - .5*active[p,s]*core[r,q])
            loop_error = max(loop_error, abs(Gamma[p,q,r,s] - value))
    out = {'checks': {}, 'frozen_electron_number': float(np.trace(D[:ncore,:ncore]))}
    check(out, 'frozen_core_density', D - active - core)
    check(out, 'frozen_core_Gamma', (Gamma - expected)[mask])
    check(out, 'frozen_core_Gamma_explicit_loop', loop_error)
    return out


def dual_rdm_diagnostics(arrays, N, ncore):
    """Formal checks are separate from the non-vetoing standard PySCF audit."""
    out = {'checks': {}, 'consistent': {}, 'pyscf_audit': {}}
    terms_out = {}
    S, C = arrays['S_AO'], arrays['mo_coeff']
    for basis, metric in (('MO', np.eye(C.shape[1])), ('AO', S)):
        for track, suffix in (('consistent', 'consistent'), ('pyscf_audit', 'pyscf')):
            D = arrays[f'P_MP2_{basis}_{suffix}']
            G = arrays[f'Gamma_MP2_{basis}_{suffix}']
            try:
                report, terms = pair_diagnostics(D, G, metric, N)
            except (ValueError, TypeError) as exc:
                if track == 'consistent':
                    raise
                report, terms = {'passed': False, 'error': str(exc), 'checks': {},
                                 'P_shape': list(D.shape), 'Gamma_shape': list(G.shape)}, {}
            out[track][basis] = report
            if track == 'consistent':
                out['checks'].update({f'consistent_{basis}_{k}': v for k,v in report['checks'].items()})
            terms_out.update({f'{basis}_{suffix}_contraction_{key}': value for key,value in terms.items()})
        for order in range(3):
            D = arrays[f'P_{basis}_order{order}']
            G = arrays[f'Gamma_{basis}_order{order}']
            report, terms = pair_diagnostics(D, G, metric, N, float(order == 0))
            out['consistent'][f'{basis}_order{order}'] = report
            out['checks'].update({f'order{order}_{basis}_{k}': v for k,v in report['checks'].items()})
            terms_out.update({f'{basis}_order{order}_contraction_{key}': value for key,value in terms.items()})
    core = frozen_core_diagnostics(arrays['P_MP2_MO_consistent'], arrays['Gamma_MP2_MO_consistent'], ncore)
    out['checks'].update(core['checks'])
    out['consistent']['frozen_core'] = core
    D, G = arrays['P_MP2_AO_consistent'], arrays['Gamma_MP2_AO_consistent']
    P = arrays['P_AO']
    conventional = arrays['Lambda_conventional_AO']
    hfref = arrays['Lambda_HFref_AO']
    for name, tensor, expected in (
        ('hf_reference', hfref, (N-1)*(D-P).T),
        ('conventional', conventional, (.5*D @ S @ D-D).T)):
        lhs = np.einsum('pqrs,rs->pq', tensor, S)
        loop = contraction_loop(tensor, S)
        check(out, name + '_cumulant_contraction', lhs - expected)
        check(out, name + '_cumulant_contraction_loop', loop - expected)
        out['consistent'][name + '_cumulant'] = {
            'contraction_max': maxabs(lhs), 'expected_contraction_max': maxabs(expected),
            'residual_max': maxabs(lhs-expected), 'formula': '(N-1)*(D-P_HF).T' if name == 'hf_reference' else '(0.5*D@S@D-D).T'}
        terms_out[name + '_cumulant_contraction_lhs'] = lhs
        terms_out[name + '_cumulant_contraction_rhs'] = expected
        terms_out[name + '_cumulant_contraction_residual'] = lhs - expected
    check(out, 'P_consistent_order_sum', arrays['P_MP2_MO_consistent'] - sum(arrays[f'P_MO_order{i}'] for i in range(3)))
    check(out, 'Gamma_consistent_order_sum', arrays['Gamma_MP2_MO_consistent'] - sum(arrays[f'Gamma_MO_order{i}'] for i in range(3)))
    out['differences'] = {}
    for basis in ('MO','AO'):
        for kind in ('P','Gamma'):
            left, right = arrays[f'{kind}_MP2_{basis}_consistent'], arrays[f'{kind}_MP2_{basis}_pyscf']
            if right.shape != left.shape or not np.isrealobj(right) or not np.isfinite(right).all():
                out['differences'][kind + '_' + basis] = {'max_abs': None, 'frobenius_norm': None,
                                                         'error': 'Standard audit tensor is invalid; original tensor retained'}
            else:
                difference = left - right
                out['differences'][kind + '_' + basis] = {'max_abs': maxabs(difference), 'frobenius_norm': float(np.linalg.norm(difference))}
    out['passed'] = all(item['passed'] for item in out['checks'].values())
    return out, terms_out
