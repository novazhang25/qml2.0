"""Optimize nine neutral, closed-shell molecules and write one CSV row each.
All Cartesian coordinates are optimized without
symmetry constraints using analytic RHF gradients. These are electronic
RHF equilibrium geometries, without vibrational or thermal corrections.
No frequency calculation is performed to certify the stationary points.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from pyscf import gto, lib, scf
from pyscf.lib.parameters import BOHR
from scipy.optimize import minimize


def bent(center, length, angle):
    half = np.deg2rad(angle / 2)
    return [(center, (0, 0, 0)),
            ("H", (length * np.sin(half), 0, length * np.cos(half))),
            ("H", (-length * np.sin(half), 0, length * np.cos(half)))]


def ammonia():
    # Pyramidal starting geometry; all atoms subsequently move freely.
    return [("N", (0, 0, 0))] + [
        ("H", (0.94 * np.cos(t), 0.94 * np.sin(t), 0.38))
        for t in np.deg2rad([0, 120, 240])
    ]


def peroxide(torsion):
    angle = np.deg2rad(100)
    phi = np.deg2rad(torsion)
    r, d = 0.97, 1.45
    return [("O", (0, 0, 0)), ("O", (0, 0, d)),
            ("H", (r * np.sin(angle), 0, r * np.cos(angle))),
            ("H", (r * np.sin(angle) * np.cos(phi),
                   r * np.sin(angle) * np.sin(phi), d - r * np.cos(angle)))]


MOLECULES = {
    "LiH": [[("Li", (0, 0, 0)), ("H", (0, 0, 1.60))]],
    "BeH2": [[("Be", (0, 0, 0)), ("H", (0, 0, -1.33)),
              ("H", (0, 0, 1.33))]],
    "H2O": [bent("O", 0.96, 104.5)],
    "NH3": [ammonia()],
    "N2": [[("N", (0, 0, 0)), ("N", (0, 0, 1.10))]],
    "CO": [[("C", (0, 0, 0)), ("O", (0, 0, 1.13))]],
    "HF": [[("H", (0, 0, 0)), ("F", (0, 0, 0.92))]],
    "H2S": [bent("S", 1.34, 92)],
    "H2O2": [peroxide(t) for t in (70, 110, 150)],
}

# Atom indices match the definitions above; each equivalent bond is reported.
BONDS = {
    "LiH": [(0, 1)], "BeH2": [(0, 1), (0, 2)],
    "H2O": [(0, 1), (0, 2)], "NH3": [(0, 1), (0, 2), (0, 3)],
    "N2": [(0, 1)], "CO": [(0, 1)], "HF": [(0, 1)],
    "H2S": [(0, 1), (0, 2)], "H2O2": [(0, 1), (0, 2), (1, 3)],
}
ANGLES = {
    "BeH2": [(1, 0, 2)], "H2O": [(1, 0, 2)],
    "NH3": [(1, 0, 2), (1, 0, 3), (2, 0, 3)],
    "H2S": [(1, 0, 2)], "H2O2": [(2, 0, 1), (0, 1, 3)],
}
FIELDS = ["molecule", "method", "basis", "status", "energy_hartree",
          "bond_lengths_angstrom", "bond_angles_deg", "HOOH_dihedral_deg",
          "max_gradient_hartree_per_bohr", "optimizer_success",
          "iterations", "energy_gradient_evaluations", "selected_start",
          "xyz_file", "message"]


def optimize(atoms, args, label):
    symbols = [atom[0] for atom in atoms]
    initial = np.array([atom[1] for atom in atoms], dtype=float) / BOHR
    initial -= initial.mean(axis=0)
    evaluations = 0

    def energy_gradient(flat):
        nonlocal evaluations
        coords = flat.reshape(-1, 3)
        mol = gto.M(atom=list(zip(symbols, coords)), basis=args.basis,
                    unit="Bohr", charge=0, spin=0, symmetry=False,
                    verbose=0, max_memory=args.memory_mb)
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-14
        mf.conv_tol_grad = 1e-10
        mf.max_cycle = 200
        mf.diis_space = 12
        mf.direct_scf_tol = 1e-14
        energy = mf.kernel()
        if not mf.converged:
            raise RuntimeError("RHF SCF did not converge")
        gradient = mf.nuc_grad_method().kernel()
        if not np.isfinite(energy) or not np.all(np.isfinite(gradient)):
            raise RuntimeError("Non-finite energy or gradient")
        evaluations += 1
        print(f"{label}: evaluation {evaluations:4d}  "
              f"E={energy:.14f} Eh  max|g|={np.max(np.abs(gradient)):.3e}",
              flush=True)
        return float(energy), gradient.ravel()

    result = minimize(energy_gradient, initial.ravel(), jac=True, method="BFGS",
                      options={"gtol": args.gtol, "maxiter": args.maxiter})
    max_gradient = float(np.max(np.abs(result.jac)))
    return {"result": result, "symbols": symbols,
            "coords": result.x.reshape(-1, 3) * BOHR,
            "gradient": max_gradient, "evaluations": evaluations,
            "converged": bool(max_gradient <= args.gtol)}


def measurements(name, symbols, xyz):
    def label(indices):
        return "-".join(f"{symbols[i]}{i + 1}" for i in indices)

    bonds = {label((i, j)): float(np.linalg.norm(xyz[i] - xyz[j]))
             for i, j in BONDS[name]}
    angles = {}
    for i, j, k in ANGLES.get(name, []):
        a, b = xyz[i] - xyz[j], xyz[k] - xyz[j]
        cosine = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        angles[label((i, j, k))] = float(np.rad2deg(np.arccos(np.clip(cosine, -1, 1))))
    dihedral = ""
    if name == "H2O2":
        axis = xyz[1] - xyz[0]
        axis /= np.linalg.norm(axis)
        a, b = xyz[2] - xyz[0], xyz[3] - xyz[1]
        a -= np.dot(a, axis) * axis
        b -= np.dot(b, axis) * axis
        dihedral = float(np.rad2deg(np.arctan2(
            np.dot(np.cross(axis, a), b), np.dot(a, b))))
    return json.dumps(bonds, sort_keys=True), json.dumps(angles, sort_keys=True), dihedral


def save_csv(path, rows):
    # Checkpoint after each molecule so completed results survive a later failure.
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--output", type=Path, default=Path("rhf_equilibrium_summary.csv"))
    parser.add_argument("--geometry-dir", type=Path, default=Path("rhf_geometries"))
    parser.add_argument("--gtol", type=float, default=1e-6,
                        help="Maximum Cartesian gradient component in Eh/Bohr")
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--memory-mb", type=int, default=4000)
    parser.add_argument("--threads", type=int, default=None)
    args = parser.parse_args()
    if args.gtol <= 0 or args.maxiter < 1 or args.memory_mb < 1:
        parser.error("gtol, maxiter, and memory-mb must be positive")
    if args.threads is not None:
        if args.threads < 1:
            parser.error("threads must be positive")
        lib.num_threads(args.threads)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.geometry_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, starts in MOLECULES.items():
        print(f"\n===== {name}: RHF/{args.basis} =====", flush=True)
        row = {"molecule": name, "method": "RHF", "basis": args.basis}
        candidates, errors = [], []
        for index, atoms in enumerate(starts, 1):
            try:
                candidate = optimize(atoms, args, f"{name} start {index}")
                candidate["start"] = index
                candidates.append(candidate)
            except Exception as exc:
                message = f"Start {index}: {type(exc).__name__}: {exc}"
                errors.append(message)
                print(message, flush=True)
        if candidates:
            converged = [c for c in candidates if c["converged"]]
            best = min(converged or candidates, key=lambda c: c["result"].fun)
            result = best["result"]
            bonds, angles, torsion = measurements(name, best["symbols"], best["coords"])
            status = "converged" if best["converged"] else "not_converged"
            xyz_path = args.geometry_dir / f"{name}_{status}.xyz"
            with xyz_path.open("w", encoding="utf-8") as handle:
                handle.write(f"{len(best['symbols'])}\n{name} RHF/{args.basis} "
                             f"{status} E={result.fun:.15f} Eh\n")
                for symbol, coord in zip(best["symbols"], best["coords"]):
                    handle.write(f"{symbol:2s} {coord[0]: .12f} "
                                 f"{coord[1]: .12f} {coord[2]: .12f}\n")
            notes = [str(result.message)] + errors
            for candidate in candidates:
                notes.append(f"Start {candidate['start']}: "
                             f"converged={candidate['converged']}, "
                             f"energy={candidate['result'].fun:.14f} Eh")
            row.update(status=status, energy_hartree=f"{result.fun:.15f}",
                       bond_lengths_angstrom=bonds, bond_angles_deg=angles,
                       HOOH_dihedral_deg=torsion,
                       max_gradient_hartree_per_bohr=best["gradient"],
                       optimizer_success=bool(result.success), iterations=result.nit,
                       energy_gradient_evaluations=sum(c["evaluations"] for c in candidates),
                       selected_start=best["start"], xyz_file=str(xyz_path.resolve()),
                       message="; ".join(notes))
        else:
            row.update(status="failed", message="; ".join(errors))
        rows.append(row)
        save_csv(args.output, rows)
        print(f"{name}: {row['status']}; summary saved to {args.output}", flush=True)
    failures = sum(row["status"] != "converged" for row in rows)
    print(f"\nCompleted {len(rows)} molecules; {failures} failed or unconverged.")
    print(f"Summary: {args.output.resolve()}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
