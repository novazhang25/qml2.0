#!/usr/bin/env python3
"""Part 3: E_re + (n+1/2)*hbar*omega < 0, n=0,1, without an energy shift.

Reads Parts 1 and 2 only. No SCF, optimization, or harmonic recalculation.
Zero is not established as a physical vibrational threshold by this workflow;
the output distinguishes the requested sign test from physical binding.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "part3-zero-energy-v2"
MOLECULES = ("LiH", "BeH2", "H2O", "NH3", "N2", "CO", "HF", "H2S", "H2O2")
REFERENCE_WARNING = (
    "E_re is unshifted absolute PySCF RHF total molecular energy, including nuclear repulsion. "
    "The workflow does not define 0 Ha as the physical threshold for allowed vibrational states. "
    "These pass flags report only the requested E_total_n < 0 sign test, not established vibrational binding."
)
TABLE_COLUMNS = (
    ("molecule", "Molecule"), ("E_re_Ha", "E_re [Ha]"),
    ("k_Ha_per_Bohr2", "k [Ha/Bohr^2]"), ("mu_eff_amu", "mu_eff [amu]"),
    ("omega_rad_per_s", "omega [rad/s]"), ("hbar_omega_Ha", "hbar_omega [Ha]"),
    ("E_n0_Ha", "E_n0 [Ha]"), ("E_n1_Ha", "E_n1 [Ha]"),
    ("E_total_n0_Ha", "E_total_n0 [Ha]"), ("E_total_n1_Ha", "E_total_n1 [Ha]"),
    ("n0_pass", "n0_pass"), ("n1_pass", "n1_pass"), ("selected_n", "selected_n"),
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value, name, positive=False):
    number = float(value)
    require(math.isfinite(number) and (not positive or number > 0),
            f"{name} must be finite" + (" and positive." if positive else "."))
    return number


def close(actual, expected, name, *, rtol=1e-10, atol=1e-13):
    require(math.isclose(finite(actual, name), finite(expected, name), rel_tol=rtol, abs_tol=atol),
            f"Inconsistent {name}: {actual!r} versus {expected!r}.")


def compute_levels(E_re_Ha, hbar_omega_Ha):
    """Apply strict <0 to the supplied Hartree values, without shifts/tolerances."""
    re = finite(E_re_Ha, "E_re_Ha")
    quantum = finite(hbar_omega_Ha, "hbar_omega_Ha", True)
    en0 = finite(0.5 * quantum, "E_n0_Ha", True)
    en1 = finite(1.5 * quantum, "E_n1_Ha", True)
    e0, e1 = finite(re + en0, "E_total_n0_Ha"), finite(re + en1, "E_total_n1_Ha")
    n0_pass, n1_pass = e0 < 0.0, e1 < 0.0
    selected = 1 if n1_pass else 0 if n0_pass else None
    return {"E_re_Ha": re, "hbar_omega_Ha": quantum, "E_n0_Ha": en0, "E_n1_Ha": en1,
            "E_total_n0_Ha": e0, "E_total_n1_Ha": e1, "n0_pass": n0_pass, "n1_pass": n1_pass,
            "selected_n": selected, "decision_criterion": "E_total_n < 0 Ha", "reference_shift_Ha": 0.0,
            "energy_reference_status": "UNVALIDATED_ZERO_REFERENCE", "energy_reference_warning": REFERENCE_WARNING,
            "physical_vibrational_binding_established": False,
            "validation_status": "PASS" if selected is not None else "NO_LEVEL_PASSES_ZERO_TEST"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def indexed(rows, summary=False):
    result = {}
    for row in rows:
        name = (row["summary"] if summary else row)["molecule"]
        require(name not in result, f"Duplicate molecule: {name}")
        result[name] = row
    return result


def analyze_molecule(name, metadata, harmonic, constants):
    """Consume Parts 1/2 unchanged. Recomputed unit expressions are checks only."""
    hs = harmonic["summary"]
    require(metadata["molecule"] == hs["molecule"] == name, "Molecule mismatch.")
    require(metadata["method"] == "RHF" and metadata["status"] == "converged"
            and metadata["optimizer_success"].lower() == "true", "Part 1 equilibrium is not converged RHF.")
    require(hs["validation_status"] == "PASS" and not harmonic.get("failures")
            and harmonic["finite_differences"]["convergence_pass"] is True, "Part 2 harmonic validation failed.")
    require(metadata["basis"].lower() == hs["RHF_basis"].lower(), "RHF basis mismatch.")
    re = finite(metadata["energy_hartree"], "Part 1 energy_hartree")
    gradient = finite(metadata["max_gradient_hartree_per_bohr"], "Part 1 nuclear gradient")
    require(0 <= gradient <= 1e-6, "Part 1 nuclear gradient exceeds the existing limit.")
    close(re, harmonic["input"]["metadata_row"]["energy_hartree"], "saved optimized energy", rtol=0, atol=1e-12)
    close(re, harmonic["stationarity"]["E_RHF_Eh"], "Part 2 stationary energy", rtol=0, atol=1e-9)
    k = finite(hs["k_q_Eh_per_Bohr2"], "saved k", True)
    mu = finite(hs["mu_eff_amu"], "saved effective mass", True)
    omega = finite(hs["omega_rad_per_s"], "saved angular frequency", True)
    c = {key: finite(constants[key], key, True)
         for key in ("Bohr_A", "Bohr_m", "Hartree_J", "amu_kg", "hbar_J_s", "c_m_per_s")}
    close(c["Bohr_m"], c["Bohr_A"] * 1e-10, "Bohr conversion", atol=0)
    omega_check = math.sqrt((k * c["Hartree_J"] / c["Bohr_m"]**2) / (mu * c["amu_kg"]))
    close(omega, omega_check, "omega versus saved k/mass", rtol=1e-10, atol=0)
    close(hs["k_q_Eh_per_A2"], k / c["Bohr_A"]**2, "curvature Angstrom conversion", atol=0)
    close(hs["k_q_N_per_m"], k * c["Hartree_J"] / c["Bohr_m"]**2, "curvature SI conversion", atol=0)
    close(hs["mu_eff_kg"], mu * c["amu_kg"], "effective mass conversion", atol=0)
    # The saved omega is used directly, not replaced by omega_check.
    quantum = c["hbar_J_s"] * omega / c["Hartree_J"]
    from_wavenumber = (2 * math.pi * c["hbar_J_s"] * c["c_m_per_s"] * 100
                       * finite(hs["harmonic_frequency_cm1"], "saved wavenumber", True) / c["Hartree_J"])
    close(quantum, from_wavenumber, "Hartree quantum versus wavenumber", atol=1e-15)
    values = compute_levels(re, quantum)
    values.update(molecule=name, basis=hs["RHF_basis"], coordinate=hs["coordinate_definition"],
                  r_e_A=hs["s_eq_A"], k_Ha_per_Bohr2=k, k_Ha_per_A2=hs["k_q_Eh_per_A2"],
                  k_N_per_m=hs["k_q_N_per_m"], mu_eff_amu=mu, mu_eff_kg=hs["mu_eff_kg"], omega_rad_per_s=omega)
    for n in (0, 1):
        for key in (f"Delta_q_n{n}_A", f"n{n}_min_A", f"n{n}_max_A"):
            values[key] = finite(hs[key], key, True)
        close(0.5 * values["k_Ha_per_A2"] * values[f"Delta_q_n{n}_A"]**2,
              values[f"E_n{n}_Ha"], f"n={n} saved turning point", rtol=1e-9)
    n = values["selected_n"]
    values["selected_range_min_A"] = None if n is None else values[f"n{n}_min_A"]
    values["selected_range_max_A"] = None if n is None else values[f"n{n}_max_A"]
    return {"summary": values, "input_validation": {
        "passed": True, "energy_source": "Part 1 CSV energy_hartree, unchanged",
        "frequency_source": "Part 2 summary omega_rad_per_s, unchanged",
        "geometry_sha256": harmonic["input"]["xyz_sha256"],
        "part2_stationary_energy_difference_Ha": harmonic["stationarity"]["E_RHF_Eh"] - re,
        "omega_from_k_mu_check_rad_per_s": omega_check, "omega_relative_error": abs(omega - omega_check) / omega,
        "hbar_omega_from_wavenumber_Ha": from_wavenumber,
        "quantum_conversion_residual_Ha": abs(quantum - from_wavenumber), "reference_shift_Ha": 0.0,
    }}


def display(value):
    if isinstance(value, bool):
        return "YES" if value else "NO"
    if value is None:
        return "None"
    return f"{value:.12g}" if isinstance(value, float) else str(value).replace("|", "\\|")


def table(rows, columns=TABLE_COLUMNS):
    lines = ["| " + " | ".join(label for _, label in columns) + " |",
             "| " + " | ".join("---" for _ in columns) + " |"]
    lines.extend("| " + " | ".join(display(row[key]) for key, _ in columns) + " |" for row in rows)
    return "\n".join(lines)


def report_text(document):
    rows = [r["summary"] for r in document["results"]]
    neither = [r["molecule"] for r in rows if r["selected_n"] is None]
    lines = ["# Part 3: explicit zero-energy sign test", "", "## Energy-reference audit", "",
             "**UNVALIDATED_ZERO_REFERENCE.** " + REFERENCE_WARNING, "",
             "Part 1 stores the BFGS objective in energy_hartree. That objective is converged PySCF RHF "
             "mf.e_tot; installed scf.hf.energy_tot adds energy_elec(...)[0] and energy_nuc(). "
             "It is not an already referenced well energy. Part 3 reads the CSV value directly without an offset.", "",
             "The old Part 3 tested the total level against a large-separation plateau. That selector and its "
             "threshold inputs have been removed. Old artifacts exist only as explicitly historical evidence "
             "under results/part3_history/pre_zero_criterion; they are not inputs to this analysis.", "",
             "## Calculation and exact criterion", "", "```text",
             "hbar_omega [Ha] = hbar_J_s * saved_omega_rad_per_s / Hartree_J",
             "E_n0 [Ha] = 0.5 * hbar_omega", "E_n1 [Ha] = 1.5 * hbar_omega",
             "E_total_n0 [Ha] = saved_E_re + E_n0", "E_total_n1 [Ha] = saved_E_re + E_n1",
             "n0_pass = E_total_n0 < 0", "n1_pass = E_total_n1 < 0",
             "selected_n = 1 if n1_pass else 0 if n0_pass else None", "```", "",
             "Equality at zero does not pass. The literal zero refers to the unchanged input energy scale. "
             "PASS denotes successful numerical/input validation, not resolution of the reference warning.", "",
             "## Complete table", "", table(rows), "",
             "Energy columns are Hartree; k is Ha/Bohr^2, mass amu and omega rad/s. CSV/JSON retain full precision.", "",
             "## Unit checks", "",
             "Existing k, mu_eff and omega are reused unchanged. Independent checks compare saved omega with "
             "sqrt((k*Hartree_J/Bohr_m^2)/(mu_eff*amu_kg)), and compare the quantum with "
             "2*pi*hbar*c*100*wavenumber/Hartree_J. Checked values never replace the saved inputs.", ""]
    checks = [("molecule", "Molecule"), ("omega_relative_error", "omega relative error"),
              ("quantum_conversion_residual_Ha", "quantum conversion residual [Ha]")]
    lines += [table([{**r["input_validation"], "molecule": r["summary"]["molecule"]}
                     for r in document["results"]], checks), "", "## Results", "",
              "Neither level passes: " + (", ".join(neither) if neither else "none of the listed molecules") + ".", ""]
    for row in rows:
        lines.append(f"- {row['molecule']}: n0_pass={display(row['n0_pass'])}, "
                     f"n1_pass={display(row['n1_pass'])}, selected_n={display(row['selected_n'])}.")
    lines += ["", "## Plots", ""] + [f"- [{name}]({name})" for name in document["plot_files"]]
    if not document["plot_files"]:
        lines.append("Plots disabled with --no-plots.")
    lines += ["", "## Provenance", "", "```json", json.dumps(document["provenance"], indent=2), "```", ""]
    return "\n".join(lines)


def write_text(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=PROJECT / "results/rhf_geometries/rhf_equilibrium_summary.csv")
    parser.add_argument("--harmonic-json", type=Path, default=PROJECT / "json/new_rhf_harmonic_bond_ranges.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "results/bond_length_part3")
    parser.add_argument("--molecules", nargs="+", choices=MOLECULES, default=list(MOLECULES))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    try:
        require(len(args.molecules) == len(set(args.molecules)), "Duplicate requested molecules.")
        with args.metadata.open(newline="", encoding="utf-8") as handle:
            metadata = indexed(csv.DictReader(handle))
        harmonic_doc = json.loads(args.harmonic_json.read_text())
        harmonic = indexed(harmonic_doc["results"], summary=True)
        constants = harmonic_doc["runtime"]["constants"]
        require(all(harmonic[name]["input"]["metadata_sha256"] == sha256(args.metadata) for name in args.molecules),
                "Part 1 metadata differs from the input used for Part 2.")
        results = [analyze_molecule(name, metadata[name], harmonic[name], constants) for name in args.molecules]
        document = {
            "schema_version": SCHEMA_VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
            "decision_criterion": "E_total_n < 0 Ha", "energy_reference_status": "UNVALIDATED_ZERO_REFERENCE",
            "energy_reference_warning": REFERENCE_WARNING,
            "provenance": {"source_files_sha256": {str(p.resolve()): sha256(p) for p in (args.metadata, args.harmonic_json)},
                           "script_sha256": sha256(__file__), "constants": constants,
                           "equilibrium_reoptimized": False, "force_constant_recomputed": False,
                           "frequency_recomputed": False, "SCF_recomputed": False, "reference_shift_Ha": 0.0},
            "results": results, "plot_files": [],
        }
        if not args.no_plots:
            from plots.bond_length_part3_plots import write_plots
            document["plot_files"] = write_plots(results, args.output_dir)
            document["provenance"]["plot_script_sha256"] = sha256(Path(__file__).parent / "plots" / "bond_length_part3_plots.py")
        rows = [r["summary"] for r in results]
        fields = [key for key, _ in TABLE_COLUMNS]
        fields += sorted(set().union(*(row.keys() for row in rows)) - set(fields))
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: display(value) if isinstance(value, bool) or value is None else value
                          for key, value in row.items()} for row in rows)
        write_text(args.output_dir / "vibrational_levels.csv", stream.getvalue())
        write_text(args.output_dir / "vibrational_levels.json", json.dumps(document, indent=2, allow_nan=False) + "\n")
        write_text(args.output_dir / "PART3_REPORT.md", report_text(document))
        print(REFERENCE_WARNING + "\n\n" + table(rows))
        print(f"\nPart 3 outputs: {args.output_dir.resolve()}")
        return 0
    except (OSError, ValueError, KeyError, TypeError, ArithmeticError, ImportError) as exc:
        print(f"Part 3 failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
