# Current Part 3 results

The current result uses `E_total_n = E_re + (n + 0.5)*hbar_omega` and the strict literal `E_total_n < 0` test for n=0 and n=1, without an energy shift.

**UNVALIDATED_ZERO_REFERENCE:** the saved E_re is absolute RHF total molecular energy including nuclear repulsion; the workflow does not establish zero Hartree as a physical allowed-vibration threshold. All nine molecules pass both sign tests and select n=1. None fails both.

- [Complete current table, units, reference warning and plots](../results/bond_length_part3/PART3_REPORT.md)
- [Full implementation audit and changed-file list](PART3_CORRECTION.md)
- [Full-precision CSV](../results/bond_length_part3/vibrational_levels.csv)
- [Current v2 JSON](../results/bond_length_part3/vibrational_levels.json)

The previous report is preserved only as [historical audit evidence](../results/part3_history/pre_zero_criterion/md/RHF_BOUND_LEVEL_SELECTION_REPORT.md). Its criterion is not used by current Part 3.
