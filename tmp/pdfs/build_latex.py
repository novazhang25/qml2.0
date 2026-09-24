from pathlib import Path
import re
root=Path('/Users/novaz/Desktop/qml2.0')
src=(root/'output/code_audit_20260924/CODE_AUDIT.md').read_text()
rows=[]
for line in src.splitlines():
 if re.match(r'\| \d+ ',line): rows.append([s.strip() for s in line.strip('|').split('|')])
entries=[
(r'Cartesian coordinates; mass-weighted proper alignment follows the raw stretch. The scan includes both endpoints and contains 30 geometries. Harmonic constants use consistent SI quantities before conversion to angstrom.',r'k=\sum_{a,b,x,y}t_{ax}H_{abxy}t_{by},\quad \mu=\sum_{a,x}m_a t_{ax}^{2},\quad \omega=\sqrt{k/\mu},\\ \Delta=\sqrt{\frac{3\hbar}{\mu\omega}},\qquad s_j=s_e-\Delta+\frac{2j\Delta}{29},\quad j=0,\ldots,29.'),
('Full spherical-AO overlap from the molecular geometry, basis, charge and spin. Integral evaluation is delegated to PySCF. Dimensionless matrix; no active restriction here.',r'S\in\mathbb R^{M\times M}.'),
('Initial-data RHF checks convergence and internal stability. Descriptor RHF is recomputed using the saved AO density as an initial guess. Density is spin summed; the Fock matrix is in Hartree. SCF and integral internals remain library boundaries.',r'P=C\,\operatorname{diag}(n_p)C^{\mathsf T},\qquad F=h+J[P]-\tfrac12K[P].'),
('FCI frozen MOs and excluded descriptor AOs define separate restrictions. MP2 and RDM construction freeze no orbitals. Descriptor exclusion uses AO/shell metadata; H2S retains 10 AOs.',r'\text{Excluded descriptor AO:}\quad Z>2\ \land\ \text{principal shell}=1.\\ \text{Descriptor MP2: frozen}=0.'),
('Restricted MP2 uses all occupied and virtual spatial MOs. Amplitude and energy construction is delegated to PySCF; saved-array amplitude checks agree to roundoff. No unaudited library-internal amplitude formula is asserted.',r't_2\in\mathbb R^{n_{\mathrm{occ}}\times n_{\mathrm{occ}}\times n_{\mathrm{virt}}\times n_{\mathrm{virt}}}.'),
('The determinant reference and first-order correction use zero singles and restricted doubles. Apply the following algebra separately to the one- and two-particle RDMs. Determinant conversion and RDM APIs are external boundaries; normalized-state diagnostics do not replace this algebra.',r'w=\langle\chi|\chi\rangle,\quad R_0=R[\Phi],\\ R_1=R[\Phi+\chi]-R[\Phi]-R[\chi],\quad R_2=R[\chi]-wR_0,\\ R=R_0+R_1+R_2.'),
('Full rank-four arrays preserve raw PySCF index ordering. Only triple-descriptor extraction performs the displayed axis permutation. There is no explicit four-spin-block pipeline.',r'\Lambda_{pqrs}\ \xrightarrow{\ \operatorname{transpose}(0,2,1,3)\ }\ \widetilde\Lambda_{prqs}.'),
('MO-to-AO transformations use real arrays and ordinary transpose, without conjugation. Every axis spans the full space.',r'D_{\mathrm{AO}}=CD_{\mathrm{MO}}C^{\mathsf T},\\ G_{abcd}=\sum_{pqrs}C_{ap}C_{bq}C_{cr}C_{ds}G_{pqrs}.'),
('Select AOs only after full-space transformations. The producer saves source-ascending indices; the loader rebuilds canonical atom/AO order. Local retained indices and full source indices are distinct.',r'v_i:\ i\mapsto\text{full source AO index},\qquad A^{\mathrm{sel}}_{ij}=A_{v_i v_j}.\\ v_{\mathrm{BeH_2,model}}=[5,1,2,3,4,6]\quad\text{(zero-based)}.'),
('Disconnected reference uses the full RHF spin-summed density. The displayed indices are free indices, not summation indices.',r'(G_0)_{pqrs}=P_{pq}P_{rs}-\tfrac12P_{ps}P_{rq}.'),
('The reference subtraction is performed in the full AO space. The conventional subtraction using correlated density is separately stored and is not the descriptor input.',r'\Lambda=G_{\mathrm{consistent}}-G_0[P_{\mathrm{RHF}}],\qquad \Lambda\in\mathbb R^{M\times M\times M\times M}.'),
('No additional all-ones contraction of the fourth axis is executed. Raw RDMs, order components and the alternate cumulant are stored diagnostics. There is no additional formula for an absent stage.',None),
('Canonical-RHF CASCI defines the active determinant space. Active one-electron and pair-packed two-electron integrals are saved but not consumed by descriptor production. Core energy includes nuclear and frozen-core terms. CASCI solving remains an external boundary.',r'p\in[n_{\mathrm{core}},M),\quad n_{\mathrm{cas}}=M-n_{\mathrm{core}},\quad N_{\mathrm{cas}}=N_e-2n_{\mathrm{core}},\\ E_{\mathrm{FCI}}=E_{\mathrm{active}}+E_{\mathrm{core}},\quad y=E_{\mathrm{FCI}}-E_{\mathrm{RHF,input}},\\ \operatorname{shape}(\mathrm{CI})=\left(\binom{n_{\mathrm{cas}}}{N_{\mathrm{cas}}/2},\binom{n_{\mathrm{cas}}}{N_{\mathrm{cas}}/2}\right).'),
('Full-space overlap powers precede selection. The producer applies no threshold or clipping. The loader requires the minimum eigenvalue to exceed the stated threshold and checks matrix identities.',r'S=U\operatorname{diag}(s)U^{\mathsf T},\quad H=U\operatorname{diag}(\sqrt{s})U^{\mathsf T},\\ J=U\operatorname{diag}(s^{-1/2})U^{\mathsf T},\qquad \min(s)>10^{-8}\ \text{(loader)}.'),
('Full Löwdin-AO matrices. No explicit transpose averaging occurs at this stage. The density is the RHF density, not the correlated density.',r'P_L=HP_{\mathrm{RHF}}H,\qquad F_L=JF_{\mathrm{AO}}J,\qquad P_L,F_L\in\mathbb R^{M\times M}.'),
('All four axes are transformed in the full space before selecting AOs. The same overlap square root is used on every axis.',r'(\Lambda_L)_{abcd}=\sum_{pqrs}H_{ap}H_{bq}H_{cr}H_{ds}(\Lambda_{\mathrm{AO}})_{pqrs}.'),
('The repeated local index is not contracted. A full triple array and its strict-triple vector are saved. Current Run 1 constructs no triple input and does not consume these arrays.',r'T_{ijk}=(\Lambda_L)_{v_i v_k v_j v_k}\quad\text{(no sum over }k\text{)},\\ \operatorname{shape}(T_{\mathrm{full}})=(n,n,n),\qquad \operatorname{shape}(T_{i<j<k})=\left(\binom n3\right).'),
('Saved diagonal and pair features use the producer ordering. The loader validates them and rebuilds canonical density and Fock matrices. FE and FE-prime use ascending eigenvalues of the selected blocks; density is dimensionless and Fock quantities are in Hartree.',r'P_i=(P_L)_{v_i v_i},\quad F_i=(F_L)_{v_i v_i},\quad P_{ij}=(P_L)_{v_i v_j},\ i<j,\\ x_{\mathrm{FE}}=\operatorname{eigvalsh}(P_L^{\mathrm{sel}}),\qquad x_{\mathrm{FE\_prime}}=\operatorname{eigvalsh}(F_L^{\mathrm{sel}}).'),
('Exactly 30 records per molecule, with IDs 001 through 030. Incomplete or erroneous records are rejected; there is no cutoff filtering. Traversal positions are zero based.',r'\operatorname{sort\ key}=(q_{\mathrm{A}},\ \text{geometry index}).'),
('The default manifest requires explicit disjoint train, validation and test sets covering every geometry. In the explicit 15-15 protocol, membership follows geometry ID rather than sorted traversal position.',r'\mathcal I_{\mathrm{train}}=\{1,3,\ldots,29\},\qquad \mathcal I_{\mathrm{test}}=\{2,4,\ldots,30\}\quad\text{(15-15)}.'),
('MB scaling pools every training geometry and selected AO, separately for density and Fock diagonals. Population standard deviation is used. Positive finite standard deviations are required; there is no epsilon floor or clipping.',r'K=N_{\mathrm{train}}n,\quad \mu_x=\frac1K\sum_{a=1}^Kx_a,\quad \sigma_x=\sqrt{\frac1K\sum_{a=1}^K(x_a-\mu_x)^2},\\ x_a^{\mathrm{std}}=\frac{x_a-\mu_x}{\sigma_x}\qquad (\mathrm{ddof}=0).'),
('Local pairs follow row-major strict-upper ordering; triples follow nested lexicographic ordering. Pair values feed scale fitting. The triple helper is unused by Run 1.',r'0\le i<j<n,\quad N_{\mathrm{pair}}=\binom n2;\qquad 0\le i<j<k<n,\quad N_{\mathrm{triple}}=\binom n3.'),
('The percentile uses linear interpolation of pooled absolute entries, including zeros. Empty or nonpositive scales reject. MB-1 fits the pair scale even though it has no pair gates. The triple scale belongs only to the MB-3 helper.',r'q_{95}=\operatorname{percentile}_{95}(|x|),\qquad \alpha=\frac{\operatorname{atanh}(0.9)}{q_{95}}.\\ \beta\text{ is fitted analogously for triple inputs.}'),
('MB shares three trainable encoding scalars across wires. Fingerprints instead use training-only per-column statistics. For fingerprint columns, replace near-zero standard deviations by one, explicitly zero those columns after scaling, and then clip. RY uses the library half-angle convention.',r'\theta_i^{\mathrm{MB}}=\frac\pi2\tanh(aP_i^{\mathrm{std}}+bF_i^{\mathrm{std}}+c),\\ m_j=[\sigma_j<10^{-12}],\quad \sigma_j^{\mathrm{safe}}=\begin{cases}1&m_j,\\\sigma_j&\text{otherwise},\end{cases}\\ u_j=\begin{cases}0&m_j,\\\frac\pi2\frac{x_j-\mu_j}{\sigma_j^{\mathrm{safe}}}&\text{otherwise},\end{cases}\qquad \theta_j=\operatorname{clip}(u_j,-\pi,\pi).'),
('Pair encoding exists for helper models. MB-1 supplies no pair inputs and queues no pair gates; this is not an active Run 1 stage.',r'\operatorname{IsingZZ}\!\left(\frac\pi2\tanh(\alpha P_{ij})\right),\qquad i<j.'),
('Triple encoding exists as a helper. No current Run 1 method supplies triple inputs, and triple-model runtime results were not numerically verified.',r'\operatorname{MultiRZ}\!\left(\frac\pi2\tanh(\beta T_{ijk})\right),\qquad i<j<k.'),
('There is a single encoding followed by two shared-parameter ansatz layers. Each layer applies the same RY angle to all wires, then sequential ring CNOTs in ascending wire order, with the wrap edge last. For two wires, both directed edges execute.',r'L=2:\qquad RY_q(\phi_\ell)\ \text{for every }q;\quad \operatorname{CNOT}_{q\to(q+1)\bmod n},\ q=0,\ldots,n-1.'),
('Measure Pauli-Z on every physical wire in wire order. The output is dimensionless. Simulation and autodifferentiation are delegated; independent statevector predictions were checked, but no separate direct-runtime Z-vector residual was recorded.',r'z_q=\langle\psi|Z_q|\psi\rangle,\qquad z\in\mathbb R^n.'),
('Zero padding adds classical features, not wires. Moment aggregation uses the unpadded measurement vector.',r'm=\max(8,n),\quad z_{\mathrm{pad}}=(z_0,\ldots,z_{n-1},0,\ldots,0)\in\mathbb R^m,\\ M_k=\frac1n\sum_{q=0}^{n-1}z_q^k,\quad k=1,2,3,\qquad f=(z_{\mathrm{pad}},M_1,M_2,M_3)\in\mathbb R^{m+3}.'),
('The scalar prediction is in Hartree; targets are not standardized. For fingerprint models, the number of wires is the fingerprint width rather than necessarily the AO count.',r'\widehat y=b+\sum_{j=1}^{m+3}w_j f_j.'),
('Initialization uses a local seeded CPU Torch generator. MB draws one complete vector; fingerprints draw separate ansatz, bias and weight blocks. The bias is then overwritten by the mean training target.',r'\theta^{(0)}\sim0.05\,\mathcal N(0,I),\quad b^{(0)}=\frac1{N_{\mathrm{train}}}\sum_{a\in\mathrm{train}}y_a,\\ D_{\mathrm{MB}}=L+m+7,\qquad D_{\mathrm{fingerprint}}=L+m+4.'),
('An independently maintained, same-seed shuffle generator produces a fresh permutation each epoch. Keep the final partial batch. Loss is in squared Hartree and divides by the actual batch size.',r'\mathcal L_B=\frac1{|B|}\sum_{a\in B}(\widehat y_a-y_a)^2.'),
('Each update clears gradients to None, differentiates, binds gradients according to the model parameter layout, and steps Torch Adam. MB uses one vector group; fingerprints use shared-storage blocks. Installed defaults and two diagnostic steps were checked; the full training trajectory was not replayed.',None),
('Checkpoint on strict improvement, retaining the earliest tie. Manifest training selects by validation MAE; 15-15 selects by training MAE. Restore parameters only, not optimizer state, before final test evaluation.',r'e_* = \min\!\left(\operatorname*{arg\,min}_{e}\mathrm{MAE}_{\mathrm{selection}}(e)\right).'),
('Compute each split separately. Convert scalar metrics from Hartree to milli-Hartree only after reduction.',r'e_a=\widehat y_a-y_a,\quad \mathrm{MAE}=\frac1N\sum_a|e_a|,\quad \mathrm{RMSE}=\sqrt{\frac1N\sum_ae_a^2},\\ \mathrm{MaxAE}=\max_a|e_a|,\quad \mathrm{NPE}=\max_ae_a-\min_ae_a,\qquad \mathrm{metric}_{\mathrm{mHa}}=1000\,\mathrm{metric}_{\mathrm{Ha}}.'),
('The full launcher requires 32 seeds. For overall results, first average molecules with equal weight within each seed, then summarize seeds. Also report the median and count. Core singleton SD/SE are NaN; the plot helper uses SD zero and SE NaN.',r'\bar x=\frac1S\sum_{s=1}^Sx_s,\quad s_x=\sqrt{\frac1{S-1}\sum_{s=1}^S(x_s-\bar x)^2},\quad \mathrm{SE}=s_x/\sqrt S.\\ x_s^{\mathrm{overall}}=\frac1{N_{\mathrm{mol}}}\sum_mx_{m,s}.'),
('Plotting checks exact configured seed coverage and equality of sorted geometry ID, split, bond length and target. Bars show test mean plus/minus SE; curves show per-geometry signed-error mean plus/minus sample SD. FE-prime is absent from the plotting CLI. Initial-data plots are a separate branch, with 31 curve points including equilibrium and 30 scan rows in CSV.',r'E_{\mathrm{relative}}=1000\,[E-E_{\mathrm{RHF}}(r_e)]\quad\mathrm{mHa};\qquad E_{\mathrm{absolute}}=E\quad\mathrm{Ha}.')]
assert len(rows)==len(entries)==37
def esc(s):
 s=re.sub(r'\[([^\]]+)\]\([^)]*\)',r'\1',s).replace('**','').replace('`','')
 s=s.replace('–','-').replace('—','-').replace('−','-')
 return ''.join({'&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}','\\':r'\textbackslash{}'}.get(c,c) for c in s)
pre=r'''\documentclass[10pt,a4paper]{article}
\usepackage[margin=18mm]{geometry}
\usepackage{fontspec}
\setmainfont{DejaVu Serif}
\setsansfont{DejaVu Sans}
\usepackage{amsmath,amssymb,longtable,array,booktabs,xcolor,fancyhdr,hyperref}
\definecolor{navy}{HTML}{163C54}
\hypersetup{colorlinks=true,urlcolor=navy,linkcolor=navy}
\pagestyle{fancy}\fancyhf{}\fancyhead[L]{\sffamily\small qml2.0/codes: scientific pipeline audit}\fancyfoot[R]{\thepage}
\setlength{\headheight}{14pt}\setlength{\parindent}{0pt}\setlength{\parskip}{5pt}
\setlength{\tabcolsep}{7pt}\renewcommand{\arraystretch}{1.2}
\newcommand{\stage}[2]{\section*{#1}\textbf{Code evidence:} #2\par}
\begin{document}
{\LARGE\sffamily\color{navy} Computational pipeline audit}\par
{\large qml2.0/codes --- formula-readable edition}\par
24 September 2026
\paragraph{Scope} Code-only audit using the supplied table as a stage checklist. This is a typeset edition of the completed audit, not a new source audit or a comparison with the supplied document. Code and datasets are unchanged.
\paragraph{Notation} $M$ is the full spatial AO/MO count; $v_i$ maps a retained local AO to a full source AO; $n$ is the selected descriptor count or fingerprint wire count. Energies are in Hartree (Ha); Cartesian geometries are in angstrom. All transposes refer to implemented real arrays. Symbol $H$ denotes the overlap square root in the basis-transformation stages, but the indexed Hessian in stage 1. Symbols such as $b$ and $J$ are local to their stated stages.
\section*{Principal findings}
\begin{enumerate}
\item Default H2S input retains six AOs while the current loader requires ten; the default nine-molecule run fails this data contract. Producer and consumer output/input defaults also differ.
\item Training accepts \texttt{FE\_prime}, but the plotting CLI does not.
\item Descriptor production does not explicitly reject unconverged RHF or guard overlap eigenvalues before inverse square roots. Downstream validation catches some failures. No failing-convergence sample was observed.
\end{enumerate}
\paragraph{Source key} I: \texttt{initialdata.py}; D: \texttt{descriptor.py}; R: \texttt{run1.py}; C: \texttt{circuit.py}; T: \texttt{train.py}; P: \texttt{plot\_run1\_results.py}. All files are under \texttt{qml2.0/codes}. Numbers identify audited source lines.
\clearpage
'''
parts=[pre]
for (title,loc,_),(body,formula) in zip(rows,entries):
 parts.append(r'\stage{'+esc(title)+'}{'+esc(loc)+'}\n'+esc(body)+'\n')
 if formula:parts.append('\\[\n\\begin{gathered}\n'+formula+'\n\\end{gathered}\n\\]\n')
# Preserve cross-stage findings and verification narrative from the completed report.
parts.append(r'\clearpage\section*{Cross-stage findings}'+'\n')
cross=src.split('## Cross-stage checks\n')[1].split('## Numerical verification')[0]
parts.append('\\begin{itemize}\n')
for l in cross.splitlines():
 if l.startswith('- '):parts.append('\\item '+esc(l[2:])+'\n')
parts.append('\\end{itemize}\n\\section*{Numerical verification}\n')
nums=src.split('## Numerical verification already completed\n')[1]
parts.append(r'\small\begin{longtable}{p{.29\linewidth}p{.36\linewidth}p{.25\linewidth}}\toprule Check & Shapes / coverage & Maximum absolute residual\\\midrule\endfirsthead\toprule Check & Shapes / coverage & Maximum absolute residual\\\midrule\endhead'+'\n')
for l in nums.splitlines():
 if l.startswith('|') and not l.startswith('| Check') and not l.startswith('|---'):
  vals=[esc(v.strip()) for v in l.strip('|').split('|')]
  parts.append(' & '.join(vals)+r'\\\addlinespace'+'\n')
parts.append('\\bottomrule\\end{longtable}\\normalsize\n')
for l in nums.splitlines():
 if l.strip() and not l.startswith('|'):parts.append(esc(l)+'\n\n')
parts.append('\\end{document}\n')
(root/'output/latex/qml2_codes_audit.tex').write_text(''.join(parts))
print('Wrote 37 stages with display equations and numerical evidence.')
