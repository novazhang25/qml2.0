from pathlib import Path
import ast,re
root=Path('/Users/novaz/Desktop/qml2.0')
a=ast.parse((root/'tmp/pdfs/build_latex.py').read_text())
entries=next(ast.literal_eval(n.value) for n in a.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='entries' for t in n.targets))
source=(root/'output/code_audit_20260924/CODE_AUDIT.md').read_text()
rows=[[x.strip() for x in l.strip('|').split('|')] for l in source.splitlines() if re.match(r'\| \d+ ',l)]
metadata=[
('Molecule specification; optimized reference; stretch tangent; Hessian; atomic masses','30 aligned geometries; scan coordinate','Cartesian','a,b: atoms; x,y: Cartesian components; j=0,...,29. Geometry in angstrom; SI used for harmonic conversion.'),
('Geometry; basis; charge; spin','S: M by M','Full AO','M: full spatial AO count. Dimensionless overlap; PySCF integral boundary.'),
('Molecule; SCF settings; saved density initial guess for descriptor RHF','P,F,C: M by M; RHF energy','AO / MO','n_p: spatial orbital occupation. P spin summed; F and total energy in Ha.'),
('Molecule registry; AO labels; shell metadata','Frozen-MO specification; retained AO map v','MO / active AO','FCI frozen-core definition is distinct from the descriptor AO exclusion. No MP2 frozen core.'),
('Recomputed canonical RHF reference','Restricted doubles t2','Full MO','Occupied and virtual axes are spatial orbitals. Amplitudes dimensionless; PySCF boundary.'),
('Reference determinant; zero singles; restricted t2','Consistent 1-RDM and 2-RDM','Determinant / MO','R denotes either RDM API; Phi is reference, chi is correction; w is its squared norm.'),
('Raw rank-four RDM-derived array','Raw array; extraction-only permutation','AO / Lowdin AO','Permutation swaps axes 1 and 2 (zero based). It is not a global change to stored raw tensors.'),
('MO RDMs; full real coefficient matrix C','Full AO 1-RDM and 2-RDM','MO to AO','a,b,c,d: AO indices; p,q,r,s: MO indices. All axes run over M.'),
('Full transformed arrays; source and canonical AO maps','Selected arrays in canonical order','Full to active Lowdin AO','v_i maps local i to full source AO. BeH2 displayed source indices are zero based.'),
('Full RHF AO density P','Disconnected G0: M to fourth power','Full AO','p,q,r,s are free indices. Factor 1/2 follows the implemented spin-summed expression.'),
('Consistent AO 2-RDM; RHF disconnected reference','Lambda: M to fourth power','Full AO','Not the separately stored conventional cumulant based on correlated density.'),
('Stored RDM diagnostics','No extra all-ones contraction','Stored diagnostics','No corresponding downstream stage is executed.'),
('Canonical RHF; configured core and active electrons','FCI energy; CI; target y; active integrals','Active MO / determinant','ncore: frozen spatial MOs; active interval excludes M. Target is in Ha.'),
('Full overlap S','H=S^(1/2); J=S^(-1/2)','Full AO','Here H is overlap square root, not stage-1 Hessian; s contains overlap eigenvalues.'),
('H,J; full RHF density; AO Fock matrix','P_L,F_L: M by M','Full Lowdin AO','P_L dimensionless; F_L in Ha. No explicit symmetrization.'),
('H; full AO Lambda','Lambda_L: M to fourth power','Full Lowdin AO','a,b,c,d,p,q,r,s each span full space. All four factors are H.'),
('Lambda_L; retained AO map v','T_full; strict-triple vector','Active Lowdin AO','i,j,k are local retained indices. Repeated k is not summed. Saved, unused by Run 1.'),
('Selected P_L,F_L; canonical AO order','Diagonals, pairs, full blocks, spectra','Active Lowdin AO / features','FE: density spectrum. FE_prime: Fock spectrum. Ascending eigenvalues; n entries each.'),
('30 accepted records per molecule','Sorted complete population','Classical records','q_A: stored scan coordinate. Sort tie breaker is geometry index; no cutoff filtering.'),
('Population; split manifest or 15-15 selection','Train / validation / test membership','Classical records','Geometry IDs are one based. Odd/even membership does not refer to traversal positions.'),
('Training diagonal entries for each descriptor type','Scalar mean/std; standardized entries','Classical features','K counts all training geometry-AO entries. Separate statistics for P and F; ddof=0.'),
('Selected local AO count n','Ordered pair/triple addresses','Local AO indices','Pairs i<j; triples i<j<k. No global AO numbering in these enumerations.'),
('Pooled training absolute pair/triple values','alpha; optional helper beta','Classical scale parameters','Percentile includes zeros and uses linear interpolation. Positive scale is mandatory.'),
('MB standardized diagonals and a,b,c; or raw fingerprint and fitted statistics','Dimensionless rotation angles','Features to circuit wires','a,b,c shared across MB wires. Fingerprint mask is per column; clipping is last.'),
('Pair entries and fitted alpha (helper)','Pair gates; none for MB-1','Circuit / wire','IsingZZ library gate; local i<j. Not executed by the launched MB-1 branch.'),
('Triple entries and fitted beta (helper)','Triple gates; none in Run 1','Circuit / wire','MultiRZ library gate; local i<j<k. Runtime not verified for this helper.'),
('Encoded state; shared layer parameters phi','Two-layer ansatz state','Circuit / parameter','One phi per layer, shared across wires. CNOT order is sequential, wrap edge last.'),
('Final circuit state','z: n entries','Wire expectations','q=0,...,n-1, physical wire order. Pauli-Z is dimensionless.'),
('Unpadded z; selected wire count n','f: m+3 entries','Classical features','m=max(8,n). Moments use n measured entries, not the padded vector.'),
('f; trainable readout weights and bias','Scalar predicted correlation energy','Parameter / scalar energy','w has m+3 entries; b is readout bias, distinct from MB encoding coefficient b. Ha.'),
('Seed; model layout; training targets','Initial model parameters','Parameter space','D is total parameter count. Gaussian draws occur before target-mean bias overwrite.'),
('Training set; persistent shuffle generator; batch size','Batches; scalar MSE','Classical / parameter','B is actual batch, including final partial batch. Loss unit Ha squared.'),
('Loss gradients; bound parameter blocks; optimizer state','Updated parameters','Parameter space','Torch Adam boundary. Gradient clearing and model-specific parameter storage inspected.'),
('Epoch metrics; current theta; selection protocol','Best theta; final evaluation','Parameter space','Strict improvement retains first tie. Optimizer state is not restored.'),
('Predictions and targets in one split','MAE, RMSE, MaxAE, NPE','Scalar statistics','e is signed prediction minus target. Convert after reduction: Ha to mHa.'),
('Per-seed, per-molecule metrics','Mean, SD, median, count, SE','Classical aggregation','S is seed count here, not overlap. Sample SD uses ddof=1. Molecules equally weighted.'),
('Configured seeds; predictions; initial-data arrays for separate branch','Error plots; energy curves','Classical plotting','Seed identity includes geometry ID, split, bond length and target. Curve and CSV sizes differ.')]
assert len(metadata)==37
def esc(s):
 s=s.replace('–','-').replace('—','-').replace('−','-')
 return ''.join({'&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_\allowbreak ', '{':r'\{','}':r'\}','^':r'\textasciicircum{}','\\':r'\textbackslash{}'}.get(c,c) for c in s)
pre=r'''\documentclass[10pt]{article}
\usepackage[paperwidth=594mm,paperheight=420mm,margin=14mm]{geometry}
\usepackage{fontspec}
\setmainfont[BoldFont=lmroman10-bold.otf,ItalicFont=lmroman10-italic.otf]{lmroman10-regular.otf}
\usepackage{amsmath,amssymb,array,longtable,booktabs,adjustbox,fancyhdr}
\pagestyle{fancy}\fancyhf{}\fancyfoot[C]{\thepage}\renewcommand{\headrulewidth}{0pt}
\setlength{\parindent}{0pt}\setlength{\tabcolsep}{1mm}\setlength{\LTpre}{6mm}
\renewcommand{\arraystretch}{1.18}\sloppy
\newcolumntype{P}[1]{>{\raggedright\arraybackslash}p{#1}}
\newcommand{\eqcell}[1]{\begin{adjustbox}{max width=\linewidth}$\displaystyle\begin{gathered}#1\end{gathered}$\end{adjustbox}}
\begin{document}
{\Large\bfseries Current-code pipeline table: qml2.0/codes}\par\vspace{3mm}
Scope: the 37 audited computational stages, formatted in the eight-column structure of the supplied PDF. This is a code-only record; formulas describe repository operations, with external-library boundaries identified. No code or datasets were modified.\par
Source key: I = initialdata.py; D = descriptor.py; R = run1.py; C = circuit.py; T = train.py; P = plot\_run1\_results.py. All source locations refer to the completed audit of 24 September 2026. Energies are Hartree unless stated.\par
Important findings: default H2S data fails the current AO-selection contract; FE\_prime is missing from the plotting CLI; descriptor production lacks explicit convergence acceptance. Numerical checks and limitations remain in the accompanying audit report.\par
\begin{longtable}{P{24mm}P{36mm}P{55mm}P{63mm}P{36mm}P{30mm}P{155mm}P{51mm}}
\toprule
Layer & Input & Exact method / function & What is done & Output & Basis & Formula & Important element definition in formula\\\midrule
\endfirsthead
\toprule
Layer & Input & Exact method / function & What is done & Output & Basis & Formula & Important element definition in formula\\\midrule
\endhead
\bottomrule\endfoot
'''
parts=[pre]
for row,(body,formula),meta in zip(rows,entries,metadata):
 inp,out,basis,defs=meta
 fm=r'\eqcell{'+formula+'}' if formula else 'No additional repository formula; see operation and boundary.'
 vals=[esc(row[0]),esc(inp),esc(row[1]),esc(body),esc(out),esc(basis),fm,esc(defs)]
 parts.append(' & '.join(vals)+r'\\[3mm]\midrule'+'\n')
parts.append(r'\end{longtable}\end{document}')
(root/'output/latex/qml2_codes_pipeline_table.tex').write_text(''.join(parts))
