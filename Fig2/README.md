# Chimera states without incoherence — numerical reproducibility

MATLAB source code for the three-dimensional, fifth-harmonic Kuramoto model on the unit sphere and the corresponding phase diagrams. The scripts are separate by design: one integrates the many-oscillator system, and the other computes theoretical stability boundaries without integrating particles.

## Files

- `phase_diagram_simulation.m` — scan the `(gamma_C1, gamma_C2)` grid and calculate time averages of azimuthal coherence relative to the **instantaneous mean-field direction**. Exports two simulation-only phase maps, diagnostics, CSV, and a numerical `.mat` result.
- `three_cluster_angle_boundaries_rho.m` — solve self-consistent three-cluster rotating states, trace the P and C1 internal-stability boundaries, validate selected solutions, and overlay exact/approximate theory curves onto the simulation phase maps if the `.mat` file exists.

No MATLAB toolboxes are required to use the serial CPU simulation or the boundary calculation. Parallel Computing Toolbox is optional for CPU `parfeval` and GPU acceleration. The simulator automatically falls back to serial CPU if GPU/parallel support is unavailable. MATLAB releases supporting local functions in scripts are needed. PDF export with `exportgraphics` uses newer MATLAB releases; a `print` fallback is provided.

## Reproducing the maps

1. Keep both `.m` files in the **same folder**, and use MATLAB with this folder as the current working directory (or run each script by its path).
2. In `phase_diagram_simulation.m`, keep `cfg.mode = 'simulate'` to compute the full scan. The default grid uses `N=1000`, `K=50`, fifth harmonic `n=5`, zero frequency heterogeneity, `gamma_C1=0:0.01:0.50`, and `gamma_C2=0:0.01:0.35`. Every initial condition consists of the point P plus rings C1 (polar angle `2*pi/5`) and C2 (`4*pi/5`), with independent small position perturbations of magnitude `1e-3`.
3. The script writes `chimera_phase_gpu_batch_rho_axis_results.mat` in the script folder; figures, `phase_meancheck_summary.csv`, and checkpoints go to `matlab_output_gpu_batch_rho_axis/`. The MAT file contains `results` and `simulationCfg`, **not old theoretical boundary estimates**.
4. Run `three_cluster_angle_boundaries_rho.m`. It reads that MAT file automatically, produces theoretical boundaries in `angle_boundary_output_rho_v4/`, and overlays them on the two numerical maps (`C1_with_angle_boundaries.pdf` and `C2_with_angle_boundaries.pdf`, along with PNG/FIG versions). Without a MAT file, it produces theoretical figures and diagnostics only.

To **redraw existing simulation data**, change `cfg.mode` in `phase_diagram_simulation.m` to `'plot'`. To resume an interrupted scan, use `cfg.mode='simulate'` and `cfg.resume=true`; checkpoint reloading is subject to matching simulation parameters. Reverting `cfg.resume=false` starts a new scan. `cfg.mode='auto'` reloads compatible saved results or starts a new scan.

## Simulation and convergence definition

The N-body field in the code is the fifth-order spherical interaction:

```text
rho = mean_i sigma_i; R = ||rho||; a_i = dot(sigma_i, rho)
(d/dt) sigma_i = K [16*a_i^4 - 12*R^2*a_i^2 + R^4] (rho - a_i*sigma_i)
                 = -K R^5 sin(5 theta_i) e_theta_i .
```

The numerical integration uses an adaptive embedded RK45 scheme with tangent-vector normalization. For each ring `C_alpha`, `|Z_alpha(t)|` is the magnitude of the mean normalized orthogonal projections of oscillators onto the plane perpendicular to the **instantaneous** `rho`. Colors are physical-time averages of `|Z_alpha|`, not `|mean(Z_alpha)|`.

The baseline discards `6000` time units; each check compares consecutive non-overlapping `2000`-unit windows using `meanTol=0.005`, requiring two consecutive passes and a minimum effective clock `int K*R^5 dt >= 20` per window. Each plotted value averages the final **4000** time units. Integration stops when the convergence condition has passed twice or reaches `tMax=30000`. Inspect `results.status`, the CSV, and diagnostic figures for unresolved points before treating the phase maps as converged. The `(0,0)` case has no C1 or C2 ring and produces undefined ring-order parameters.

Simulation random seeds are deterministic functions of the grid coordinates and `cfg.seedBase=20260919`. CPU/GPU integration may still differ slightly due to floating-point execution. The release keeps the underlying numerical routines and checks unchanged from the rho-axis GPU/CPU scan (checkVersion=3, engineVersion=5).

## Theory-line interpretation

- **Red solid:** C1 cluster internal instability on the self-consistent rotating **three-cluster** state.
- **Black solid:** P cluster internal instability on the same three-cluster state.
- **Black dashed:** approximate P-stability criterion on the chimera side; **not** an exact three-cluster stability curve.
- **Blue dotted:** approximate positive/negative azimuthal-feedback balance, displayed over the full x interval. The code additionally exports its analytical intersection with the approximate P criterion, solely as a diagnostic.

The P-unstable black region combines the selected exact three-cluster P boundary with a conservative approximate cutoff on the chimera side; the junction is approximate and should not be misinterpreted as an additional physical phase boundary. Position stability and C2 internal stability are filtering criteria for the solid three-cluster curves. The boundary script validates a subset of roots by comparisons to the original vector dynamics. The theory describes selected local-stability boundaries, not all attractors, basin boundaries, or an exact stability proof for the chimera state.

## Publication and archive guidance

The full simulation may be computationally intensive. It is generally preferable to archive a checksum-tagged final numerical MAT file with the repository **release** or a data repository and keep runtime checkpoints out of Git history. Do not commit `.mat` data with obsolete `theory` fields as the current theoretical result. If original simulation results are reused from an earlier script version, the cleaned plot mode ignores the legacy theory field and the boundary script reads only the numerical data.

Code has been checked for internal configuration and file-format compatibility, but the full 1836-point simulation and MATLAB numerical continuation have **not been executed in this environment**. Prior to publication, run both scripts with the MATLAB version, platform, and result archive intended for replication and inspect `validation.csv`, `trace_endpoints.csv`, convergence diagnostics, and all figure outputs.
