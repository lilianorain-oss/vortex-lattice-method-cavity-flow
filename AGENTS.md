# AGENTS.md

Research prototype: unsteady VLM + ventilation/cavity coupling for a surface-piercing strut. No package, tests, lint, CI, or dependency manifest.

## showing
Don't show me every thinking process you have done in the terminal.
At the end I want a clear report of what you did and an answer but no need to see everything you have tried so it's easier to scrolll in the chat

## git rules
You are not allowed to use git. No add, no commit and no push
Git is usefull for me to track your modification so it's important you don't use it

## files creation
if you need to create files / results you all put them in a repository opencode in the global repository

## Layout

- `src/VLMPanel.py`, `src/VLMSurface.py`, `src/VLMSolver.py` — library. Load path is `VLMSurface -> VLMSolver -> VLMPanel`.
- `vlm_unsteady_wake.ipynb` — current driver (unsteady wake + `_ventilation()` coupling). Edit `src/` via this notebook.
- `vlm_weissinger.ipynb` — older monolithic VLM (classes inlined); do not treat as source of truth for `src/`.
- `llm_prandtl.ipynb` — separate Prandtl lifting-line prototype, unrelated to `src/`.
- `data/*.csv` — cavity/ventilation curves (`cavity`, `cl`, `a0`, `alpha`, `Lc`, `sigma`); semicolon-separated, comma decimals.
- `docs/` — reference PDFs + `xflr5_wing_30_15.avl`; read-only.

## Run

- Deps (unpinned, install manually): `numpy`, `numba`, `matplotlib`, `jupyter`. No requirements file.
- Always run from repo root: `src/VLMSurface.py` does `sys.path.append("./src/")` + `from src.VLMPanel import VLMPanel`, so imports break from `src/` as cwd.
- No test suite. Verify by running a small case in `vlm_unsteady_wake.ipynb` (few span/chord panels, short `t`) and checking `Cl_2d` / `loads`.

## Gotchas

- `VLMSolver._induced_velocity` is `@nb.njit(parallel=True)` — first call pays JIT compile time; keep it vectorized, don't break the `prange` loop or add Python objects inside.
- `rc` cutoff (`a_ratio * min panel width`, default `a_ratio=0.4`) zeroes induction inside `d2 < rc^2`; wake rollup uses `core = c_mean * ratio` (Scully, default `ratio=0.08`). Tuning these changes stability, not just accuracy.
- `_time_sim()` defaults to `t = 10*c_mean/|U|`, `dt = t/50`; cost grows per step (wake grows + `inv_A @ b` each step). Keep `N/M` small when iterating.
- `_ventilation()` iterates up to 400x to `1e-3` on `|a0*aoa - Cl|`, calling `_correct()` + full `_time_sim()` each pass — expensive; prints "Did not converge" on failure. `aoa` is clamped at `0.001` in `_ventilation_coef`.
- `boundary`: `"symmetric"` = wall image (negated mirror gamma), `"antisymmetric"` = free-surface-ish (same-sign mirror gamma); anything else disables the mirror wing. String must match exactly.
- `plan=0` is vertical (z-x), `plan=1` remaps to horizontal (y-x). `symmetric=True` doubles `N` via merge — pass pre-symmetry `n`.
- `tip_shed` is `"both"/"left"/"right"/"none"` (left/right defined for horizontal wing; for vertical, left=top, right=bottom).
- `VLMSurface(..., fly_height=H)` + `origin z = -H` trims the emerged part above `z=0`: cut before global rotations, N decreases (reference mesh size kept; waterline panel < 81% of reference merged with the first immersed one). `H=0` is a strict no-op. `L_foils(fly_height=...)` in `vlm_unsteady_wake.ipynb`; check script in `opencode/verify_fly_height.py`.
- `surface.loads_distrib` = raw per-section normal-force sums (NOT width-normalized; misleading as intensity under cosine spacing, correct for integrated loads).
- `_time_sim(..., computations=[...])` overwrites `surface.CL` / `surface.Cl_2d` with per-time nested lists (use them before `_kuttas_loads()`, which restores scalars); with `computations=[]` (default) every plain `_time_sim()` leaves `CL=[]`, `Cl_2d=[]` until kutta runs. Times must equal `k*dt` exactly (`3*0.1 != 0.3` in fp) and only match `classic` steps (cosine steps are uneven).
- `_kuttas_loads()` sets scalar `surface.CL` (lift, norm of Fy+Fz `2‖F_tot[1:]‖/(S·u²)`, same formula as `global_cl`) and `surface.CD` (induced drag only, `2|Fx|/(S·u²)`).

## GUI (`src/gui_app.py`, launcher `gui_vlm.py` à la racine)

- Tkinter + matplotlib embarqué (`FigureCanvasTkAgg`). Onglets : Surfaces (multi-surfaces), VLMSolver (time_sim/kutta/ventilation), Study (balayages P1×P2), Trim (batch `trim_foils` + `Save surfaces` → N presets).
- Presets surfaces en JSON dans `opencode/saved_surfaces/` ; le nom du preset = toujours le nom du fichier (`load_preset`), jamais le champ embarqué.
- Conventions UI : anglais, labels mathtext (`$C_L$`, `$\beta$`, `$A_R$`), pas de titres sur les plots, légende unique dans la 3D, lignes seules + `ylim(bottom=0)` sur les distribs, CSV `;` + décimales virgule + ligne header (`write_csv`) ; export Study : lignes sans `CL` (surface absente du run, décochée, point en échec) sautées (`_cl_missing`), labels dist `$s/b$`.
- Runs longs en threads + `msg_q` + polling `after(150)` ; snapshotter les widgets avant (jamais de Tk depuis un worker) ; patcher `messagebox.showerror` dans les scripts headless (bloquant sinon).
- Study : presets rejoués à chaque run avec `surface._reset()` avant (sinon reshape `gamma_wake`) ; frontière héritée des surfaces (`common_boundary`, erreur si mixte) ; `Fn_h` force U selon x (incompatible `Ux/Uy/Uz`) ; sweep temps = un seul `_time_sim(t=max)` par groupe (`classic` forcé, temps multiples de `dt`), erreur en mode ventilation.
- Study sweep surfaces (`solver: surface set`, syntaxe `A; A+B`, un seul param set) : composition par run (ticks ignorées, loggé) ; résultats indexés par nom + position (`per_surf/dists/posgrid` alignés `[pool][n1][n2]`) ; plots par séries de positions (`_series()` : index dans chaque composition ; `_draw_series_cl/_draw_series_dist`), une figure par ligne, pas de fallback per-surface en mode sweep.
- Trim : wrapper fin de `trim_foils` (bloquant par point, pas de Cancel dedans), un couple alpha/beta global, cibles N/kgf avec rescale eau/air (×1025/rho), P1 ≠ alpha/beta/__surfaceset__.
- Tests : `python opencode/test_gui_smoke.py` (rapide, sans solves lourds ; exige display pour les parties Tk) ; e2e manuels lents `opencode/check_series_{e2e,multi_e2e}.py` (petites surfaces n=4,m=2, origines décalées — surfaces coïncidentes identiques = matrice singulière).
