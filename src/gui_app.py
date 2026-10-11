"""Tkinter GUI for the unsteady VLM + ventilation prototype.

Units in the GUI: angles in degrees, distances in m, speeds in m/s or knots
(selectable, converted to m/s before calling the solver).
Conversions deg->rad are done at build time before calling VLMSurface.
Solver-level numeric defaults match VLMSolver/VLMSurface code defaults
(ratio=0.08, a_ratio=0.01, k_ratio=0.4, distribution="classic",
root_ang=tip_ang=0, radius=center=1, fly_height=0).

Layout:
  Tab "Surfaces" : "+ New surface" creates one per-surface tab with the
  full VLMSurface form (left) + Build button + preview (right).
  Preview shows 3 views: 2D axis 0 always, 2D axis 1 if plan==0 else axis 2,
  and 3D. The mirror (boundary condition) display can be toggled.
  Tab "Solver" : VLMSolver params + time_sim / kutta loads / ventilation runs.
  The solver boundary is taken from the surfaces (all must match).
"""
import os
import sys
import ast
import copy
import json
import math
import operator
import queue
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np
import matplotlib

matplotlib.use("TkAgg")
# same fonts as vlm_unsteady_wake.ipynb cell 1
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["Times New Roman"]
matplotlib.rcParams["mathtext.fontset"] = "stix"
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator, MultipleLocator, LinearLocator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.VLMSurface import VLMSurface
from src.VLMSolver import VLMSolver
from src.Utilities import chord_fn, trim_foils

DEG = np.pi / 180.0

# GUI prefill values, mirroring the VLMSurface / VLMSolver __init__ defaults.
# (chord c0/AR are app-level: the code default c=None cannot build a mesh.)
SURFACE_DEFAULTS = dict(ox=0.0, oy=0.0, oz=0.0, plan=0, boundary="none",
                        shedding="both", span=1.0, chord_mode="rectangular", c0=0.1,
                        AR=1.0, lam=1.8, chord="", alpha=0.0, beta=0.0, sweep=0.0,
                        dihedral=0.0, twist="0", rake=0.0, n=10, m=10, sym=False,
                        space=False, vent=True, root_ang=0.0, tip_ang=0.0,
                        radius=1.0, center=1.0, fly_h=0.0, c_line=0.5)
SOLVER_DEFAULTS = dict(ux=1.0, uy=0.0, uz=0.0, ratio=0.08, a_ratio=0.01,
                       k_ratio=0.4, distribution="classic")

# Medium densities [kg/m3] — only used to scale surface.loads into forces.
# The VLM solve itself is unchanged by this choice.
RHO = {"water": 1025.0, "air": 1.225}

# Speed / force unit factors
KT = 1.852 / 3.6      # 1 knot in m/s
KG = 9.81             # 1 kgf in N
GRAV = 9.81


def medium_density(name):
    """Density [kg/m3] for the 'water'/'air' selector (default: water)."""
    return RHO.get(str(name).strip().lower(), 1025.0)


def speed_to_si(value, unit):
    """Flow-speed entry to m/s ('m/s' or 'kt'); accepts expressions."""
    v = eval_expr(value)
    if str(unit).strip().lower() == "kt":
        return v * KT
    return v


def speed_from_si(value_si, unit):
    """m/s value back to the display unit ('m/s' or 'kt')."""
    if str(unit).strip().lower() == "kt":
        return float(value_si) / KT
    return float(value_si)


def force_factor(unit):
    """loads*rho is Newtons; divide by 9.81 for kgf."""
    return 1.0 / KG if str(unit).strip().lower() == "kgf" else 1.0


def scaled_loads(surface, rho, unit="N"):
    """surface.loads (force / density) scaled to N or kgf, or None."""
    if surface.loads is None:
        return None
    return np.asarray(surface.loads, dtype=float) * float(rho) * force_factor(unit)


# Backwards-compatible alias (Newtons).
def scaled_loads_N(surface, rho):
    return scaled_loads(surface, rho, "N")


def global_cl(surface, u_norm):
    """Global lift coef 2*||loads[1:]||/(area*u^2) (density cancels), or None."""
    if surface.loads is None or not surface.area or u_norm <= 0:
        return None
    return 2.0 * float(np.linalg.norm(np.asarray(surface.loads, dtype=float)[1:])) / (float(surface.area) * float(u_norm) ** 2)


def _is_num(x):
    return isinstance(x, (int, float, np.floating)) and not isinstance(x, bool)


def surf_coeffs(s):
    """(CL lift, CD induced drag) globals if scalar, else (None, None).
    Guards the multi-time list overwrite of _time_sim(computations=[...])."""
    cl = s.CL if _is_num(getattr(s, "CL", None)) else None
    cd = s.CD if _is_num(getattr(s, "CD", None)) else None
    return cl, cd


def cl2d_array(s):
    """1D Cl_2d array or None (rejects the multi-time nested list)."""
    try:
        c = np.asarray(s.Cl_2d, dtype=float) if s.Cl_2d is not None else None
    except (ValueError, TypeError):
        return None
    if c is None or c.ndim != 1 or len(c) == 0:
        return None
    return c


def plan_time_fast(p1, p2, solver_base):
    """Single-run time sweep plan. Returns (fast dict | None, note | None).
    If dt is provided in solver_base and no other param is swept (single group),
    computes dt/comps/times upfront for backward compatibility.
    Otherwise, returns requested t values; dt/comps/times computed per-group in _study_fast."""
    tside = None
    for tag, pp in (("p1", p1), ("p2", p2)):
        if pp is not None and (pp["scope"], pp["key"]) == ("solver", "t"):
            tside = tag
    if tside is None:
        return None, None
    swept = {k for s, k in ([(p1["scope"], p1["key"])] + ([(p2["scope"], p2["key"])] if p2 else []))
             if s == "solver"}
    if swept & {"dt", "distribution"}:
        return None, None
    base = dict(solver_base)
    note = None
    if base.get("dist", "classic") != "classic":
        base["dist"] = "classic"
        note = "note: time sweep forces classic time distribution"
    tvals = p1["values"] if tside == "p1" else p2["values"]
    tmax = max(float(v) for v in tvals)
    if not tmax > 0:
        raise ValueError("time sweep needs t > 0")
    # If dt is explicitly provided AND no other parameter (single group),
    # compute upfront for backward compatibility
    dt_raw = base.get("dt")
    dt_eff = eval_or(dt_raw, None) if dt_raw not in (None, "") else None
    has_other_param = p2 is not None if tside == "p1" else p1 is not None
    if dt_eff is not None and not has_other_param:
        ks, tvs = [], []
        for v in tvals:
            tv = float(v)
            k = int(round(tv / dt_eff))
            if k < 1 or abs(k * dt_eff - tv) > 1e-9 * max(1.0, abs(tv)):
                raise ValueError(f"time {v:g} is not a multiple of dt={dt_eff:g}")
            ks.append(k)
            tvs.append(tv)
        order = sorted(range(len(tvs)), key=lambda i: tvs[i])
        return (dict(tside=tside, dt=dt_eff, t_run=tmax,
                     comps=[ks[i] * dt_eff for i in order],
                     order=order, times=[tvs[i] for i in order],
                     base=base), note)
    else:
        # dt will be computed per-group from geometry in _study_fast
        return (dict(tside=tside, t_requested=tvals, t_run=tmax, base=base), note)


def common_boundary(named_surfs):
    """Solver boundary taken from the surfaces; error if they disagree."""
    bcs = {s.boundary for _, s in named_surfs}
    if len(bcs) > 1:
        raise ValueError(f"All selected surfaces must share the same boundary condition, got {sorted(bcs)}.")
    return next(iter(bcs))


def froude_chord(surface, u_norm):
    """Froude number based on the mean chord, or None."""
    c = float(np.mean(np.asarray(surface.chord, dtype=float)))
    if c <= 0 or u_norm <= 0:
        return None
    return float(u_norm) / np.sqrt(GRAV * c)


def froude_depth(surface, u_norm):
    """Froude number based on the max immersion, or None if not immersed."""
    h = surface.max_depth
    if h is None or float(h) <= 0 or u_norm <= 0:
        return None
    return float(u_norm) / np.sqrt(GRAV * float(h))


def emerged_bc_error(surface):
    """Error message if the mesh sticks out above z=0 with a mirror BC.

    z is positive downwards, so emerged corners have z<0. Truncated
    (fly_height>0) meshes are cut at the waterline and exempt. Tolerance
    scales with the mean chord to ignore mm-scale global-rotation nudges.
    """
    if surface.boundary not in ("symmetric", "antisymmetric"):
        return None
    if float(surface.fly_height or 0.0) > 0:
        return None
    zmin = float(np.asarray(surface.wing["real"], dtype=float)[:, 2].min())
    tol = 0.25 * float(np.mean(np.asarray(surface.chord, dtype=float)))
    if zmin < -tol:
        return (f"Surface pierces the free surface (min z={zmin:.3f} m) with a mirror "
                f"boundary condition ('{surface.boundary}'): set fly_height>0 to truncate "
                "the emerged part or move the origin below z=0.")
    return None


# ---------- helpers (unit handling / parsing) ----------

_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod}


def eval_expr(text):
    """Safe arithmetic eval for input cells (e.g. '2*0.4', '(1+2)/3')."""
    try:
        tree = ast.parse(str(text).strip(), mode="eval")
    except SyntaxError as ex:
        raise ValueError(f"bad expression {text!r}") from ex

    def _ev(node):
        if isinstance(node, ast.Expression):
            return _ev(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError(f"bad expression {text!r}")
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            v = _ev(node.operand)
            return v if isinstance(node.op, ast.UAdd) else -v
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            return _BINOPS[type(node.op)](_ev(node.left), _ev(node.right))
        raise ValueError(f"bad expression {text!r}")

    return _ev(tree)


def eval_or(text, default):
    return default if str(text).strip() == "" else eval_expr(text)


def _nice_ceiling(x):
    e = math.floor(math.log10(x))
    f = x / 10 ** e
    for m in (1, 2, 2.5, 5, 10):
        if f <= m + 1e-12:
            return m * 10 ** e
    return 10 * 10 ** e


def _fit_small_locator(lo, hi, step_big):
    """Locator with at least step_big spacing (2 ticks if needed)."""
    if hi - lo >= step_big:
        step = _nice_ceiling(step_big)
        inside = [t for t in MultipleLocator(step).tick_values(lo, hi)
                  if lo - 1e-9 <= t <= hi + 1e-9]
        if len(inside) >= 2:
            return MultipleLocator(step)
    return LinearLocator(numticks=2)


def _balance_2d_ticks(ax):
    """Give the narrow axis at least the tick spacing of the wide one
    (2 ticks if needed). Tick density only, scales untouched."""
    (x0, x1), (y0, y1) = sorted(ax.get_xlim()), sorted(ax.get_ylim())
    dx, dy = x1 - x0, y1 - y0
    if dx <= 0 or dy <= 0:
        return
    if dx >= dy:
        blo, bhi, slo, shi, saxis = x0, x1, y0, y1, ax.yaxis
    else:
        blo, bhi, slo, shi, saxis = y0, y1, x0, x1, ax.xaxis
    ticks = MaxNLocator(nbins=6, prune="both").tick_values(blo, bhi)
    step_big = ticks[1] - ticks[0] if len(ticks) >= 2 else (bhi - blo)
    if not step_big > 0:
        return
    saxis.set_major_locator(_fit_small_locator(slo, shi, step_big))


def _balance_3d_ticks(ax):
    """Same as _balance_2d_ticks for the two narrowest 3D axes."""
    lims = [sorted(ax.get_xlim3d()), sorted(ax.get_ylim3d()), sorted(ax.get_zlim3d())]
    ranges = [b - a for a, b in lims]
    if any(r <= 0 for r in ranges):
        return
    big = max(range(3), key=lambda i: ranges[i])
    ticks = MaxNLocator(nbins=5, prune="both").tick_values(*lims[big])
    step_big = ticks[1] - ticks[0] if len(ticks) >= 2 else ranges[big]
    if not step_big > 0:
        return
    for i, axis in enumerate((ax.xaxis, ax.yaxis, ax.zaxis)):
        if i != big:
            axis.set_major_locator(_fit_small_locator(*lims[i], step_big))

def parse_float(entry, name, default=None):
    txt = entry.get().strip()
    if txt == "":
        if default is not None:
            return default
        raise ValueError(f"{name}: empty value")
    return float(txt.replace(",", ".")) if "," not in txt and txt.count(",") == 0 else float(txt.replace(",", "."))


def parse_float_list(entry, name):
    txt = entry.get().strip().replace(",", " ")
    parts = [p for p in txt.split() if p]
    if not parts:
        raise ValueError(f"{name}: empty value")
    return [float(p) for p in parts]


def build_chord_array(mode, shape, ar, b, n_pre, sym, space, lam, c0):
    """Chord law at the n_pre+1 spanwise stations (pre-symmetry)."""
    nstat = n_pre + 1
    if mode == "uniform":
        return np.ones(nstat) * float(c0)
    return np.asarray(chord_fn(nstat, float(ar), float(b), bool(sym), bool(space), shape, lam=float(lam)))


def views_for_plan(plan):
    """2D axes rule: always axis 0, plus axis 1 if plan==0 else axis 2."""
    return (0, 1 if int(plan) == 0 else 2)


def enforce_shedding(chord_mode, shedding):
    """Elliptical chord laws taper to a zero tip chord: tip shedding is
    forced to 'none', anything else is kept as-is."""
    if chord_mode == "elliptical":
        return "none"
    return shedding


def linear_twist_weights(n_pre, b, sym, space, root_ang, tip_ang, radius, center, delta):
    """Spanwise weights of a scalar twist, mirroring VLMSurface.

    Replicates the pre-twist grid of VLMSurface._build_wing (station spacing,
    foil shape, dihedral rotation, symmetry merge) and the linear law of
    VLMSurface._apply_twist (phi_j = |z| / z_tip): returns w_j, the weight of
    each of the n_pre+1 stations (2*n_pre+1 merged stations if sym, like the
    library's doubled N). _generate_foil/_unit_rot are reused unbound (they
    don't use self); only the station spacing lines are mirrored.
    """
    le = 0.5 * b if sym else b
    if space:
        if sym:
            theta = np.linspace(np.pi / 2, np.pi, n_pre + 1)
            s = le * (-np.cos(theta))
        else:
            theta = np.linspace(0, np.pi, n_pre + 1)
            s = le * (1 - np.cos(theta)) / 2
    else:
        s = np.linspace(0.0, le, n_pre + 1)
    y, z = VLMSurface._generate_foil(None, s, root_ang, tip_ang, radius, center)
    pts = VLMSurface._unit_rot(None, np.column_stack(
        [np.zeros_like(y), y, z]), -delta, 0)
    zz = pts[:, 2]
    if sym:
        zz = np.concatenate([-zz[::-1][:-1], zz])  # mirror merge (z flips about z=0)
    return np.abs(zz) / zz[-1]


def safe_name(s):
    """Filename-safe preset name (keeps spaces)."""
    return "".join(c if c not in '/\\:*?"<>|' else "_" for c in str(s)).strip() or "surface"


def load_preset(path):
    """Read a surface preset JSON. The preset name is always the file name
    (stale embedded names from older saves are ignored)."""
    with open(path) as f:
        state = json.load(f)
    if not isinstance(state, dict) or not isinstance(state.get("entries"), dict) \
            or not isinstance(state.get("vars"), dict):
        raise ValueError("not a surface preset (need entries + vars)")
    return {"name": os.path.splitext(os.path.basename(path))[0],
            "entries": dict(state["entries"]), "vars": dict(state["vars"])}


class _StudyCancel(Exception):
    pass


# ---------- study helpers (sweep parsing, surface states) ----------

# combo label -> (scope, key, kind, short name, unit). scope "surface" needs a
# target surface from the study config, scope "solver" is global to the run.
STUDY_FIELDS = {
    "surface: origin x [m]": ("surface", "ox", "float", "ox", "m"),
    "surface: origin y [m]": ("surface", "oy", "float", "oy", "m"),
    "surface: origin z [m]": ("surface", "oz", "float", "oz", "m"),
    "surface: span b [m]": ("surface", "span", "float", "b", "m"),
    "surface: c0 [m]": ("surface", "c0", "float", "c0", "m"),
    "surface: AR": ("surface", "AR", "float", "AR", "-"),
    "surface: taper": ("surface", "lam", "float", "taper", "-"),
    "surface: alpha [deg]": ("surface", "alpha", "float", "alpha", "deg"),
    "surface: beta [deg]": ("surface", "beta", "float", "beta", "deg"),
    "surface: sweep [deg]": ("surface", "sweep", "float", "sweep", "deg"),
    "surface: dihedral [deg]": ("surface", "dihedral", "float", "dihedral", "deg"),
    "surface: twist [deg]": ("surface", "twist", "float", "twist", "deg"),
    "surface: rake [deg]": ("surface", "rake", "float", "rake", "deg"),
    "surface: chord [m]": ("surface", "chord", "str", "c", "m"),
    "surface: chord line": ("surface", "c_line", "float", "c_line", "-"),
    "surface: root angle [deg]": ("surface", "root_ang", "float", "root", "deg"),
    "surface: tip angle [deg]": ("surface", "tip_ang", "float", "tip", "deg"),
    "surface: radius [m]": ("surface", "radius", "float", "radius", "m"),
    "surface: center": ("surface", "center", "float", "center", "-"),
    "surface: fly height [m]": ("surface", "fly_h", "float", "fly_h", "m"),
    "surface: n": ("surface", "n", "int", "n", "-"),
    "surface: m": ("surface", "m", "int", "m", "-"),
    "surface: plan": ("surface", "plan", "str", "plan", "-"),
    "surface: boundary": ("surface", "boundary", "str", "boundary", "-"),
    "surface: shedding": ("surface", "shedding", "str", "shedding", "-"),
    "surface: chord law": ("surface", "chord_mode", "str", "chord", "-"),
    "surface: symmetric": ("surface", "sym", "bool", "sym", "-"),
    "surface: spacing": ("surface", "space", "bool", "space", "-"),
    "surface: ventilated": ("surface", "vent", "bool", "vent", "-"),
    "solver: Ux": ("solver", "ux", "float", "Ux", "spd"),
    "solver: Uy": ("solver", "uy", "float", "Uy", "spd"),
    "solver: Uz": ("solver", "uz", "float", "Uz", "spd"),
    "solver: Fn_h": ("solver", "fn", "float", "Fn_h", "-"),
    "solver: t [s]": ("solver", "t", "float", "t", "s"),
    "solver: dt [s]": ("solver", "dt", "float", "dt", "s"),
    "solver: ratio": ("solver", "ratio", "float", "ratio", "-"),
    "solver: a_ratio": ("solver", "a_ratio", "float", "a_ratio", "-"),
    "solver: k_ratio": ("solver", "k_ratio", "float", "k_ratio", "-"),
    "solver: distribution": ("solver", "distribution", "str", "dist", "-"),
    "solver: surface set": ("solver", "__surfaceset__", "set", "set", "-"),
}

_BOOL_WORDS = {"1": True, "true": True, "yes": True, "y": True,
               "0": False, "false": False, "no": False, "n": False}


def parse_sweep_values(text, kind):
    """Parse ';'-separated sweep values. Numeric items are expressions or
    start:stop:step ranges (inclusive stop); str items are literals."""
    items = [t.strip() for t in str(text).split(";") if t.strip()]
    if not items:
        raise ValueError("empty sweep values")
    out = []
    for it in items:
        if kind == "str":
            out.append(it)
            continue
        if kind == "set":
            # surface set: preset names joined with "+" (e.g. "A; A+B; B")
            names = [n.strip() for n in it.split("+") if n.strip()]
            if not names:
                raise ValueError(f"bad surface set {it!r} (use Name or Name+Name)")
            out.append(names)
            continue
        if kind == "bool":
            key = it.lower()
            if key not in _BOOL_WORDS:
                raise ValueError(f"bad boolean {it!r} (use true/false or 1/0)")
            out.append(_BOOL_WORDS[key])
            continue
        if ":" in it:
            parts = it.split(":")
            if len(parts) not in (2, 3):
                raise ValueError(f"bad range {it!r} (use start:stop[:step])")
            nums = [eval_expr(p) for p in parts]
            if len(nums) == 2:
                start, stop = nums
                step = 1.0 if stop >= start else -1.0
            else:
                start, stop, step = nums
            if step == 0 or (stop - start) * step < 0:
                raise ValueError(f"bad range {it!r} (step sign must go from start to stop)")
            count = int(math.floor((stop - start) / step + 1e-9)) + 1
            if count < 1 or count > 10000:
                raise ValueError(f"bad range {it!r}")
            vals = [start + i * step for i in range(count)]
            out.extend(int(round(v)) if kind == "int" else v for v in vals)
            continue
        v = eval_expr(it)
        out.append(int(v) if kind == "int" else v)
    if not out:
        raise ValueError("empty sweep values")
    return out


def fly_coupled_overrides(state, v):
    """fly_height sweep coupling: set fly_h AND shift origin z by the same
    amount (z+ downwards, free surface at z=0) so the waterline keeps cutting
    the right station — like origin z=-h with fly_height=h in run_trim_all."""
    E = state.get("entries", {})
    try:
        hb = eval_expr(E.get("fly_h", "0") or "0")
    except Exception:
        hb = 0.0
    try:
        zb = eval_expr(E.get("oz", "0") or "0")
    except Exception:
        zb = 0.0
    v = float(v)
    return {"fly_h": v, "oz": zb - (v - hb)}


def param_overrides(st, pp, vv):
    """Override dict for one sweep param on one surface state.
    pp: {scope, key, surf|None, all} (Study/Trim parsed sweep)."""
    if pp is None or pp["scope"] != "surface":
        return {}
    if not (pp["all"] or pp.get("surf") == st.get("name")):
        return {}
    if pp["key"] == "fly_h":
        return fly_coupled_overrides(st, vv)
    return {pp["key"]: vv}


def apply_state_overrides(state, overrides):
    """Copy of a surface form-state ({"name","entries","vars"}) with
    {key: value} applied. entries <- formatted text, vars <- raw values."""
    E = dict(state["entries"])
    V = dict(state["vars"])
    for key, val in overrides.items():
        if key in E:
            E[key] = val if isinstance(val, str) else f"{val:.6g}"
        elif key in V:
            V[key] = val
        else:
            raise ValueError(f"unknown surface field {key!r}")
    return {"name": state.get("name", "?"), "entries": E, "vars": V}


def parse_surface_state(E, V):
    """E: {key: text}, V: {key: value} -> VLMSurface kwargs (+_shedding_forced).
    Same parsing as the Surfaces tab (expressions allowed)."""
    D = SURFACE_DEFAULTS
    n_pre = int(eval_expr(E["n"]))
    m = int(eval_expr(E["m"]))
    p = dict(
        origin=np.array([eval_expr(E["ox"]), eval_expr(E["oy"]), eval_expr(E["oz"])]),
        plan=int(V["plan"]),
        boundary=V["boundary"],
        shedding=V["shedding"],
        b=eval_expr(E["span"]),
        alpha=eval_expr(E["alpha"]) * DEG,
        beta=eval_expr(E["beta"]) * DEG,
        lamb=eval_expr(E["sweep"]) * DEG,
        delta=eval_expr(E["dihedral"]) * DEG,
        sym=bool(V["sym"]), space=bool(V["space"]),
        n=n_pre, m=m, vent=bool(V["vent"]),
        root_ang=eval_expr(E["root_ang"]), tip_ang=eval_expr(E["tip_ang"]),
        radius=eval_expr(E["radius"]), center=eval_expr(E["center"]),
        fly_height=eval_expr(E["fly_h"]),
        c_line=eval_expr(str(E.get("c_line", "0.5") or "0.5")),
    )
    tw = str(E["twist"]).strip()
    rake = eval_expr(str(E.get("rake", "0") or "0").strip() or "0") * DEG
    if "," in tw or " " in tw.strip():
        arr = np.array([eval_expr(x) for x in tw.replace(",", " ").split()]) * DEG
        if len(arr) != n_pre + 1:
            raise ValueError(f"twist list: {len(arr)} values, need n+1={n_pre + 1}")
        if p["sym"]:
            # the library doubles N on merge and indexes one twist value per
            # merged station: mirror the half-wing list (a plain n+1 list
            # raises IndexError in _apply_twist)
            arr = np.concatenate([arr[::-1][:-1], arr])
        p["phi"] = arr + rake
    elif rake == 0.0:
        p["phi"] = eval_expr(tw) * DEG
    else:
        # scalar twist is a linear root->tip ramp in the library: discretize it
        # the same way, then shift every station by the uniform rake
        w = linear_twist_weights(n_pre, p["b"], p["sym"], p["space"],
                                 p["root_ang"], p["tip_ang"], p["radius"],
                                 p["center"], p["delta"])
        p["phi"] = w * (eval_expr(tw) * DEG) + rake
    mode = V["chord_mode"]
    if mode == "uniform":
        p["c"] = build_chord_array("uniform", "rectangular", 1.0, p["b"], n_pre,
                                   p["sym"], p["space"], 1.8,
                                   eval_or(E["c0"], D["c0"]))
    elif mode == "custom":
        raw = str(E.get("chord", "") or "")
        vals = [eval_expr(x) for x in raw.replace(",", " ").split()]
        if len(vals) != n_pre + 1:
            raise ValueError(f"custom chord: {len(vals)} values, need n+1={n_pre + 1}")
        if any(v <= 0 for v in vals):
            raise ValueError("custom chord: all values must be > 0")
        p["c"] = np.array(vals)
    else:
        p["c"] = build_chord_array("law", mode, eval_or(E["AR"], D["AR"]), p["b"],
                                   n_pre, p["sym"], p["space"],
                                   eval_or(E["lam"], D["lam"]), 0.0)
    # safety: elliptical laws taper to a zero tip chord -> no tip shedding
    forced = enforce_shedding(mode, p["shedding"])
    p["_shedding_forced"] = forced != p["shedding"]
    p["shedding"] = forced
    return p


def make_surface(p):
    """Build a VLMSurface from parse_surface_state() output."""
    return VLMSurface(origin=p["origin"], plan=p["plan"], boundary=p["boundary"],
                      shedding=p["shedding"], b=p["b"], c=p["c"], alpha=p["alpha"],
                      beta=p["beta"], lamb=p["lamb"], delta=p["delta"], phi=p["phi"],
                      sym=p["sym"], space=p["space"], n=p["n"], m=p["m"], vent=p["vent"],
                      root_ang=p["root_ang"], tip_ang=p["tip_ang"], radius=p["radius"],
                      center=p["center"], fly_height=p["fly_height"],
                      c_line=p.get("c_line", 0.5))


def build_study_surface(state, overrides):
    """Study-run surface: preset form-state + overrides -> built VLMSurface."""
    st = apply_state_overrides(state, overrides)
    return make_surface(parse_surface_state(st["entries"], st["vars"]))


def surface_dist(s):
    """Distributions of a solved surface (lists, CSV/JSON friendly):
    centered curvilinear abscissa s (/b), spanwise circulation gamma (|sum|
    chordwise, like plot_gamma, one value per section), Cl_2d, sectional
    normal loads, total loads (None when missing). ld is l/l_max (the lib
    normalizes loads_distrib in _kuttas_loads); gamma is left raw here and
    normalized at plot time."""
    try:
        sco = [float(v) for v in np.asarray(span_coord(s), dtype=float)]
    except Exception:
        try:
            sco = [float(v) for v in np.asarray(s._spanwise_coord(), dtype=float)]
        except Exception:
            sco = None
    g = None
    if s.gamma is not None and np.size(s.gamma):
        try:
            m, n = s.M, s.N
            g = [float(v) for v in np.abs(np.sum(np.asarray(s.gamma, dtype=float).reshape(m, n), axis=0))]
        except (ValueError, TypeError):
            g = None
    _cl = cl2d_array(s)
    cl = [float(v) for v in _cl] if _cl is not None else None
    lo = [float(v) for v in np.ravel(np.asarray(s.loads, dtype=float))] \
        if s.loads is not None else None
    ld = None
    if getattr(s, "loads_distrib", None) is not None:
        try:
            ld = [float(v) for v in np.ravel(np.asarray(s.loads_distrib, dtype=float))]
        except (ValueError, TypeError):
            ld = None
    return {"s": sco, "gamma": g, "cl": cl, "loads": lo, "ld": ld}


def forced_speed_mode(targets):
    """targets: [(scope, key)]. 'fn' forces Froude (U along x),
    Ux/Uy/Uz force speed mode; both together is an error."""
    solv = [k for s, k in targets if s == "solver"]
    use_fn = "fn" in solv
    use_u = any(k in ("ux", "uy", "uz") for k in solv)
    if use_fn and use_u:
        raise ValueError("cannot sweep Fn_h together with Ux/Uy/Uz (Froude forces U along x).")
    if use_fn:
        return "Froude Fn_h"
    return None


LATEX = {"alpha": r"$\alpha$", "beta": r"$\beta$", "sweep": r"$\lambda$",
         "dihedral": r"$\delta$", "twist": r"$\phi$", "rake": r"$rake$",
         "c": r"$c$", "c_line": r"$c_{line}$", "span": r"$b$",
         "c0": r"$c_0$", "AR": r"$A_R$", "CL": r"$C_L$",
         "t": r"$t$", "dt": r"$dt$", "fly_h": r"$h$",
         "Fn_h": r"$F_{n_h}$", "Ux": r"$U_x$", "Uy": r"$U_y$", "Uz": r"$U_z$",
         "Fz": r"$F_z$", "Fy": r"$F_y$", "Fx": r"$F_x$"}


def tex(short):
    """LaTeX plot label for a sweep-field short name (CSV/log stay plain)."""
    return LATEX.get(short, short)


def field_label(short, surf, all_flag, unit, use_tex=False):
    s = tex(short) if use_tex else short
    pre = f"{s} ({surf})" if surf else (f"{s} (all)" if all_flag else s)
    return f"{pre} [{unit}]" if unit not in ("", "-") else pre


def fmt_val(v):
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        return f"{v:g}"
    return str(v)


def fmt_fr(v):
    """French CSV number: semicolon-safe, comma decimals (like data/*.csv)."""
    if v is None:
        return ""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        if isinstance(v, float) and np.isnan(v):
            return ""
        return f"{v:.6g}".replace(".", ",")
    return str(v)


def write_csv(path, header, rows):
    """Semicolon-separated CSV with comma decimals + header line."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(";".join(header) + "\n")
        for row in rows:
            f.write(";".join(fmt_fr(v) for v in row) + "\n")


def thin_indices(n, xmax):
    """Evenly spaced indices in [0, n), at most xmax curves, always keeping
    the first and the last value. Invalid xmax -> all indices."""
    if n <= 0:
        return []
    try:
        xmax = int(xmax)
    except (TypeError, ValueError):
        return list(range(n))
    if n <= xmax:
        return list(range(n))
    if xmax < 2:
        return sorted({0, n - 1})
    return sorted(set(round(i * (n - 1) / (xmax - 1)) for i in range(xmax)))


def vals_text(vals, kind="float"):
    s = ", ".join(sweep_text(kind, v) for v in vals[:8])
    return "[" + s + (", …" if len(vals) > 8 else "") + "]"


def sweep_text(kind, v):
    if kind == "set":
        return "+".join(v)
    return fmt_val(v)


def broadcast_targets(fz_text, fy_text, n):
    """Parse trim load-target entries (";"/ranges like sweeps). A single
    value is broadcast to n points, otherwise lengths must match n."""
    fz = parse_sweep_values(fz_text, "float")
    fy = parse_sweep_values(fy_text, "float")
    if len(fz) == 1:
        fz = fz * n
    if len(fy) == 1:
        fy = fy * n
    if len(fz) != n or len(fy) != n:
        raise ValueError(f"load targets need 1 or {n} values (Fz: {len(fz)}, Fy: {len(fy)})")
    return fz, fy


def csv_val(raw, tick):
    """CSV cell: numbers stay numeric (comma decimals via write_csv),
    anything else uses the display string."""
    if isinstance(raw, bool):
        return tick
    if isinstance(raw, (int, float)):
        return raw
    return tick


def _cl_missing(v):
    """True when a result cell holds no usable CL (surface not in this run's
    composition, unticked, or failed point) — such rows are skipped on export."""
    return v is None or (isinstance(v, (int, float)) and not isinstance(v, bool)
                         and np.isnan(float(v)))


def parse_solver_dict(d):
    """d: solver form strings -> (kw, t, dt, dist). Pure (worker-safe)."""
    kw = dict(ratio=eval_expr(d["ratio"]), a_ratio=eval_expr(d["a_ratio"]),
              k_ratio=eval_expr(d["k_ratio"]))
    return kw, eval_or(d["t"], None), eval_or(d["dt"], None), d["dist"]


def flow_U_from(d, named):
    """d: solver form strings, named: [(name, built surface)].
    Returns (U [m/s], info text). Pure (worker-safe). Froude mode uses the
    max immersion over all surfaces (z max on wing['real']) and forces U along x."""
    if d["speed_mode"] == "Froude Fn_h":
        fn = eval_expr(d["fn"])
        h = max(float(np.asarray(s.wing["real"], dtype=float)[:, 2].max()) for _, s in named)
        if h <= 0:
            raise ValueError(f"max immersion h={h:.3f} m must be > 0 in Froude mode.")
        u = float(fn) * np.sqrt(GRAV * h)
        return np.array([u, 0.0, 0.0]), f"Fn_h={fn:g} h_max={h:.3f} m -> U=[{u:.3f}, 0, 0] m/s"
    unit = d["speed_unit"]
    U = np.array([speed_to_si(d["ux"], unit),
                  speed_to_si(d["uy"], unit),
                  speed_to_si(d["uz"], unit)])
    return U, f"U=[{U[0]:.3f}, {U[1]:.3f}, {U[2]:.3f}] {unit} ({np.linalg.norm(U):.3f} m/s)"


def build_solver_param_fields(app, parent):
    """Solver-parameter widgets with VLMSolver defaults (shared Study/Trim).
    Returns (entries, vars, next_row). Keys: ux,uy,uz,fn,ratio,a_ratio,k_ratio,
    t,dt + speed_unit,speed_mode,dist."""
    S, SV = {}, {}
    r = 0
    ttk.Label(parent, text="flow speed U").grid(row=r, column=0, sticky="w", padx=4, pady=1)
    SV["speed_unit"] = tk.StringVar(value="m/s")
    ttk.Combobox(parent, textvariable=SV["speed_unit"], values=["m/s", "kt"], width=6, state="readonly").grid(row=r, column=1, sticky="w", padx=4, pady=1); r += 1
    S["ux"] = app._entry(parent, r, "U x", SOLVER_DEFAULTS["ux"]); r += 1
    S["uy"] = app._entry(parent, r, "U y", SOLVER_DEFAULTS["uy"]); r += 1
    S["uz"] = app._entry(parent, r, "U z", SOLVER_DEFAULTS["uz"]); r += 1
    ttk.Label(parent, text="imposed").grid(row=r, column=0, sticky="w", padx=4, pady=1)
    SV["speed_mode"] = tk.StringVar(value="speed U")
    ttk.Combobox(parent, textvariable=SV["speed_mode"], values=["speed U", "Froude Fn_h"], width=10, state="readonly").grid(row=r, column=1, padx=4, pady=1); r += 1
    S["fn"] = app._entry(parent, r, "Fn_h (max immersion)", ""); r += 1
    S["ratio"] = app._entry(parent, r, "ratio (wake rollup)", SOLVER_DEFAULTS["ratio"]); r += 1
    S["a_ratio"] = app._entry(parent, r, "a_ratio (lin. system)", SOLVER_DEFAULTS["a_ratio"]); r += 1
    S["k_ratio"] = app._entry(parent, r, "k_ratio (kutta)", SOLVER_DEFAULTS["k_ratio"]); r += 1
    S["t"] = app._entry(parent, r, "t [s] (blank=auto)", ""); r += 1
    S["dt"] = app._entry(parent, r, "dt [s] (blank=auto)", ""); r += 1
    ttk.Label(parent, text="time distribution").grid(row=r, column=0, sticky="w", padx=4, pady=1)
    SV["dist"] = tk.StringVar(value=SOLVER_DEFAULTS["distribution"])
    ttk.Combobox(parent, textvariable=SV["dist"], values=["classic", "cosine"], width=10, state="readonly").grid(row=r, column=1, padx=4, pady=1); r += 1
    return S, SV, r


def snapshot_param_dict(S, SV):
    """entries/vars dicts -> plain solver-form strings (UI thread only)."""
    d = {k: S[k].get() for k in ("ux", "uy", "uz", "fn", "ratio", "a_ratio", "k_ratio", "t", "dt")}
    d.update(speed_unit=SV["speed_unit"].get(), speed_mode=SV["speed_mode"].get(),
             dist=SV["dist"].get())
    return d


# ---------- drawing on embedded axes (same style as Utilities.plot2D/plot3D) ----------

def _xy_for_axis(pts, axis):
    pts = np.asarray(pts, dtype=float)
    if axis == 0:      # back view (y-z)
        return pts[:, 1], pts[:, 2]
    if axis == 1:      # lateral view (x-z)
        return pts[:, 0], pts[:, 2]
    return pts[:, 1], pts[:, 0]  # top view (y-x), vu d'en haut


def _closed(pnts):
    pnts = list(pnts)
    return np.array(pnts + [pnts[0]], dtype=float)


def _nolabel(first, label):
    return label if first else "_nolegend_"


def draw_2d_panels(ax, named_surfs, axis, show_mirror=True, with_wake=False):
    """2D view mirroring Utilities.plot2D: black wing, red rings, blue wake,
    dimgray/forestgreen/darkviolet mirror, equal scale. No legend here,
    the 3D view carries it."""
    ax.clear()
    for _, surf in named_surfs:
        for panel in surf.wing_panels.get("real", []):
            x, y = _xy_for_axis(_closed(panel.pnt), axis)
            ax.plot(x, y, color="black", lw=0.8)
            xr, yr = _xy_for_axis(_closed(panel.vrt), axis)
            ax.plot(xr, yr, color="red", lw=0.8)
        if with_wake:
            for panel in surf.wake_panels.get("real", []):
                x, y = _xy_for_axis(_closed(panel.pnt), axis)
                ax.plot(x, y, color="blue", lw=0.6)
        if show_mirror and surf.wing_panels.get("mirror"):
            for panel in surf.wing_panels["mirror"]:
                x, y = _xy_for_axis(_closed(panel.pnt), axis)
                ax.plot(x, y, color="dimgray", lw=0.6)
                xr, yr = _xy_for_axis(_closed(panel.vrt), axis)
                ax.plot(xr, yr, color="forestgreen", lw=0.6)
            if with_wake:
                for panel in surf.wake_panels.get("mirror", []):
                    x, y = _xy_for_axis(_closed(panel.pnt), axis)
                    ax.plot(x, y, color="darkviolet", lw=0.6)
    if axis == 0:
        ax.set_xlabel(r"$y$ [m]"); ax.set_ylabel(r"$z$ [m]"); ax.set_title("Front view")
        ax.margins(0.3)
    elif axis == 1:
        ax.set_xlabel(r"$x$ [m]"); ax.set_ylabel(r"$z$ [m]"); ax.set_title("Lateral view")
    else:
        ax.set_xlabel(r"$y$ [m]"); ax.set_ylabel(r"$x$ [m]"); ax.set_title("Top view")
    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    # air out the numbering only: tick spacing, never the scales
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, prune="both"))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, prune="both"))
    _balance_2d_ticks(ax)


def draw_3d_panels(ax, named_surfs, show_mirror=False, with_wake=True, title="3D view", equal=True,
                 legend_out=False):
    """3D view mirroring Utilities.plot3D: same colors, z inverted.
    equal=True keeps physical scales (surface preview); equal=False lets the
    data fill the axes. title=None skips it. legend_out=True moves the legend
    right, outside the box, so the box holds data only."""
    ax.clear()
    f = {"wing": True, "rings": True, "wake": True, "mwing": True, "mrings": True, "mwake": True}
    any_lab = False
    for _, surf in named_surfs:
        for panel in surf.wing_panels.get("real", []):
            p = _closed(panel.pnt)
            ax.plot(p[:, 1], p[:, 0], p[:, 2], color="black", lw=0.7, label=_nolabel(f["wing"], "Wing"))
            any_lab = any_lab or f["wing"]; f["wing"] = False
            v = _closed(panel.vrt)
            ax.plot(v[:, 1], v[:, 0], v[:, 2], color="red", lw=0.8, label=_nolabel(f["rings"], "Vortex rings"))
            any_lab = any_lab or f["rings"]; f["rings"] = False
        if with_wake:
            for panel in surf.wake_panels.get("real", []):
                p = _closed(panel.pnt)
                ax.plot(p[:, 1], p[:, 0], p[:, 2], color="blue", lw=0.6, label=_nolabel(f["wake"], "Wake"))
                any_lab = any_lab or f["wake"]; f["wake"] = False
        if show_mirror and surf.wing_panels.get("mirror"):
            for panel in surf.wing_panels["mirror"]:
                p = _closed(panel.pnt)
                ax.plot(p[:, 1], p[:, 0], p[:, 2], color="dimgray", lw=0.6, label=_nolabel(f["mwing"], "Mirrored wing"))
                any_lab = any_lab or f["mwing"]; f["mwing"] = False
                v = _closed(panel.vrt)
                ax.plot(v[:, 1], v[:, 0], v[:, 2], color="forestgreen", lw=0.6, label=_nolabel(f["mrings"], "Mirrored rings"))
                any_lab = any_lab or f["mrings"]; f["mrings"] = False
            if with_wake:
                for panel in surf.wake_panels.get("mirror", []):
                    p = _closed(panel.pnt)
                    ax.plot(p[:, 1], p[:, 0], p[:, 2], color="darkviolet", lw=0.6, label=_nolabel(f["mwake"], "Mirrored wake"))
                    any_lab = any_lab or f["mwake"]; f["mwake"] = False
    ax.view_init(elev=30, azim=40)
    if equal:
        try:
            x_lim = ax.get_xlim3d(); y_lim = ax.get_ylim3d(); z_lim = ax.get_zlim3d()
            ax.set_xlim(x_lim[0], x_lim[1])
            ax.set_ylim(y_lim[0], y_lim[1])
            ax.set_zlim(z_lim[1], z_lim[0])
            sizes = [abs(x_lim[1] - x_lim[0]), abs(y_lim[1] - y_lim[0]), abs(z_lim[1] - z_lim[0])]
            m = max(sizes)
            # a flat dimension (e.g. constant y on a straight vertical foil)
            # would collapse the equal box: negligible but nonzero thickness
            sizes = [v if v > 1e-9 * m else 1e-3 * m for v in sizes]
            ax.set_box_aspect(sizes)
        except Exception:
            pass
    else:
        try:
            z_lim = ax.get_zlim3d()
            ax.set_zlim(z_lim[1], z_lim[0])
        except Exception:
            pass
    ax.set_xlabel("y", style="italic")
    ax.set_ylabel("x", style="italic")
    ax.set_zlabel("z", style="italic")
    if title:
        ax.set_title(title)
    # air out the numbering only: tick spacing, never the scales
    _balance_3d_ticks(ax)
    if any_lab:
        if legend_out:
            ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5))
        else:
            ax.legend(fontsize=7)


def span_coord(surf):
    """Centered control-point spanwise positions, normalized by span (/b).

    Unlike VLMSurface._spanwise_coord (arclength from the first control
    point, so first point exactly 0 and last well below 1 on coarse meshes),
    the half end-panel widths are included: first point > 0, last < 1.
    Raises on failure (caller falls back).
    """
    m, n = surf.M, surf.N
    panels = surf.wing_panels["real"]
    if len(panels) != m * n:
        raise ValueError("panel count mismatch")
    ctrl = np.array([panels[i * n + j].ctr for j in range(n) for i in range(m)],
                    dtype=float).reshape(n, m, 3).mean(axis=1)[:, 1:]
    d = np.linalg.norm(np.diff(ctrl, axis=0), axis=1) if n > 1 else np.zeros(0)
    w0 = float(np.mean([np.linalg.norm(panels[i * n].width[1:]) for i in range(m)]))
    s = np.concatenate([[0.0], np.cumsum(d)])
    return (w0 / 2.0 + s) / surf.span


def _span_x(surf, n_fallback):
    try:
        return np.asarray(span_coord(surf), dtype=float)
    except Exception:
        pass
    try:
        return np.asarray(surf._spanwise_coord(), dtype=float)
    except Exception:
        return np.arange(n_fallback, dtype=float)


def draw_cl_single(ax, name, surf):
    """Sectional Cl distribution of one surface, styled like plot_Cl_2d:
    plain line, no markers, no legend, no title (the row header names the
    surface, Froude numbers are in the console)."""
    ax.clear()
    cl = cl2d_array(surf)
    if cl is None:
        return
    ax.plot(_span_x(surf, len(cl)), cl)
    ax.set_xlabel(r"$y/b$" if surf.plan == 1 else r"$z/b$")
    ax.set_ylabel(r"$C_l$")
    mx = float(np.max(cl)) if len(cl) else 0.0
    if mx > 0:
        ax.set_ylim(0, 1.1 * mx)
    ax.grid(True)


def draw_gamma_single(ax, name, surf):
    """Spanwise circulation of one surface, styled like plot_gamma:
    normalized by max, plain line, no markers, no legend, no title."""
    ax.clear()
    if surf.gamma is None or np.size(surf.gamma) == 0:
        return
    m, n = surf.M, surf.N
    circ = np.abs(np.sum(np.asarray(surf.gamma, dtype=float).reshape(m, n), axis=0))
    mx = float(np.max(circ)) if len(circ) else 0.0
    if mx <= 0:
        return
    ax.plot(_span_x(surf, len(circ)), circ / mx)
    ax.set_xlabel(r"$y/b$" if surf.plan == 1 else r"$z/b$")
    ax.set_ylabel(r"$\Gamma/\Gamma_\text{\max}$")
    ax.set_ylim(0, 1.1)
    ax.grid(True)


def draw_vent_single(ax, name, surf, curves=None):
    """Ventilation curves of one surface, styled like plot_ventilation
    (same colors, mathtext labels and font). curves selects which of
    Lc/c, Cl_2d, alphaeff/alpha, a0/2pi, sigma_c to show. y (z/b) inverted
    like a depth axis, on the curvilinear span coordinate as in the Cl plot."""
    ax.clear()
    vent = getattr(surf, "ventilation", {}) or {}
    if not vent or "Lc/c" not in vent:
        return
    n = len(np.asarray(vent["Lc/c"]))
    y = _span_x(surf, n)
    show = {"lc": True, "cl": True, "alpha": True, "a0": True, "sigma": True}
    if curves:
        show.update(curves)
    if show["lc"]:
        ax.plot(np.asarray(vent["Lc/c"], dtype=float), y, color="gray", label=r"$L_c/c$")
    if "a0" in vent and "alpha_eff" in vent:
        a0 = np.asarray(vent["a0"], dtype=float)
        ae = np.asarray(vent["alpha_eff"], dtype=float)
        if show["cl"]:
            ax.plot(a0 * np.sin(ae), y, color="black", label=r"$C_{l_\text{2D}}$")
        if show["alpha"] and "avg_aoa" in vent and float(vent["avg_aoa"]) != 0:
            ax.plot(ae / float(vent["avg_aoa"]), y, color="red", label=r"$\alpha_{\text{eff}}/\alpha$")
        if show["a0"]:
            ax.plot(a0 / (2 * np.pi), y, color="magenta", label=r"$a_0/(2\pi)$")
    if show["sigma"] and "sigma_c" in vent:
        ax.plot(np.asarray(vent["sigma_c"], dtype=float), y, color="blue", label=r"$\sigma_c$")
    ax.set_xlabel(r"$x/c$")
    ax.set_ylabel(r"$y/b$" if surf.plan == 1 else r"$z/b$")
    ax.set_xlim(-0.2, 1.9)
    ax.invert_yaxis()
    ax.grid(True)
    ax.legend(fontsize=7)


# ---------- per-surface tab ----------

class SurfaceTab(ttk.Frame):
    def __init__(self, master, app, name):
        super().__init__(master)
        self.app = app
        self.name = name
        self.built = None  # VLMSurface after Build
        self.vars = {}
        self._build_form()
        self._build_chord_window()
        self._build_preview()

    # -- form --
    def _field(self, parent, row, label, default="", width=12):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=1)
        e = ttk.Entry(parent, width=width)
        e.insert(0, str(default))
        e.grid(row=row, column=1, sticky="ew", padx=4, pady=1)
        return e

    def _build_form(self):
        left = ttk.Frame(self)
        left.pack(side="left", fill="y", padx=6, pady=6)
        canvas = tk.Canvas(left, width=400, highlightthickness=0)
        scroll = ttk.Scrollbar(left, orient="vertical", command=canvas.yview)
        form = ttk.Frame(canvas)
        form.bind("<Configure>", lambda ev: canvas.configure(scrollregion=canvas.bbox("all")))
        _win = canvas.create_window((0, 0), window=form, anchor="nw")
        canvas.bind("<Configure>", lambda ev: canvas.itemconfig(_win, width=ev.width))
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side="left", fill="y", expand=True)
        scroll.pack(side="right", fill="y")
        form.columnconfigure(1, weight=1)

        E = self.entries = {}
        V = self.vars
        D = SURFACE_DEFAULTS
        r = 0
        E["ox"] = self._field(form, r, "origin x [m]", D["ox"]); r += 1
        E["oy"] = self._field(form, r, "origin y [m]", D["oy"]); r += 1
        E["oz"] = self._field(form, r, "origin z [m]", D["oz"]); r += 1

        ttk.Label(form, text="plan (0 vertical z-x / 1 horizontal y-x)").grid(row=r, column=0, sticky="w", padx=4, pady=1)
        V["plan"] = tk.StringVar(value=str(D["plan"])); ttk.Combobox(form, textvariable=V["plan"], values=["0", "1"], width=10, state="readonly").grid(row=r, column=1, padx=4, pady=1); r += 1
        ttk.Label(form, text="boundary (mirror)").grid(row=r, column=0, sticky="w", padx=4, pady=1)
        V["boundary"] = tk.StringVar(value=D["boundary"]); ttk.Combobox(form, textvariable=V["boundary"], values=["antisymmetric", "symmetric", "none"], width=10, state="readonly").grid(row=r, column=1, padx=4, pady=1); r += 1
        ttk.Label(form, text="tip shedding").grid(row=r, column=0, sticky="w", padx=4, pady=1)
        V["shedding"] = tk.StringVar(value=D["shedding"]); ttk.Combobox(form, textvariable=V["shedding"], values=["both", "left", "right", "none"], width=10, state="readonly").grid(row=r, column=1, padx=4, pady=1); r += 1

        E["span"] = self._field(form, r, "span b [m]", D["span"]); r += 1
        ttk.Button(form, text="Chord parameters", command=self.open_chord).grid(row=r, column=0, columnspan=2, pady=4); r += 1

        E["alpha"] = self._field(form, r, "alpha [deg]", D["alpha"]); r += 1
        E["beta"] = self._field(form, r, "beta [deg]", D["beta"]); r += 1
        E["sweep"] = self._field(form, r, "sweep [deg]", D["sweep"]); r += 1
        E["dihedral"] = self._field(form, r, "dihedral [deg]", D["dihedral"]); r += 1
        E["twist"] = self._field(form, r, "twist [deg]", D["twist"]); r += 1
        E["rake"] = self._field(form, r, "rake [deg] (uniform twist offset)", D["rake"]); r += 1
        E["n"] = self._field(form, r, "n spanwise (pre-symmetry)", D["n"]); r += 1
        E["m"] = self._field(form, r, "m chordwise", D["m"]); r += 1
        for key, label in [("sym", "symmetric (mirror + merge)"), ("space", "cosine spacing"), ("vent", "ventilated")]:
            V[key] = tk.BooleanVar(value=D[key])
            ttk.Checkbutton(form, text=label, variable=V[key]).grid(row=r, column=0, columnspan=2, sticky="w", padx=4, pady=1); r += 1
        E["root_ang"] = self._field(form, r, "root angle [deg]", D["root_ang"]); r += 1
        E["tip_ang"] = self._field(form, r, "tip angle [deg]", D["tip_ang"]); r += 1
        E["radius"] = self._field(form, r, "radius [m]", D["radius"]); r += 1
        E["center"] = self._field(form, r, "center [0-1]", D["center"]); r += 1
        E["fly_h"] = self._field(form, r, "fly height H [m]", D["fly_h"]); r += 1
        V["show_mirror"] = tk.BooleanVar(value=False)
        ttk.Checkbutton(form, text="show mirror (boundary)", variable=V["show_mirror"],
                        command=self._redraw_preview).grid(row=r, column=0, columnspan=2, sticky="w", padx=4, pady=1); r += 1

        ttk.Button(form, text="Build (init + plot)", command=self.on_build).grid(row=r, column=0, columnspan=2, pady=8)
        r += 1
        self.status = ttk.Label(form, text="not built", wraplength=320)
        self.status.grid(row=r, column=0, columnspan=2, padx=4, pady=4)
        self.app._bind_wheel(canvas, canvas)
        self._hook_form_wheel(form, canvas)

    def _build_chord_window(self):
        """Hidden chord-parameters window (same pattern as the Study/Trim
        'Solver parameters' windows). Widgets write into self.entries /
        self.vars, so form_state / set_form_state / parse are unchanged."""
        self.ch_win = tk.Toplevel(self)
        self.ch_win.title("Chord parameters")
        self.ch_win.protocol("WM_DELETE_WINDOW", self.ch_win.withdraw)
        win = ttk.Frame(self.ch_win, padding=10)
        win.pack(fill="both", expand=True)
        win.columnconfigure(1, weight=1)
        E, V, D = self.entries, self.vars, SURFACE_DEFAULTS
        r = 0
        ttk.Label(win, text="chord law").grid(row=r, column=0, sticky="w", padx=4, pady=1)
        V["chord_mode"] = tk.StringVar(value=D["chord_mode"]); ttk.Combobox(win, textvariable=V["chord_mode"], values=["uniform", "rectangular", "elliptical", "tapered", "custom"], width=10, state="readonly").grid(row=r, column=1, padx=4, pady=1); r += 1
        E["c0"] = self._field(win, r, "uniform c0 [m]", D["c0"]); r += 1
        E["AR"] = self._field(win, r, "AR (law)", D["AR"]); r += 1
        E["lam"] = self._field(win, r, "taper ratio (law)", D["lam"]); r += 1
        E["chord"] = self._field(win, r, "custom chord [m] (n+1 values)", D["chord"]); r += 1
        E["c_line"] = self._field(win, r, "chord line [0 LE - 1 TE]", D["c_line"]); r += 1
        V["chord_mode"].trace_add("write", lambda *a: self._on_chord_mode_change())
        self.ch_win.withdraw()

    def open_chord(self):
        self.ch_win.deiconify()
        self.ch_win.lift()

    def _hook_form_wheel(self, root, canvas):
        self.app._bind_wheel(root, canvas)
        for ch in root.winfo_children():
            self._hook_form_wheel(ch, canvas)

    def _build_preview(self):
        right = ttk.Frame(self)
        right.pack(side="right", fill="both", expand=True, padx=6, pady=6)
        self.figs, self.axes, self.canvases = [], [], []
        specs = [((3.4, 2.8), False, 0, 0, 1), ((3.4, 2.8), False, 0, 1, 1),
                 ((7.0, 3.0), True, 1, 0, 2)]
        for (w, h), is3d, r, c, cs in specs:
            fig = Figure(figsize=(w, h), constrained_layout=is3d)
            if not is3d:
                # no titles on 2D views: small bottom margin so the xlabel is not cropped
                fig.subplots_adjust(left=0.12, right=0.95, top=0.92, bottom=0.15)
            ax = fig.add_subplot(111, projection="3d") if is3d else fig.add_subplot(111)
            cv = FigureCanvasTkAgg(fig, master=right)
            cv.get_tk_widget().grid(row=r, column=c, columnspan=cs, sticky="nsew", padx=4, pady=4)
            self.figs.append(fig); self.axes.append(ax); self.canvases.append(cv)
        right.columnconfigure(0, weight=1)
        right.columnconfigure(1, weight=1)
        right.rowconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

    def read_params(self):
        E = {k: e.get() for k, e in self.entries.items()}
        V = {k: v.get() for k, v in self.vars.items()}
        return parse_surface_state(E, V)

    def _on_chord_mode_change(self):
        self.vars["shedding"].set(enforce_shedding(self.vars["chord_mode"].get(),
                                                   self.vars["shedding"].get()))

    def form_state(self):
        """JSON-serializable snapshot of the form (GUI units, expressions kept)."""
        return {"name": self.name,
                "entries": {k: e.get() for k, e in self.entries.items()},
                "vars": {k: v.get() for k, v in self.vars.items() if k != "show_mirror"}}

    def set_form_state(self, state):
        for k, e in self.entries.items():
            if k in state.get("entries", {}):
                e.delete(0, "end")
                e.insert(0, str(state["entries"][k]))
        for k, v in self.vars.items():
            if k in state.get("vars", {}):
                v.set(state["vars"][k])
        if state.get("name"):
            self.name = str(state["name"])
            try:
                self.app.surf_nb.tab(self, text=self.name)
            except Exception:
                pass
        self.built = None
        self.status.config(text="loaded (not built)")

    def on_build(self):
        try:
            p = self.read_params()
            surf = make_surface(p)
        except Exception as ex:
            messagebox.showerror("Build surface", str(ex))
            self.status.config(text=f"ERROR: {ex}")
            return
        err = emerged_bc_error(surf)
        if err is not None:
            messagebox.showerror("Build surface", err)
            self.status.config(text=f"ERROR: {err}")
            self.built = None
            self.app.refresh_solver_list()
            return
        self.built = surf
        self.app.refresh_solver_list()
        self._redraw_preview()
        note = " [tip shedding forced to none: elliptical law]" if p["_shedding_forced"] else ""
        self.status.config(text=f"OK: N={surf.N} M={surf.M} panels={len(surf.wing_panels['real'])}" + note)

    def _redraw_preview(self):
        if self.built is None:
            return
        named = [(self.name, self.built)]
        show_mirror = bool(self.vars["show_mirror"].get())
        ax0, ax_dyn = views_for_plan(self.built.plan)
        draw_2d_panels(self.axes[0], named, ax0, show_mirror=show_mirror)
        draw_2d_panels(self.axes[1], named, ax_dyn, show_mirror=show_mirror)
        draw_3d_panels(self.axes[2], named, show_mirror=show_mirror, with_wake=False)
        for fig, cv in zip(self.figs, self.canvases):
            cv.draw()


# ---------- main app ----------

class StudyTab(ttk.Frame):
    """Parametric study: cartesian sweep of 1-2 parameters (surface or solver
    fields), time_sim + kutta or ventilation per point, CL global per surface."""
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self.config = []      # [{"name", "entries", "vars"}]
        self.results = None   # per surface: [[cl(i1,i2)]]
        self.meta = None
        self.posgrid = None
        self._stop = False
        self._build()

    def _combo(self, parent, row, label, values, default=""):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=1)
        var = tk.StringVar(value=default)
        cb = ttk.Combobox(parent, textvariable=var, values=values, width=22, state="readonly")
        cb.grid(row=row, column=1, sticky="ew", padx=4, pady=1)
        return var, cb

    def _build(self):
        left_outer = ttk.Frame(self, width=380)
        left_outer.pack(side="left", fill="y", padx=6, pady=6)
        pcanvas = tk.Canvas(left_outer, width=360, highlightthickness=0)
        pscroll = ttk.Scrollbar(left_outer, orient="vertical", command=pcanvas.yview)
        left = ttk.Frame(pcanvas)
        left.bind("<Configure>", lambda ev: pcanvas.configure(scrollregion=pcanvas.bbox("all")))
        _pwin = pcanvas.create_window((0, 0), window=left, anchor="nw")
        pcanvas.bind("<Configure>", lambda ev: pcanvas.itemconfig(_pwin, width=ev.width))
        pcanvas.configure(yscrollcommand=pscroll.set)
        pscroll.pack(side="right", fill="y")
        pcanvas.pack(side="left", fill="both", expand=True)
        self.pcanvas = pcanvas
        left.columnconfigure(1, weight=1)
        r = 0
        ttk.Label(left, text="Study type").grid(row=r, column=0, sticky="w", padx=4, pady=1)
        self.c_type = tk.StringVar(value="time_sim + kutta")
        ttk.Combobox(left, textvariable=self.c_type, values=["time_sim + kutta", "ventilation"],
                     width=22, state="readonly").grid(row=r, column=1, padx=4, pady=1); r += 1
        ttk.Label(left, text="Base surfaces (ticked = in solver, rebuilt every run):").grid(row=r, column=0, columnspan=2, sticky="w", padx=4, pady=2); r += 1
        self.cfg_body = ttk.Frame(left)
        self.cfg_body.grid(row=r, column=0, columnspan=2, sticky="ew", padx=4, pady=2); r += 1
        brow = ttk.Frame(left)
        brow.grid(row=r, column=0, columnspan=2, pady=2); r += 1
        ttk.Button(brow, text="Add saved…", command=self.add_saved).pack(side="left", padx=2)
        ttk.Button(brow, text="Snapshot tabs", command=self.snapshot_tabs).pack(side="left", padx=2)
        ttk.Button(brow, text="Clear", command=self.clear_cfg).pack(side="left", padx=2)
        self.p1 = self._param_group(left, "Param 1 (x axis)", r, allow_none=False)
        r = self.p1["next_row"]
        self.p2 = self._param_group(left, "Param 2 (curves)", r, allow_none=True)
        r = self.p2["next_row"]
        ttk.Button(left, text="Solver parameters", command=self.open_study_solver).grid(row=r, column=0, columnspan=2, pady=4); r += 1
        ttk.Label(left, text="t/dt/ratios/U from the study solver parameters (VLMSolver defaults); sweep overrides apply per run.",
                  wraplength=340).grid(row=r, column=0, columnspan=2, sticky="w", padx=4, pady=4); r += 1
        self.runs_label = ttk.Label(left, text="runs: -")
        self.runs_label.grid(row=r, column=0, columnspan=2, sticky="w", padx=4, pady=1); r += 1
        bro = ttk.Frame(left)
        bro.grid(row=r, column=0, columnspan=2, pady=4); r += 1
        self.btn_run = ttk.Button(bro, text="Run study", command=self.run_study)
        self.btn_run.pack(side="left", padx=3)
        self.btn_cancel = ttk.Button(bro, text="Cancel", command=self.cancel_study, state="disabled")
        self.btn_cancel.pack(side="left", padx=3)
        erow = ttk.Frame(left)
        erow.grid(row=r, column=0, columnspan=2, pady=2); r += 1
        self.exp_dist = tk.BooleanVar(value=False)
        ttk.Checkbutton(erow, text="distributions", variable=self.exp_dist).pack(side="left", padx=2)
        ttk.Button(erow, text="Export CSV", command=self.export_csv).pack(side="left", padx=2)
        ttk.Label(left, text="Dist plots (stacked curves vs s)").grid(row=r, column=0, columnspan=2, sticky="w", padx=4, pady=(6, 1)); r += 1
        self.d_show = tk.BooleanVar(value=True)
        ttk.Checkbutton(left, text="show dist plots", variable=self.d_show,
                        command=self._draw_results).grid(row=r, column=0, columnspan=2, sticky="w", padx=4, pady=0); r += 1
        self.d_dist, _ = self._combo(left, r, "distribution", ["Cl_2d", "gamma", "loads"], "Cl_2d"); r += 1
        self.d_fix2, self.d_fix2_cb = self._combo(left, r, "fix P2 at", ["-"], "-"); r += 1
        self.d_fix1, self.d_fix1_cb = self._combo(left, r, "fix P1 at", ["-"], "-"); r += 1
        ttk.Label(left, text="max curves").grid(row=r, column=0, sticky="w", padx=4, pady=1)
        self.e_maxc = ttk.Entry(left, width=22)
        self.e_maxc.insert(0, "8")
        self.e_maxc.grid(row=r, column=1, sticky="ew", padx=4, pady=1); r += 1
        self.d_dist.trace_add("write", lambda *a: self._draw_results())
        for cb in (self.d_fix2_cb, self.d_fix1_cb):
            cb.bind("<<ComboboxSelected>>", lambda e: self._draw_results())
        self.study_bar = ttk.Progressbar(left, maximum=100)
        self.study_bar.grid(row=r, column=0, columnspan=2, sticky="ew", padx=4, pady=2); r += 1
        self.status = ttk.Label(left, text="idle", wraplength=340)
        self.status.grid(row=r, column=0, columnspan=2, sticky="w", padx=4, pady=2); r += 1
        self.log = tk.Text(left, height=10, width=46)
        self.log.grid(row=r, column=0, columnspan=2, padx=4, pady=4); r += 1
        self._hook_params_wheel(left, pcanvas)

        right = ttk.Frame(self)
        right.pack(side="right", fill="both", expand=True, padx=6, pady=6)
        ttk.Label(right, text="CL global vs param 1, one curve per param 2 value (scroll per surface)").pack(side="top", anchor="w", padx=4)
        self.scroll_wrap = ttk.Frame(right)
        self.scroll_wrap.pack(side="top", fill="both", expand=True)
        self.res_scroll = tk.Canvas(self.scroll_wrap, highlightthickness=0)
        vscroll = ttk.Scrollbar(self.scroll_wrap, orient="vertical", command=self.res_scroll.yview)
        hscroll = ttk.Scrollbar(self.scroll_wrap, orient="horizontal", command=self.res_scroll.xview)
        self.res_inner = ttk.Frame(self.res_scroll)
        self.res_inner.bind("<Configure>", lambda e: self.res_scroll.configure(scrollregion=self.res_scroll.bbox("all")))
        self.res_scroll.create_window((0, 0), window=self.res_inner, anchor="nw")
        self.res_scroll.configure(yscrollcommand=vscroll.set, xscrollcommand=hscroll.set)
        vscroll.pack(side="right", fill="y")
        hscroll.pack(side="bottom", fill="x")
        self.res_scroll.pack(side="left", fill="both", expand=True)
        self.app._bind_wheel(self.res_scroll, self.res_scroll)
        self.app._bind_wheel(self.res_inner, self.res_scroll)
        self._build_study_solver_window()

    def _hook_params_wheel(self, root, canvas):
        self.app._bind_wheel(canvas, canvas)
        if root.winfo_class() != "Text":
            self.app._bind_wheel(root, canvas)
        for ch in root.winfo_children():
            self._hook_params_wheel(ch, canvas)

    def _build_study_solver_window(self):
        """Hidden solver-parameters window for study (VLMSolver defaults)."""
        self.st_win = tk.Toplevel(self)
        self.st_win.title("Study solver parameters")
        self.st_win.protocol("WM_DELETE_WINDOW", self.st_win.withdraw)
        adv = ttk.Frame(self.st_win, padding=10)
        adv.pack(fill="both", expand=True)
        adv.columnconfigure(1, weight=1)
        S, SV, r = build_solver_param_fields(self.app, adv)
        ttk.Label(adv, text="medium (export loads)").grid(row=r, column=0, sticky="w", padx=4, pady=1)
        SV["medium"] = tk.StringVar(value="water")
        ttk.Combobox(adv, textvariable=SV["medium"], values=["water", "air"], width=10, state="readonly").grid(row=r, column=1, padx=4, pady=1); r += 1
        ttk.Label(adv, text="force unit (export loads)").grid(row=r, column=0, sticky="w", padx=4, pady=1)
        SV["force_unit"] = tk.StringVar(value="N")
        ttk.Combobox(adv, textvariable=SV["force_unit"], values=["N", "kgf"], width=10, state="readonly").grid(row=r, column=1, padx=4, pady=1); r += 1
        self.st_entries, self.st_vars = S, SV
        self.st_win.withdraw()

    def open_study_solver(self):
        self.st_win.deiconify()
        self.st_win.lift()

    def study_snapshot(self):
        """Study solver form as plain strings (UI thread only)."""
        d = snapshot_param_dict(self.st_entries, self.st_vars)
        d.update(medium=self.st_vars["medium"].get(), force_unit=self.st_vars["force_unit"].get())
        return d

    def _param_group(self, parent, title, row, allow_none):
        ttk.Label(parent, text=title).grid(row=row, column=0, columnspan=2, sticky="w", padx=4, pady=(6, 1)); row += 1
        fields = ["(none)"] + sorted(STUDY_FIELDS)
        surf_var, surf_cb = self._combo(parent, row, "surface", ["-"], "-"); row += 1
        field_var, _ = self._combo(parent, row, "field", fields, "(none)" if allow_none else "surface: alpha [deg]"); row += 1
        ttk.Label(parent, text="values (a; b / a:b:c)").grid(row=row, column=0, sticky="w", padx=4, pady=1)
        vals = ttk.Entry(parent, width=24)
        vals.grid(row=row, column=1, sticky="ew", padx=4, pady=1); row += 1
        info = ttk.Label(parent, text="")
        info.grid(row=row, column=0, columnspan=2, sticky="w", padx=4, pady=0); row += 1
        g = {"surf": surf_var, "surf_cb": surf_cb, "field": field_var, "vals": vals, "info": info}
        field_var.trace_add("write", lambda *a, g=g: self._on_field_change(g))
        vals.bind("<FocusOut>", lambda e: self._refresh_counts())
        self._on_field_change(g)
        return dict(g, next_row=row)

    def _on_field_change(self, g):
        label = g["field"].get()
        is_none = label == "(none)"
        is_solv = (not is_none) and STUDY_FIELDS[label][0] == "solver"
        g["surf_cb"].config(state="disabled" if (is_none or is_solv) else "readonly")
        g["vals"].config(state="disabled" if is_none else "normal")
        self._refresh_counts()

    # -- base config --
    def _active(self):
        return [c for c in self.config if c.get("use", True)]

    def _refresh_cfg(self):
        for w in self.cfg_body.winfo_children():
            w.destroy()
        for i, st in enumerate(self.config):
            row = ttk.Frame(self.cfg_body)
            row.pack(fill="x")
            var = tk.BooleanVar(value=bool(st.get("use", True)))
            ttk.Checkbutton(row, variable=var,
                            command=lambda i=i, var=var: self._toggle_use(i, var)).pack(side="left")
            ttk.Label(row, text=st["name"]).pack(side="left", padx=4)
            ttk.Button(row, text="x", width=2, command=lambda i=i: self.remove_at(i)).pack(side="right")
        names = [c["name"] for c in self._active()]
        for g in (self.p1, self.p2):
            cur = g["surf"].get()
            g["surf_cb"].config(values=["ALL"] + names)
            if cur not in (["ALL"] + names):
                g["surf"].set("ALL")
        self._refresh_counts()

    def _toggle_use(self, i, var):
        self.config[i]["use"] = bool(var.get())
        self._refresh_cfg()

    def _unique_name(self, base):
        names = {c["name"] for c in self.config}
        if base not in names:
            return base
        i = 2
        while f"{base} ({i})" in names:
            i += 1
        return f"{base} ({i})"

    def remove_at(self, i):
        del self.config[i]
        self._refresh_cfg()

    def add_saved(self):
        from tkinter import filedialog as _fd
        path = _fd.askopenfilename(initialdir=self.app._surf_dir(), filetypes=[("JSON", "*.json")])
        if not path:
            return
        try:
            state = load_preset(path)
            state["name"] = self._unique_name(state["name"])
            state["use"] = True
            self.config.append(state)
            self._refresh_cfg()
        except Exception as ex:
            messagebox.showerror("Add saved", str(ex))

    def snapshot_tabs(self):
        if not self.app.surf_tabs:
            messagebox.showerror("Snapshot", "no surface tabs")
            return
        for tab in self.app.surf_tabs:
            st = tab.form_state()
            st["name"] = self._unique_name(tab.name)
            st["use"] = True
            self.config.append(st)
        self._refresh_cfg()

    def remove_at(self, i):
        del self.config[i]
        self._refresh_cfg()

    def clear_cfg(self):
        self.config = []
        self._refresh_cfg()

    # -- sweep parsing --
    def _parse_sweep(self, g, num):
        label = g["field"].get()
        if label == "(none)":
            return None
        scope, key, kind, short, unit = STUDY_FIELDS[label]
        surf, all_flag = None, False
        if scope == "surface":
            if g["surf"].get() == "ALL":
                all_flag = True
            else:
                names = [c["name"] for c in self._active()]
                if g["surf"].get() not in names:
                    raise ValueError(f"param {num}: pick a base surface")
                surf = g["surf"].get()
        values = parse_sweep_values(g["vals"].get(), kind)
        if key == "__surfaceset__":
            pool = {c["name"] for c in self.config}
            for grp in values:
                for n in grp:
                    if n not in pool:
                        raise ValueError(f"param {num}: unknown preset {n!r} (add it to the base first)")
        if unit == "spd":
            unit = self.st_vars["speed_unit"].get()
        if key == "__surfaceset__":
            lab = "surface set"
        else:
            pre = f"{short} ({surf})" if surf else (f"{short} (all)" if all_flag else short)
            lab = f"{pre} [{unit}]" if unit not in ("", "-") else pre
        return dict(scope=scope, key=key, kind=kind, short=short, unit=unit,
                    surf=surf, all=all_flag, values=values, label=lab,
                    label_tex="surface set" if key == "__surfaceset__"
                    else field_label(short, surf, all_flag, unit, use_tex=True))

    def _refresh_counts(self):
        if not hasattr(self, "runs_label") or not hasattr(self, "p1") or not hasattr(self, "p2"):
            return
        try:
            p1 = self._parse_sweep(self.p1, 1)
            if p1 is None:
                n1 = 1
                self.p1["info"].config(text="[base]")
            else:
                self.p1["info"].config(text=vals_text(p1["values"], p1["kind"]))
                n1 = len(p1["values"])
        except Exception:
            self.runs_label.config(text="runs: ?")
            return
        try:
            p2 = self._parse_sweep(self.p2, 2)
        except Exception:
            self.p2["info"].config(text="?")
            self.runs_label.config(text="runs: ?")
            return
        if p2 is None:
            self.p2["info"].config(text="")
            n2 = 1
        else:
            self.p2["info"].config(text=vals_text(p2["values"], p2["kind"]))
            n2 = len(p2["values"])
        self.runs_label.config(text=f"runs: {n1} x {n2} = {n1 * n2}")

    # -- run --
    def set_status(self, txt):
        self.status.config(text=txt)

    def _log(self, txt):
        self.log.insert("end", txt + "\n")
        self.log.see("end")

    def cancel_study(self):
        self._stop = True
        self.set_status("cancelling…")

    def run_study(self):
        try:
            if not self._active():
                raise ValueError("no surface in solver: tick at least one base surface")
            p1 = self._parse_sweep(self.p1, 1)
            p2 = self._parse_sweep(self.p2, 2)
            if p1 is None and p2 is None:
                p1 = dict(scope="none", key="none", kind="str", short="base", unit="",
                          surf=None, all=False, values=["base"], label="base")
            elif p1 is None:
                raise ValueError("param 2 needs param 1 (set param 1 first)")
            if p2 is not None and (p1["scope"], p1["key"]) == (p2["scope"], p2["key"]) \
                    and (p1["all"] or p2["all"] or p1["surf"] == p2["surf"]):
                raise ValueError("param 1 and 2 overlap on the same field")
            nsets = sum(1 for p in (p1, p2) if p is not None and p["key"] == "__surfaceset__")
            if nsets > 1:
                raise ValueError("only one of param 1/2 can be the surface set")
            forced = forced_speed_mode([(p1["scope"], p1["key"])] +
                                       ([(p2["scope"], p2["key"])] if p2 else []))
            if (p1["scope"], p1["key"]) == ("surface", "fly_h") or \
                    (p2 is not None and (p2["scope"], p2["key"]) == ("surface", "fly_h")):
                self._log("note: fly_height sweep shifts origin z with fly_height (z+ downwards)")
            solver_base = self.study_snapshot()
            vent = self.c_type.get() == "ventilation"
            t_sweep = (p1["scope"], p1["key"]) == ("solver", "t") or \
                (p2 is not None and (p2["scope"], p2["key"]) == ("solver", "t"))
            if vent and t_sweep:
                raise ValueError("time sweep is not available in ventilation mode (internal time settings)")
            if vent:
                swept = {k for s, k in ([(p1["scope"], p1["key"])] + ([(p2["scope"], p2["key"])] if p2 else [])) if s == "solver"}
                if (swept & {"dt", "distribution"}) or solver_base["t"].strip() or solver_base["dt"].strip():
                    self._log("warning: ventilation uses internal time settings (t/dt/distribution ignored)")
            if nsets:
                self._log("note: surface-set sweep defines the per-run composition (ticks ignored)")
            n1 = len(p1["values"])
            n2 = len(p2["values"]) if p2 else 1
            x_num = p1["kind"] in ("float", "int")
            meta = dict(p1_label=p1["label"], x_num=x_num,
                        x_vals=list(p1["values"]),
                        x_ticks=[sweep_text(p1["kind"], v) for v in p1["values"]],
                        p1_tex=p1["label_tex"],
                        p1_short=p1["short"],
                        p2_label=(p2["label"] if p2 else "none"),
                        p2_ticks=([sweep_text(p2["kind"], v) for v in p2["values"]] if p2 else []),
                        p2_vals=(list(p2["values"]) if p2 else []),
                        p2_short=(p2["short"] if p2 else None),
                        surf_side=next((tag for tag, pp in (("p1", p1), ("p2", p2))
                                        if pp is not None and (pp["scope"], pp["key"]) == ("solver", "__surfaceset__")), None),
                        medium=solver_base.get("medium", "water"),
                        funit=solver_base.get("force_unit", "N"),
                        curve_labels=([f"{p2['short']}={sweep_text(p2['kind'], v)}" for v in p2["values"]] if p2 else []),
                        curve_tex=([f"{tex(p2['short'])}={sweep_text(p2['kind'], v)}" for v in p2["values"]] if p2 else []))
            if t_sweep := next((tag for tag, pp in (("p1", p1), ("p2", p2))
                                if pp is not None and (pp["scope"], pp["key"]) == ("solver", "t")), None):
                meta["t_side"] = t_sweep
            names = [c["name"] for c in self.config]
            status_txt = f"running {n1 * n2} points…"
            fast = None
            if t_sweep := next((tag for tag, pp in (("p1", p1), ("p2", p2))
                                if pp is not None and (pp["scope"], pp["key"]) == ("solver", "t")), None):
                fast, fnote = plan_time_fast(p1, p2, solver_base)
                if fnote is not None:
                    self._log(fnote)
            if fast is not None:
                solver_base = fast["base"]
                other = p2 if fast["tside"] == "p1" else p1
                ng = len(other["values"]) if other is not None else 1
                status_txt = f"running {ng} grouped runs ({n1 * n2} time points)…"
            payload = dict(config=[dict(name=c["name"], entries=dict(c["entries"]), vars=dict(c["vars"]),
                                        use=bool(c.get("use", True)))
                                   for c in self.config],
                           p1=dict(scope=p1["scope"], key=p1["key"], surf=p1["surf"],
                                   all=p1["all"], values=list(p1["values"])),
                           p2=(dict(scope=p2["scope"], key=p2["key"], surf=p2["surf"],
                                    all=p2["all"], values=list(p2["values"]))
                               if p2 else None),
                           vent=vent, solver_base=solver_base, forced=forced, meta=meta, names=names,
                           fast=fast)
        except Exception as ex:
            messagebox.showerror("Study", str(ex))
            return
        self.results = None
        self.meta = meta
        self._stop = False
        self.btn_run.config(state="disabled")
        self.btn_cancel.config(state="normal")
        self.study_bar["value"] = 0
        self.set_status(status_txt)
        threading.Thread(target=self._study_worker, args=(payload,), daemon=True).start()

    def _study_fast(self, pay, per_surf, dists, posgrid, pool, by_name, pidx, n1, n2, total, app, done):
        """Single-run time sweeps: one _time_sim per group of the non-t param.
        If dt is provided in fast dict (single group, backward compat), use it.
        Otherwise, compute dt per-group from geometry and adjust t values."""
        f = pay["fast"]
        tside = f["tside"]
        other = pay["p2"] if tside == "p1" else pay["p1"]
        ovals = other["values"] if other is not None else [None]
        
        # Store actual t values used per group for plotting
        actual_times_per_group = []
        
        # Check if we have pre-computed dt (backward compat) or need per-group
        has_precomputed = "dt" in f and "comps" in f and "times" in f
        
        for oi, w in enumerate(ovals):
            if self._stop:
                raise _StudyCancel()
            try:
                v1, v2 = (None, w) if tside == "p1" else (w, None)
                states = None
                for pp, vv in ((pay["p1"], v1), (pay["p2"], v2)):
                    if pp is not None and pp["scope"] == "solver" and pp["key"] == "__surfaceset__":
                        states = [by_name[n] for n in vv]
                if states is None:
                    states = [c for c in pool if c.get("use", True)]
                if not states:
                    raise ValueError("empty surface set for this run")
                surfs = []
                for st in states:
                    ovd = {}
                    if other is not None:
                        ovd.update(param_overrides(st, other, w))
                    surfs.append(build_study_surface(st, ovd))
                
                if has_precomputed:
                    # Backward compat: single group with pre-computed dt
                    dt = f["dt"]
                    comps = f["comps"]
                    t_run = f["t_run"]
                    order = f["order"]
                    times_actual = [f["times"][i] for i in order]
                    actual_times_per_group.append(times_actual)
                else:
                    # Per-group: compute dt from this group's geometry
                    # dt = 5 * c_mean / |U| / N_max (same as VLMSolver._time_sim)
                    # First compute U from the surfaces
                    sd = dict(f["base"])
                    if pay["forced"]:
                        sd["speed_mode"] = pay["forced"]
                    if other is not None and other["scope"] == "solver" \
                            and other["key"] != "__surfaceset__":
                        sd[other["key"]] = w if isinstance(w, str) else f"{w:.6g}"
                    named = [(st["name"], s) for st, s in zip(states, surfs)]
                    U, _info = flow_U_from(sd, named)
                    U_norm = float(np.linalg.norm(U))
                    c_mean = sum(float(np.mean(np.asarray(s.chord, dtype=float))) for s in surfs) / len(surfs)
                    N_max = max(s.N for s in surfs)
                    dt = 5.0 * c_mean / U_norm / N_max
                    
                    # Adjust requested t values to nearest multiples of dt
                    t_requested = f["t_requested"]
                    t_adjusted = []
                    ks = []
                    for tv in t_requested:
                        tv_f = float(tv)
                        k = max(1, int(round(tv_f / dt)))
                        t_adj = k * dt
                        t_adjusted.append(t_adj)
                        ks.append(k)
                    
                    # Check for duplicate k values (multiple t mapping to same step)
                    seen_k = set()
                    dup = [k for k in ks if k in seen_k or seen_k.add(k)]
                    if dup:
                        raise ValueError(
                            f"time sweep: requested times {t_requested} map to duplicate steps {dup} "
                            f"with auto dt={dt:.3g}. Increase t values or specify dt manually."
                        )
                    
                    # Sort by actual time
                    order = sorted(range(len(t_adjusted)), key=lambda i: t_adjusted[i])
                    comps = [ks[i] * dt for i in order]
                    t_run = max(t_adjusted)
                    times_actual = [t_adjusted[i] for i in order]
                    actual_times_per_group.append(times_actual)
                
                sd = dict(f["base"])
                if pay["forced"]:
                    sd["speed_mode"] = pay["forced"]
                if other is not None and other["scope"] == "solver" \
                        and other["key"] != "__surfaceset__":
                    sd[other["key"]] = w if isinstance(w, str) else f"{w:.6g}"
                named = [(st["name"], s) for st, s in zip(states, surfs)]
                U, _info = flow_U_from(sd, named)
                kw, _t, _dt, _dist = parse_solver_dict(sd)
                bc = common_boundary(named)
                vlm = VLMSolver(surfs, U, bc, ratio=kw["ratio"],
                                a_ratio=kw["a_ratio"], k_ratio=kw["k_ratio"])
                vlm._time_sim(t=t_run, dt=dt, distribution="classic",
                              computations=comps)
                CLt = surfs[0].CL
                if len(CLt) != len(comps) or any(len(c) != len(surfs) for c in CLt):
                    raise ValueError("time harvest mismatch: some computation times were skipped")
                CDt = surfs[0].Cl_2d
                if len(CDt) != len(comps) or any(len(c) != len(surfs) for c in CDt):
                    raise ValueError("time harvest mismatch: some computation times were skipped")
                
                # Map results using the actual times order
                for j, it in enumerate(order):
                    for a, st in enumerate(states):
                        pi = pidx[st["name"]]
                        v = float(CLt[j][a])
                        if tside == "p1":
                            per_surf[pi][it][oi] = v
                            posgrid[pi][it][oi] = a
                        else:
                            per_surf[pi][oi][it] = v
                            posgrid[pi][oi][it] = a
                vlm._kuttas_loads()
                itmax = order[-1]
                for a, s in enumerate(surfs):
                    pi = pidx[states[a]["name"]]
                    dmax = surface_dist(s)
                    s_list = dmax["s"] or []
                    if tside == "p1":
                        dists[pi][itmax][oi] = dmax
                    else:
                        dists[pi][oi][itmax] = dmax
                    for j, it in enumerate(order):
                        if it == itmax:
                            continue
                        try:
                            clj = [float(v) for v in CDt[j][a]]
                        except (TypeError, ValueError):
                            continue
                        cell = {"s": list(s_list), "gamma": None, "cl": clj, "loads": None}
                        if tside == "p1":
                            dists[pi][it][oi] = cell
                        else:
                            dists[pi][oi][it] = cell
                bits = []
                for a, st in enumerate(states):
                    bits.append(f"{st['name']} CL(t)=[{', '.join(f'{CLt[j][a]:.4f}' for j in range(len(order)))}]")
                app.msg_q.put(("study_log", True, (self, f"(group {oi + 1}/{len(ovals)}): " + "; ".join(bits))))
                done += len(comps)
                app.msg_q.put(("study_progress", True, (self, done, total)))
            except _StudyCancel:
                raise
            except Exception as ex:
                app.msg_q.put(("study_log", True, (self, f"(group {oi + 1}/{len(ovals)}) FAILED: {ex}")))
                # comps might not be defined if error occurred before assignment
                # skip progress increment for this group
        
        # Store actual times per group in the payload for plotting
        pay["actual_times_per_group"] = actual_times_per_group
        return done

    def _study_worker(self, pay):
        app = self.app
        n1 = len(pay["p1"]["values"])
        n2 = len(pay["p2"]["values"]) if pay["p2"] else 1
        total = n1 * n2
        pool = pay["config"]
        by_name = {c["name"]: c for c in pool}
        pidx = {c["name"]: i for i, c in enumerate(pool)}
        per_surf = [[[float("nan")] * n2 for _ in range(n1)] for _ in pool]
        dists = [[[None] * n2 for _ in range(n1)] for _ in pool]
        posgrid = [[[None] * n2 for _ in range(n1)] for _ in pool]

        def run_states(v1, v2):
            for pp, vv in ((pay["p1"], v1), (pay["p2"], v2)):
                if pp is not None and pp["scope"] == "solver" and pp["key"] == "__surfaceset__":
                    return [by_name[n] for n in vv]
            return [c for c in pool if c.get("use", True)]

        try:
            done = 0
            if pay.get("fast"):
                done = self._study_fast(pay, per_surf, dists, posgrid, pool, by_name, pidx,
                                        n1, n2, total, app, done)
            else:
                for i1, v1 in enumerate(pay["p1"]["values"]):
                    p2vals = pay["p2"]["values"] if pay["p2"] else [None]
                    for i2, v2 in enumerate(p2vals):
                        if self._stop:
                            raise _StudyCancel()
                        try:
                            states = run_states(v1, v2)
                            if not states:
                                raise ValueError("empty surface set for this run")
                            surfs = []
                            for st in states:
                                ov = {}
                                for pp, vv in ((pay["p1"], v1), (pay["p2"], v2)):
                                    ov.update(param_overrides(st, pp, vv))
                                surfs.append(build_study_surface(st, ov))
                            sd = dict(pay["solver_base"])
                            if pay["forced"]:
                                sd["speed_mode"] = pay["forced"]
                            for pp, vv in ((pay["p1"], v1), (pay["p2"], v2)):
                                if pp is not None and pp["scope"] == "solver" \
                                        and pp["key"] != "__surfaceset__":
                                    sd[pp["key"]] = vv if isinstance(vv, str) else f"{vv:.6g}"
                            named = [(st["name"], s) for st, s in zip(states, surfs)]
                            U, _info = flow_U_from(sd, named)
                            kw, t, dt, dist = parse_solver_dict(sd)
                            bc = common_boundary(named)
                            vlm = VLMSolver(surfs, U, bc, ratio=kw["ratio"],
                                            a_ratio=kw["a_ratio"], k_ratio=kw["k_ratio"])
                            if pay["vent"]:
                                vlm._ventilation()
                            else:
                                vlm._time_sim(t=t, dt=dt, distribution=dist)
                                vlm._kuttas_loads()
                            bits = []
                            for a, (st, s) in enumerate(zip(states, surfs)):
                                cl, _cd = surf_coeffs(s)
                                pi = pidx[st["name"]]
                                per_surf[pi][i1][i2] = float(cl) if cl is not None else float("nan")
                                posgrid[pi][i1][i2] = a
                                dists[pi][i1][i2] = surface_dist(s)
                                bits.append(f"{st['name']} CL={per_surf[pi][i1][i2]:.4f}")
                            app.msg_q.put(("study_log", True, (self, f"({done + 1}/{total}): " + "; ".join(bits))))
                        except _StudyCancel:
                            raise
                        except Exception as ex:
                            app.msg_q.put(("study_log", True, (self, f"({done + 1}/{total}) FAILED: {ex}")))
                        done += 1
                        app.msg_q.put(("study_progress", True, (self, done, total)))
            actual_times = pay.get("actual_times_per_group")
            app.msg_q.put(("study_done", True, (self, "done", pay["names"], per_surf, dists, posgrid, actual_times)))
        except _StudyCancel:
            app.msg_q.put(("study_done", True, (self, "cancelled (partial)", pay["names"], per_surf, dists, posgrid, None)))
        except Exception as ex:
            app.msg_q.put(("study_done", False, (self, f"ERROR: {ex}")))

    def _study_finished(self, text, names, per_surf, dists=None, posgrid=None, actual_times_per_group=None):
        self.btn_run.config(state="normal")
        self.btn_cancel.config(state="disabled")
        self.study_bar["value"] = 100.0 if per_surf else 0.0
        self.set_status(text)
        self._log(text)
        if per_surf is not None:
            self.results = per_surf
            self.res_names = names
            self.dists = dists
            self.posgrid = posgrid
            if actual_times_per_group is not None:
                # Store actual times for fast time sweep plotting
                self.meta["actual_times_per_group"] = actual_times_per_group
            self._draw_results()

    def export_csv(self):
        if self.results is None or self.meta is None:
            messagebox.showerror("Export CSV", "nothing to export (run a study first)")
            return
        from tkinter import filedialog as _fd
        path = _fd.asksaveasfilename(defaultextension=".csv", initialdir=os.path.join(ROOT, "opencode"),
                                     initialfile="study.csv", filetypes=[("CSV", "*.csv")])
        if not path:
            return
        try:
            m = self.meta
            rows = []
            for si, name in enumerate(self.res_names):
                for i1 in range(len(m["x_vals"])):
                    xv = csv_val(m["x_vals"][i1], m["x_ticks"][i1])
                    if not m["curve_labels"]:
                        cl = self.results[si][i1][0]
                        if _cl_missing(cl):
                            continue
                        rows.append([name, m["p1_label"], xv, "none", "", cl])
                    else:
                        for j, lab in enumerate(m["curve_labels"]):
                            pv = m["p2_vals"][j] if j < len(m["p2_vals"]) else ""
                            cl = self.results[si][i1][j]
                            if _cl_missing(cl):
                                continue
                            rows.append([name, m["p1_label"], xv, m["p2_label"],
                                         csv_val(pv, lab), cl])
            write_csv(path, ["surface", "P1", "P1val", "P2", "P2val", "CL"], rows)
            if self.exp_dist.get():
                self._export_dist(path)
            self._log(f"exported {path}")
        except Exception as ex:
            messagebox.showerror("Export CSV", str(ex))

    def _export_dist(self, base_path):
        m = self.meta
        rho = medium_density(m.get("medium", "water"))
        funit = m.get("funit", "N")
        rows = []
        for si, name in enumerate(self.res_names):
            for i1 in range(len(m["x_vals"])):
                xv = csv_val(m["x_vals"][i1], m["x_ticks"][i1])
                n2 = len(m["curve_labels"]) if m["curve_labels"] else 1
                for j in range(n2):
                    cl = self.results[si][i1][j]
                    if m["curve_labels"]:
                        pv = m["p2_vals"][j] if j < len(m["p2_vals"]) else ""
                        lab = csv_val(pv, m["curve_labels"][j])
                    else:
                        lab = ""
                    d = self.dists[si][i1][j] if self.dists else None
                    if d is None or _cl_missing(cl):
                        continue  # no distributions captured (failed point or non max-t cell)
                    s_list = d.get("s") or []
                    g_list = d.get("gamma") or []
                    c_list = d.get("cl") or []
                    lo = d.get("loads") or [None, None, None]
                    F = ([v * rho * force_factor(funit) for v in lo]
                         if all(v is not None for v in lo) else [None, None, None])
                    n = max(len(s_list), len(g_list), len(c_list), 1)
                    for k in range(n):
                        rows.append([name, m["p1_label"], xv, lab, cl,
                                     s_list[k] if k < len(s_list) else None,
                                     g_list[k] if k < len(g_list) else None,
                                     c_list[k] if k < len(c_list) else None,
                                     F[0], F[1], F[2]])
        root, _ = os.path.splitext(base_path)
        dist_path = root + "_dist.csv"
        write_csv(dist_path, ["surface", "P1", "P1val", "P2val", "CL", "s",
                              "gamma", "Cl_2d", f"Fx[{funit}]", f"Fy[{funit}]", f"Fz[{funit}]"], rows)
        self._log(f"exported {dist_path}")

    def _draw_results(self):
        for w in self.res_inner.winfo_children():
            w.destroy()
        if self.results is None or self.meta is None:
            return
        m = self.meta
        p2t = m.get("p2_ticks", [])
        self.d_fix2_cb.config(values=p2t if p2t else ["-"])
        if self.d_fix2.get() not in (p2t or ["-"]):
            self.d_fix2.set((p2t or ["-"])[-1])
        self.d_fix1_cb.config(values=m["x_ticks"])
        if self.d_fix1.get() not in m["x_ticks"]:
            self.d_fix1.set(m["x_ticks"][-1])
        
        # For fast time sweeps, use actual times instead of requested times
        has_fast_times = "actual_times_per_group" in m and m["actual_times_per_group"] is not None
        if has_fast_times:
            actual_times = m["actual_times_per_group"]
            t_side = m.get("t_side", "p1")
        else:
            actual_times = None
            t_side = None
        
        x = np.array(m["x_vals"], dtype=float) if m["x_num"] else np.arange(len(m["x_vals"]))
        if m.get("surf_side") and getattr(self, "posgrid", None):
            idx = {name: si for si, name in enumerate(getattr(self, "res_names", []))}
            for _p, members in self._series(m):
                if not any(np.any(~np.isnan(np.array(self.results[idx[n]], dtype=float)))
                           for n in members):
                    continue
                ttk.Label(self.res_inner, text=f"series {_p + 1}: {', '.join(members)}",
                          font=("TkDefaultFont", 10, "bold")).pack(anchor="w", padx=4, pady=(6, 0))
                self._draw_series_cl(members, _p, m, idx, actual_times, t_side)
                if self.d_show.get():
                    self._draw_series_dist(members, _p, m, "p1", idx)
                    if m["p2_vals"]:
                        self._draw_series_dist(members, _p, m, "p2", idx)
            return
        for si, name in enumerate(getattr(self, "res_names", [])):
            mat = np.array(self.results[si], dtype=float)
            if not np.any(~np.isnan(mat)):
                continue  # never in any run (e.g. unticked and no set sweep)
            ttk.Label(self.res_inner, text=name,
                      font=("TkDefaultFont", 10, "bold")).pack(anchor="w", padx=4, pady=(6, 0))
            row = ttk.Frame(self.res_inner)
            row.pack(fill="x", padx=4)
            fig = Figure(figsize=(7.0, 3.2), constrained_layout=True)
            ax = fig.add_subplot(111)
            cv = FigureCanvasTkAgg(fig, master=row)
            cv.get_tk_widget().pack(side="left", padx=4)
            self.app._bind_wheel(cv.get_tk_widget(), self.res_scroll)
            if m["curve_labels"]:
                for j, lab in enumerate(m["curve_labels"]):
                    y = mat[:, j]
                    ok = ~np.isnan(y)
                    if np.any(ok):
                        if has_fast_times and t_side == "p2":
                            # t is P2: x-axis should use actual times for this P1 value
                            # actual_times[group_idx] where group_idx = P1 value index
                            # For non-surface-set sweep with fast time, each column j is a P2 value
                            # The actual times per group (P1 value) are in actual_times[si]
                            pass  # x already correct for this surface
                        ax.plot(x[ok], y[ok], label=m["curve_tex"][j])
                ax.legend(fontsize=8)
            else:
                y = mat[:, 0]
                ok = ~np.isnan(y)
                if np.any(ok):
                    ax.plot(x[ok], y[ok])
            ax.set_xlabel(m.get("p1_tex", m["p1_label"]))
            ax.set_ylabel("$C_L$")
            if not m["x_num"]:
                ax.set_xticks(x)
                ax.set_xticklabels(m["x_ticks"])
            ax.grid(True)
            cv.draw()
            if self.d_show.get():
                self._draw_dist_row(si, name, m, "p1")
                if m["p2_vals"]:
                    self._draw_dist_row(si, name, m, "p2")

    def _series(self, m):
        """Position series for surface sweeps: [(pos, [names])]. Surfaces are
        grouped by their index within each run's composition (1st surfaces
        together, 2nd together, ...); names follow res_names order."""
        n1 = len(m["x_vals"])
        n2 = len(m["p2_vals"]) if m["p2_vals"] else 1
        series = {}
        for si, name in enumerate(getattr(self, "res_names", [])):
            for i1 in range(n1):
                for i2 in range(n2):
                    p = None
                    try:
                        if self.posgrid:
                            p = self.posgrid[si][i1][i2]
                    except (IndexError, TypeError):
                        p = None
                    if p is not None and name not in series.setdefault(p, []):
                        series[p].append(name)
        return [(p, series[p]) for p in sorted(series)]

    def _max_curves(self):
        txt = self.e_maxc.get().strip()
        if not txt:
            return None
        try:
            return max(2, int(txt))
        except ValueError:
            return None

    def _plot_dist_fig(self, curves, ylabel):
        """Draw one stacked-distribution figure (plain lines, y min 0,
        max-curves thinning, no title). Returns True if drawn."""
        xmax = self._max_curves()
        if xmax is not None:
            curves = [curves[i] for i in thin_indices(len(curves), xmax)]
        if not curves:
            return False
        row = ttk.Frame(self.res_inner)
        row.pack(fill="x", padx=4)
        fig = Figure(figsize=(7.0, 3.2), constrained_layout=True)
        ax = fig.add_subplot(111)
        cv = FigureCanvasTkAgg(fig, master=row)
        cv.get_tk_widget().pack(side="left", padx=4)
        self.app._bind_wheel(cv.get_tk_widget(), self.res_scroll)
        for x, y, lab in curves:
            ax.plot(x, y, label=lab)
        ax.set_xlabel("$s/b$")
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8)
        ax.grid(True)
        cv.draw()
        return True

    def _dist_xy(self, d, kind, m):
        """(x, y) or (None, None) for one dist cell (spanwise s vs distribution).
        loads (l) and gamma are normalized by their max (lib stores ld already
        as l/l_max, gamma as |sum| chordwise)."""
        if not d:
            return None, None
        if kind == "loads":
            y = d.get("ld")
            if not y:
                return None, None
        elif kind == "gamma":
            y = d["gamma"]
            if not y:
                return None, None
            mx = max(y)
            if not mx > 0:
                return None, None
            y = [v / mx for v in y]
        else:
            y = d["cl"]
        x = d["s"]
        if not y or not x or len(y) != len(x):
            return None, None
        return x, y

    def _dist_ylabel(self, kind, m):
        if kind == "loads":
            return r"$l/l_{\text{\max}}$"
        return "$C_l$" if kind == "Cl_2d" else r"$\Gamma/\Gamma_\text{\max}$"

    def _pos(self, si, i1, i2):
        try:
            pg = self.posgrid
            if pg:
                return pg[si][i1][i2]
        except (IndexError, TypeError):
            pass
        return None

    def _draw_series_cl(self, members, pos, m, idx, actual_times=None, t_side=None):
        """CL figure for one position series (surface sweep). tside p2:
        x = P1, one curve per member surface. tside p1: x = sets,
        one curve per P2 value. No titles (row header names the series).
        If actual_times provided (fast time sweep), use those for x-axis."""
        tside = m.get("surf_side")
        n1 = len(m["x_vals"])
        n2 = len(m["p2_vals"]) if m["p2_vals"] else 1
        row = ttk.Frame(self.res_inner)
        row.pack(fill="x", padx=4)
        fig = Figure(figsize=(7.0, 3.2), constrained_layout=True)
        ax = fig.add_subplot(111)
        cv = FigureCanvasTkAgg(fig, master=row)
        cv.get_tk_widget().pack(side="left", padx=4)
        self.app._bind_wheel(cv.get_tk_widget(), self.res_scroll)
        
        # Determine x-axis values for fast time sweep
        if actual_times is not None and t_side is not None:
            # actual_times[group_idx] = list of actual times for that group
            # group_idx corresponds to the non-t, non-surface-set parameter
            # For surface_set sweep with fast time:
            # - If t_side == "p1": t is P1, groups are P2 values (surface sets)
            # - If t_side == "p2": t is P2, groups are P1 values (surface sets)
            pass  # handled below per branch
        
        if tside == "p2":
            if actual_times is not None and t_side == "p1":
                # t is P1, surface_set is P2
                # For this position series (fixed P2 value), use actual times for that group
                # group_idx corresponds to which P2 value (surface set) this position is
                # We need to find which group this position belongs to
                # actual_times[p2_idx] gives times for that P2 value
                # Find the P2 index for this position
                p2_idx = None
                for j in range(n2):
                    if any(self._pos(idx[name], i, j) == pos for i in range(n1) for name in members):
                        p2_idx = j
                        break
                if p2_idx is not None and p2_idx < len(actual_times):
                    x = np.array(actual_times[p2_idx], dtype=float)
                else:
                    x = np.array(m["x_vals"], dtype=float) if m["x_num"] else np.arange(n1)
            else:
                x = np.array(m["x_vals"], dtype=float) if m["x_num"] else np.arange(n1)
            for name in members:
                si = idx[name]
                cols = [j for j in range(n2)
                        if any(self._pos(si, i, j) == pos for i in range(n1))]
                if not cols:
                    continue
                y = np.array([self.results[si][i][cols[0]] for i in range(n1)], dtype=float)
                ok = ~np.isnan(y)
                if np.any(ok):
                    ax.plot(x[ok], y[ok], label=name)
            ax.legend(fontsize=8)
            ax.set_xlabel(m.get("p1_tex", m["p1_label"]))
            if not m["x_num"]:
                ax.set_xticks(x)
                ax.set_xticklabels([f"{v:.3g}" for v in x])
        else:
            if actual_times is not None and t_side == "p2":
                # t is P2, surface_set is P1
                # For this position series (fixed P1 value), use actual times for that group
                p1_idx = None
                for i in range(n1):
                    if any(self._pos(idx[name], i, j) == pos for j in range(n2) for name in members):
                        p1_idx = i
                        break
                if p1_idx is not None and p1_idx < len(actual_times):
                    x = np.array(actual_times[p1_idx], dtype=float)
                else:
                    x = np.arange(n1)
            else:
                x = np.arange(n1)
            labs = m["curve_tex"] if m["curve_labels"] else [None] * n2
            for j, lab in enumerate(labs):
                ys = []
                for i in range(n1):
                    v = float("nan")
                    for name in members:
                        si = idx[name]
                        if self._pos(si, i, j) == pos:
                            v = self.results[si][i][j]
                            break
                    ys.append(v)
                y = np.array(ys, dtype=float)
                ok = ~np.isnan(y)
                if np.any(ok):
                    ax.plot(x[ok], y[ok], label=lab)
            if m["curve_labels"]:
                ax.legend(fontsize=8)
            ax.set_xlabel(m.get("p1_tex", m["p1_label"]))
            ax.set_xticks(x)
            ax.set_xticklabels([f"{v:.3g}" for v in x] if actual_times is not None else m["x_ticks"])
        ax.set_ylabel("$C_L$")
        ax.grid(True)
        cv.draw()

    def _draw_series_dist(self, members, pos, m, moving, idx):
        """Stacked dist figure for one position series (surface sweep).
        Curves stacked over sets resolve to the series' surface in each set
        (legend = surface names); over normal params, tick labels.
        Plain lines, y min at 0, max-curves thinning, no titles."""
        kind = self.d_dist.get()
        n1 = len(m["x_vals"])
        n2 = len(m["p2_vals"]) if m["p2_vals"] else 1
        items = []  # (si, i1, i2, lab)
        if moving == "p1":
            if m.get("surf_side") == "p1":
                p2t = m.get("p2_ticks", [])
                jf = p2t.index(self.d_fix2.get()) if p2t and self.d_fix2.get() in p2t else 0
                seen = set()
                for i1 in range(n1):
                    s = next((name for name in members
                              if self._pos(idx[name], i1, jf) == pos), None)
                    if s is None:
                        continue
                    lab = s if s not in seen else f"{s}@{m['x_ticks'][i1]}"
                    seen.add(lab)
                    items.append((idx[s], i1, jf, lab))
            else:
                p2t = m.get("p2_ticks", [])
                jf = p2t.index(self.d_fix2.get()) if p2t and self.d_fix2.get() in p2t else 0
                s = next((name for name in members
                          if any(self._pos(idx[name], i, jf) == pos for i in range(n1))), None)
                if s is None:
                    return
                short = m.get("p1_short")
                pre = f"{tex(short)}=" if short else ""
                ticks = m["x_ticks"]
                for i1 in range(n1):
                    items.append((idx[s], i1, jf, f"{pre}{ticks[i1]}"))
        else:
            p2t = m.get("p2_ticks", [])
            if not p2t:
                return
            if m.get("surf_side") == "p2":
                xt = m["x_ticks"]
                jf = list(xt).index(self.d_fix1.get()) if self.d_fix1.get() in list(xt) else len(xt) - 1
                seen = set()
                for i2 in range(len(p2t)):
                    s = next((name for name in members
                              if self._pos(idx[name], jf, i2) == pos), None)
                    if s is None:
                        continue
                    lab = s if s not in seen else f"{s}@{p2t[i2]}"
                    seen.add(lab)
                    items.append((idx[s], jf, i2, lab))
            else:
                xt = m["x_ticks"]
                jf = list(xt).index(self.d_fix1.get()) if self.d_fix1.get() in list(xt) else len(xt) - 1
                s = next((name for name in members
                          if any(self._pos(idx[name], jf, j) == pos for j in range(len(p2t)))), None)
                if s is None:
                    return
                short = m.get("p2_short")
                pre = f"{tex(short)}=" if short else ""
                for i2 in range(len(p2t)):
                    items.append((idx[s], jf, i2, f"{pre}{p2t[i2]}"))
        curves = []
        for si, i1, i2, lab in items:
            xy = self._dist_xy(self.dists[si][i1][i2] if self.dists else None, kind, m)
            if xy[0] is None:
                continue
            curves.append((xy[0], xy[1], lab))
        self._plot_dist_fig(curves, self._dist_ylabel(kind, m))

    def _draw_dist_row(self, si, name, m, moving):
        """Stacked distribution curves vs spanwise s: the moving param varies,
        the other is fixed at the selected value. Plain lines, y min at 0,
        at most max-curves curves (first+last kept). No titles."""
        kind = self.d_dist.get()
        short = m.get("p1_short") if moving == "p1" else m.get("p2_short")
        pre = f"{tex(short)}=" if short else ""
        if moving == "p1":
            n_move = len(m["x_vals"])
            ticks = m["x_ticks"]
            p2t = m.get("p2_ticks", [])
            if p2t:
                jf = p2t.index(self.d_fix2.get()) if self.d_fix2.get() in p2t else len(p2t) - 1
            else:
                jf = 0
            cells = [(i1, jf, f"{pre}{ticks[i1]}") for i1 in range(n_move)]
        else:
            p2t = m.get("p2_ticks", [])
            if not p2t:
                return
            n_move = len(p2t)
            ticks = p2t
            xt = m["x_ticks"]
            jf = list(xt).index(self.d_fix1.get()) if self.d_fix1.get() in list(xt) else len(xt) - 1
            cells = [(jf, i2, f"{pre}{ticks[i2]}") for i2 in range(n_move)]
        curves = []
        for i1, i2, lab in cells:
            xy = self._dist_xy(self.dists[si][i1][i2] if self.dists else None, kind, m)
            if xy[0] is None:
                continue
            curves.append((xy[0], xy[1], lab))
        self._plot_dist_fig(curves, self._dist_ylabel(kind, m))


class TrimTab(ttk.Frame):
    """Operating-point trim: batch of trim_foils calls over one parameter.
    Single (alpha, beta) pair per point, applied to every run surface."""
    TRIM_EXCL = {"solver: surface set", "surface: alpha [deg]", "surface: beta [deg]"}

    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self.config = []
        self.tres = None   # dict lists per P1 value
        self._stop = False
        self._build()

    def _combo(self, parent, row, label, values, default=""):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=1)
        var = tk.StringVar(value=default)
        cb = ttk.Combobox(parent, textvariable=var, values=values, width=22, state="readonly")
        cb.grid(row=row, column=1, sticky="ew", padx=4, pady=1)
        return var, cb

    def _build(self):
        left = ttk.Frame(self, width=380)
        left.pack(side="left", fill="y", padx=6, pady=6)
        left.columnconfigure(1, weight=1)
        app = self.app
        r = 0
        ttk.Label(left, text="Base surfaces (ticked = in solver):").grid(row=r, column=0, columnspan=2, sticky="w", padx=4, pady=2); r += 1
        self.cfg_body = ttk.Frame(left)
        self.cfg_body.grid(row=r, column=0, columnspan=2, sticky="ew", padx=4, pady=2); r += 1
        brow = ttk.Frame(left)
        brow.grid(row=r, column=0, columnspan=2, pady=2); r += 1
        ttk.Button(brow, text="Add saved…", command=self.add_saved).pack(side="left", padx=2)
        ttk.Button(brow, text="Snapshot tabs", command=self.snapshot_tabs).pack(side="left", padx=2)
        ttk.Button(brow, text="Clear", command=self.clear_cfg).pack(side="left", padx=2)
        ttk.Label(left, text="Targets").grid(row=r, column=0, columnspan=2, sticky="w", padx=4, pady=(6, 1)); r += 1
        self.e_fz = app._entry(left, r, "Fz target (up +)", "160"); r += 1
        self.e_fy = app._entry(left, r, "Fy target", "0"); r += 1
        ttk.Label(left, text="force unit").grid(row=r, column=0, sticky="w", padx=4, pady=1)
        self.c_unit = tk.StringVar(value="kgf")
        ttk.Combobox(left, textvariable=self.c_unit, values=["N", "kgf"], width=10, state="readonly").grid(row=r, column=1, padx=4, pady=1); r += 1
        ttk.Label(left, text="medium").grid(row=r, column=0, sticky="w", padx=4, pady=1)
        self.c_med = tk.StringVar(value="water")
        ttk.Combobox(left, textvariable=self.c_med, values=["water", "air"], width=10, state="readonly").grid(row=r, column=1, padx=4, pady=1); r += 1
        ttk.Label(left, text="Initial guess / tolerances").grid(row=r, column=0, columnspan=2, sticky="w", padx=4, pady=(6, 1)); r += 1
        self.e_a0 = app._entry(left, r, "alpha0 [deg]", "6.8"); r += 1
        self.e_b0 = app._entry(left, r, "beta0 [deg]", "1.22"); r += 1
        self.e_tol = app._entry(left, r, "tol [kgf]", "0.5"); r += 1
        self.e_maxit = app._entry(left, r, "maxit", "15"); r += 1
        fields = ["(none)"] + [l for l in sorted(STUDY_FIELDS) if l not in self.TRIM_EXCL]
        ttk.Label(left, text="Param 1 (batch)").grid(row=r, column=0, columnspan=2, sticky="w", padx=4, pady=(6, 1)); r += 1
        self.p1_surf, self.p1_surf_cb = self._combo(left, r, "surface", ["ALL"], "ALL"); r += 1
        self.p1_field, _ = self._combo(left, r, "field", fields, "surface: fly height [m]"); r += 1
        self.p1_vals = app._entry(left, r, "values (a; b / a:b:c)", "", width=24); r += 1
        self.p1_info = ttk.Label(left, text="")
        self.p1_info.grid(row=r, column=0, columnspan=2, sticky="w", padx=4, pady=0); r += 1
        self.p1_field.trace_add("write", lambda *a: (self._on_p1_field(), self._refresh_counts()))
        self.p1_vals.bind("<FocusOut>", lambda e: self._refresh_counts())
        self._on_p1_field()
        ttk.Button(left, text="Solver parameters", command=self.open_trim_solver).grid(row=r, column=0, columnspan=2, pady=4); r += 1
        ttk.Label(left, text="t/dt/ratios/U from the trim solver parameters (VLMSolver defaults); P1 overrides apply per point.",
                  wraplength=340).grid(row=r, column=0, columnspan=2, sticky="w", padx=4, pady=4); r += 1
        self.runs_label = ttk.Label(left, text="runs: -")
        self.runs_label.grid(row=r, column=0, columnspan=2, sticky="w", padx=4, pady=1); r += 1
        bro = ttk.Frame(left)
        bro.grid(row=r, column=0, columnspan=2, pady=4); r += 1
        self.btn_run = ttk.Button(bro, text="Run trim", command=self.run_trim)
        self.btn_run.pack(side="left", padx=3)
        self.btn_cancel = ttk.Button(bro, text="Cancel", command=self.cancel_trim, state="disabled")
        self.btn_cancel.pack(side="left", padx=3)
        ttk.Button(bro, text="Export CSV", command=self.export_csv).pack(side="left", padx=3)
        srow = ttk.Frame(left)
        srow.grid(row=r, column=0, columnspan=2, pady=2); r += 1
        ttk.Label(srow, text="P1 point:").pack(side="left", padx=2)
        self.c_point = tk.StringVar(value="(all points)")
        self.c_point_cb = ttk.Combobox(srow, textvariable=self.c_point, values=["(all points)"],
                                       width=12, state="readonly")
        self.c_point_cb.pack(side="left", padx=2)
        ttk.Label(srow, text="base:").pack(side="left", padx=2)
        self.e_base = ttk.Entry(srow, width=12)
        self.e_base.insert(0, "trimmed")
        self.e_base.pack(side="left", padx=2)
        ttk.Button(srow, text="Save surfaces", command=self.save_trimmed).pack(side="left", padx=2)
        self.trim_bar = ttk.Progressbar(left, maximum=100)
        self.trim_bar.grid(row=r, column=0, columnspan=2, sticky="ew", padx=4, pady=2); r += 1
        self.status = ttk.Label(left, text="idle", wraplength=340)
        self.status.grid(row=r, column=0, columnspan=2, sticky="w", padx=4, pady=2); r += 1
        self.log = tk.Text(left, height=8, width=46)
        self.log.grid(row=r, column=0, columnspan=2, padx=4, pady=4); r += 1

        right = ttk.Frame(self)
        right.pack(side="right", fill="both", expand=True, padx=6, pady=6)
        ttk.Label(right, text="Trimmed alpha/beta and achieved loads vs param 1").pack(side="top", anchor="w", padx=4)
        self.scroll_wrap = ttk.Frame(right)
        self.scroll_wrap.pack(side="top", fill="both", expand=True)
        self.res_scroll = tk.Canvas(self.scroll_wrap, highlightthickness=0)
        vscroll = ttk.Scrollbar(self.scroll_wrap, orient="vertical", command=self.res_scroll.yview)
        hscroll = ttk.Scrollbar(self.scroll_wrap, orient="horizontal", command=self.res_scroll.xview)
        self.res_inner = ttk.Frame(self.res_scroll)
        self.res_inner.bind("<Configure>", lambda e: self.res_scroll.configure(scrollregion=self.res_scroll.bbox("all")))
        self.res_scroll.create_window((0, 0), window=self.res_inner, anchor="nw")
        self.res_scroll.configure(yscrollcommand=vscroll.set, xscrollcommand=hscroll.set)
        vscroll.pack(side="right", fill="y")
        hscroll.pack(side="bottom", fill="x")
        self.res_scroll.pack(side="left", fill="both", expand=True)
        self.app._bind_wheel(self.res_scroll, self.res_scroll)
        self.app._bind_wheel(self.res_inner, self.res_scroll)
        bot = ttk.Frame(right)
        bot.pack(side="bottom", fill="x")
        self.summary = tk.Text(bot, height=8)
        self.summary.pack(fill="x", padx=4, pady=4)
        self.summary.insert("end", "No trim results yet.")
        self.summary.config(state="disabled")
        self._build_trim_solver_window()

    # -- base config (same preset pattern as Study) --
    def _active(self):
        return [c for c in self.config if c.get("use", True)]

    def _refresh_cfg(self):
        for w in self.cfg_body.winfo_children():
            w.destroy()
        for i, st in enumerate(self.config):
            row = ttk.Frame(self.cfg_body)
            row.pack(fill="x")
            var = tk.BooleanVar(value=bool(st.get("use", True)))
            ttk.Checkbutton(row, variable=var,
                            command=lambda i=i, var=var: self._toggle_use(i, var)).pack(side="left")
            ttk.Label(row, text=st["name"]).pack(side="left", padx=4)
            ttk.Button(row, text="x", width=2, command=lambda i=i: self.remove_at(i)).pack(side="right")
        names = [c["name"] for c in self._active()]
        cur = self.p1_surf.get()
        self.p1_surf_cb.config(values=["ALL"] + names)
        if cur not in (["ALL"] + names):
            self.p1_surf.set("ALL")
        self._refresh_counts()

    def _toggle_use(self, i, var):
        self.config[i]["use"] = bool(var.get())
        self._refresh_cfg()

    def _unique_name(self, base):
        names = {c["name"] for c in self.config}
        if base not in names:
            return base
        i = 2
        while f"{base} ({i})" in names:
            i += 1
        return f"{base} ({i})"

    def add_saved(self):
        from tkinter import filedialog as _fd
        path = _fd.askopenfilename(initialdir=self.app._surf_dir(), filetypes=[("JSON", "*.json")])
        if not path:
            return
        try:
            state = load_preset(path)
            state["name"] = self._unique_name(state["name"])
            state["use"] = True
            self.config.append(state)
            self._refresh_cfg()
        except Exception as ex:
            messagebox.showerror("Add saved", str(ex))

    def snapshot_tabs(self):
        if not self.app.surf_tabs:
            messagebox.showerror("Snapshot", "no surface tabs")
            return
        for tab in self.app.surf_tabs:
            st = tab.form_state()
            st["name"] = self._unique_name(tab.name)
            st["use"] = True
            self.config.append(st)
        self._refresh_cfg()

    def remove_at(self, i):
        del self.config[i]
        self._refresh_cfg()

    def clear_cfg(self):
        self.config = []
        self._refresh_cfg()

    def set_status(self, txt):
        self.status.config(text=txt)

    def _log(self, txt):
        self.log.insert("end", txt + "\n")
        self.log.see("end")

    def cancel_trim(self):
        self._stop = True
        self.set_status("cancelling…")

    # -- sweep parsing --
    def _parse_p1(self):
        label = self.p1_field.get()
        if label == "(none)":
            return None
        scope, key, kind, short, unit = STUDY_FIELDS[label]
        surf, all_flag = None, False
        if scope == "surface":
            if self.p1_surf.get() == "ALL":
                all_flag = True
            else:
                names = [c["name"] for c in self._active()]
                if self.p1_surf.get() not in names:
                    raise ValueError("param 1: pick a base surface")
                surf = self.p1_surf.get()
        values = parse_sweep_values(self.p1_vals.get(), kind)
        if unit == "spd":
            unit = self.ts_vars["speed_unit"].get()
        pre = f"{short} ({surf})" if surf else (f"{short} (all)" if all_flag else short)
        lab = f"{pre} [{unit}]" if unit not in ("", "-") else pre
        return dict(scope=scope, key=key, kind=kind, short=short, unit=unit,
                    surf=surf, all=all_flag, values=values, label=lab,
                    label_tex=field_label(short, surf, all_flag, unit, use_tex=True))

    def _on_p1_field(self):
        is_none = self.p1_field.get() == "(none)"
        self.p1_surf_cb.config(state="disabled" if is_none else "readonly")
        self.p1_vals.config(state="disabled" if is_none else "normal")

    def _refresh_counts(self):
        try:
            p1 = self._parse_p1()
            if p1 is None:
                self.p1_info.config(text="[base]")
                self.runs_label.config(text="runs: 1")
                return
            self.p1_info.config(text=vals_text(p1["values"], p1["kind"]))
            self.runs_label.config(text=f"runs: {len(p1['values'])}")
        except Exception:
            self.runs_label.config(text="runs: ?")

    # -- run --
    def _build_trim_solver_window(self):
        """Hidden solver-parameters window for trim (VLMSolver defaults)."""
        self.ts_win = tk.Toplevel(self)
        self.ts_win.title("Trim solver parameters")
        self.ts_win.protocol("WM_DELETE_WINDOW", self.ts_win.withdraw)
        adv = ttk.Frame(self.ts_win, padding=10)
        adv.pack(fill="both", expand=True)
        adv.columnconfigure(1, weight=1)
        S, SV, _ = build_solver_param_fields(self.app, adv)
        self.ts_entries, self.ts_vars = S, SV
        self.ts_win.withdraw()

    def open_trim_solver(self):
        self.ts_win.deiconify()
        self.ts_win.lift()

    def trim_snapshot(self):
        """Trim solver form as plain strings (UI thread only)."""
        return snapshot_param_dict(self.ts_entries, self.ts_vars)

    def run_trim(self):
        try:
            active = self._active()
            if not active:
                raise ValueError("no surface in solver: tick at least one base surface")
            p1 = self._parse_p1()
            if p1 is None:
                p1 = dict(scope="none", key="none", kind="str", short="base", unit="",
                          surf=None, all=False, values=["base"], label="base")
            if (p1["scope"], p1["key"]) == ("surface", "alpha") or (p1["scope"], p1["key"]) == ("surface", "beta"):
                raise ValueError("param 1 cannot be alpha/beta (trim solves them)")
            solver_base = self.trim_snapshot()
            forced = forced_speed_mode([(p1["scope"], p1["key"])])
            if p1["scope"] == "solver" and not forced \
                    and solver_base["speed_mode"] == "Froude Fn_h" \
                    and p1["key"] in ("ux", "uy", "uz"):
                raise ValueError("sweeping Ux/Uy/Uz needs speed-U mode (trim solver parameters)")
            if (p1["scope"], p1["key"]) == ("surface", "fly_h"):
                self._log("note: fly_height sweep shifts origin z with fly_height (z+ downwards)")
            rho = medium_density("water")
            rho_med = medium_density(self.c_med.get())
            funit = self.c_unit.get()
            t_kgf = lambda v: v / 9.81 if funit == "N" else v
            scale = 1025.0 / rho_med
            npts = len(p1["values"])
            fz_in, fy_in = broadcast_targets(self.e_fz.get(), self.e_fy.get(), npts)
            Xz = [t_kgf(v) * scale for v in fz_in]
            Xy = [t_kgf(v) * scale for v in fy_in]
            a0 = eval_expr(self.e_a0.get())
            b0 = eval_expr(self.e_b0.get())
            tol = eval_expr(self.e_tol.get())
            maxit = int(eval_expr(self.e_maxit.get()))
            if rho_med != rho:
                self._log(f"note: air targets rescaled x{scale:.1f} for the water-based trim")
            x_num = p1["kind"] in ("float", "int")
            to_disp = lambda X: X * 9.81 / 1025.0 * rho_med / (9.81 if funit == "kgf" else 1.0)
            meta = dict(p1_label=p1["label"], x_num=x_num,
                        p1_tex=p1["label_tex"],
                        p1=dict(scope=p1["scope"], key=p1["key"], surf=p1["surf"], all=p1["all"]),
                        x_vals=list(p1["values"]),
                        x_ticks=[sweep_text(p1["kind"], v) for v in p1["values"]],
                        fz_t=[to_disp(X) for X in Xz],
                        fy_t=[to_disp(X) for X in Xy],
                        funit=funit)
            payload = dict(config=[dict(name=c["name"], entries=dict(c["entries"]), vars=dict(c["vars"]))
                                    for c in active],
                           p1=dict(scope=p1["scope"], key=p1["key"], surf=p1["surf"],
                                   all=p1["all"], values=list(p1["values"])),
                           solver_base=solver_base, forced=forced, meta=meta,
                           Xz=Xz, Xy=Xy, a0=a0, b0=b0, tol=tol, maxit=maxit,
                           rho_med=rho_med, funit=funit)
        except Exception as ex:
            messagebox.showerror("Trim", str(ex))
            return
        self.tres = None
        self.meta = meta
        self._stop = False
        self.btn_run.config(state="disabled")
        self.btn_cancel.config(state="normal")
        self.trim_bar["value"] = 0
        self.set_status(f"trimming {len(p1['values'])} points… (each point blocks: no cancel inside)")
        threading.Thread(target=self._trim_worker, args=(payload,), daemon=True).start()

    def _trim_worker(self, pay):
        app = self.app
        vals = pay["p1"]["values"]
        n = len(vals)
        alpha = [float("nan")] * n
        beta = [float("nan")] * n
        fz = [float("nan")] * n
        fy = [float("nan")] * n
        a0, b0 = pay["a0"], pay["b0"]
        try:
            for i1, v1 in enumerate(vals):
                if self._stop:
                    raise _StudyCancel()
                try:
                    sd = dict(pay["solver_base"])
                    if pay["forced"]:
                        sd["speed_mode"] = pay["forced"]
                    p1 = pay["p1"]
                    if p1["scope"] == "solver":
                        sd[p1["key"]] = v1 if isinstance(v1, str) else f"{v1:.6g}"
                    is_fly = p1["scope"] == "surface" and p1["key"] == "fly_h"
                    h = float(v1) if is_fly else 0.0

                    def probe():
                        out = []
                        for st in pay["config"]:
                            ov = param_overrides(st, p1, v1)
                            ov["alpha"] = a0
                            ov["beta"] = b0
                            out.append(build_study_surface(st, ov))
                        return out

                    probe_surfs = probe()
                    probe_named = [(st["name"], s) for st, s in zip(pay["config"], probe_surfs)]
                    U, _info = flow_U_from(sd, probe_named)
                    kw, t, dt, dist = parse_solver_dict(sd)
                    bc = common_boundary(probe_named)

                    def build(a_deg, b_deg, h):
                        surfs = []
                        for st in pay["config"]:
                            ov = param_overrides(st, p1, v1)
                            ov["alpha"] = a_deg
                            ov["beta"] = b_deg
                            surfs.append(build_study_surface(st, ov))
                        return VLMSolver(surfs, U, bc, ratio=kw["ratio"],
                                         a_ratio=kw["a_ratio"], k_ratio=kw["k_ratio"])

                    aa, bb = trim_foils(build, [h], pay["Xz"][i1], Fy_targets=pay["Xy"][i1],
                                        alpha0=a0, beta0=b0, tol=pay["tol"], maxit=pay["maxit"],
                                        t=t, dt=dt, distribution=dist, verbose=False)
                    a0, b0 = float(aa[0]), float(bb[0])
                    # achieved loads: one extra solve at the solution
                    vlm = build(a0, b0, h)
                    vlm._time_sim(t=t, dt=dt, distribution=dist)
                    vlm._kuttas_loads()
                    F = pay["rho_med"] * np.sum([s.loads for s in vlm.surfaces], axis=0)
                    if pay["funit"] == "kgf":
                        F = F / 9.81
                    alpha[i1], beta[i1] = a0, b0
                    fz[i1], fy[i1] = float(-F[2]), float(F[1])
                    app.msg_q.put(("trim_log", True,
                                   (self, f"({i1 + 1}/{n}): alpha={a0:.3f} beta={b0:.3f} "
                                          f"Fz={fz[i1]:.1f} Fy={fy[i1]:.1f} {pay['funit']}")))
                except _StudyCancel:
                    raise
                except Exception as ex:
                    app.msg_q.put(("trim_log", True, (self, f"({i1 + 1}/{n}) FAILED: {ex}")))
                app.msg_q.put(("trim_progress", True, (self, i1 + 1, n)))
            app.msg_q.put(("trim_done", True, (self, "done",
                                              dict(alpha=alpha, beta=beta, fz=fz, fy=fy))))
        except _StudyCancel:
            app.msg_q.put(("trim_done", True, (self, "cancelled (partial)",
                                              dict(alpha=alpha, beta=beta, fz=fz, fy=fy))))
        except Exception as ex:
            app.msg_q.put(("trim_done", False, (self, f"ERROR: {ex}")))

    def _trim_finished(self, text, res):
        self.btn_run.config(state="normal")
        self.btn_cancel.config(state="disabled")
        self.trim_bar["value"] = 100.0 if res else 0.0
        self.set_status(text)
        self._log(text)
        if res is not None:
            self.tres = res
            self.c_point_cb.config(values=["(all points)"] + list(self.meta["x_ticks"]))
            self.c_point.set("(all points)")
            self._log("pick a P1 point (or all) + base name above, then Save surfaces")
            self._draw_trim()

    def _draw_trim(self):
        for w in self.res_inner.winfo_children():
            w.destroy()
        if self.tres is None or self.meta is None:
            return
        m = self.meta
        x = np.array(m["x_vals"], dtype=float) if m["x_num"] else np.arange(len(m["x_vals"]))
        row = ttk.Frame(self.res_inner)
        row.pack(fill="x", padx=4)
        fig = Figure(figsize=(7.0, 3.2), constrained_layout=True)
        ax = fig.add_subplot(111)
        cv = FigureCanvasTkAgg(fig, master=row)
        cv.get_tk_widget().pack(side="left", padx=4)
        self.app._bind_wheel(cv.get_tk_widget(), self.res_scroll)
        for key in ("alpha", "beta"):
            y = np.array(self.tres[key], dtype=float)
            ok = ~np.isnan(y)
            if np.any(ok):
                ax.plot(x[ok], y[ok], marker="o", ms=3, label=f"{tex(key)} [deg]")
        ax.set_xlabel(m.get("p1_tex", m["p1_label"]))
        ax.set_ylabel("deg")
        if not m["x_num"]:
            ax.set_xticks(x)
            ax.set_xticklabels(m["x_ticks"])
        ax.legend(fontsize=8)
        ax.grid(True)
        cv.draw()
        row2 = ttk.Frame(self.res_inner)
        row2.pack(fill="x", padx=4)
        fig2 = Figure(figsize=(7.0, 3.2), constrained_layout=True)
        ax2 = fig2.add_subplot(111)
        cv2 = FigureCanvasTkAgg(fig2, master=row2)
        cv2.get_tk_widget().pack(side="left", padx=4)
        self.app._bind_wheel(cv2.get_tk_widget(), self.res_scroll)
        for key, tk_, fsym in (("fz", "fz_t", "Fz"), ("fy", "fy_t", "Fy")):
            lab = f"{tex(fsym)} achieved [{m['funit']}]"
            y = np.array(self.tres[key], dtype=float)
            ok = ~np.isnan(y)
            if np.any(ok):
                ax2.plot(x[ok], y[ok], marker="o", ms=3, label=lab)
            yt = np.array(m[tk_], dtype=float)
            ax2.plot(x, yt, color="gray", ls="--" if key == "fz" else ":",
                     label=f"{tex(fsym)} target [{m['funit']}]")
        ax2.set_xlabel(m.get("p1_tex", m["p1_label"]))
        ax2.set_ylabel(m["funit"])
        if not m["x_num"]:
            ax2.set_xticks(x)
            ax2.set_xticklabels(m["x_ticks"])
        ax2.legend(fontsize=8)
        ax2.grid(True)
        cv2.draw()
        lines = ["P1 | alpha [deg] | beta [deg] | Fz | Fy | " + m["funit"]]
        for i, xv in enumerate(m["x_ticks"]):
            lines.append(f"{xv} | {self.tres['alpha'][i]:.3f} | {self.tres['beta'][i]:.3f} | "
                         f"{self.tres['fz'][i]:.2f} | {self.tres['fy'][i]:.2f}")
        self.summary.config(state="normal")
        self.summary.delete("1.0", "end")
        self.summary.insert("end", "\n".join(lines))
        self.summary.config(state="disabled")

    def save_trimmed(self):
        if self.tres is None or self.meta is None:
            messagebox.showerror("Save surfaces", "nothing to save (run a trim first)")
            return
        base = safe_name(self.e_base.get().strip() or "trimmed")
        tick = self.c_point.get()
        if tick == "(all points)":
            idxs = list(range(len(self.meta["x_vals"])))
        elif tick in self.meta["x_ticks"]:
            idxs = [self.meta["x_ticks"].index(tick)]
        else:
            messagebox.showerror("Save surfaces", "pick a P1 point")
            return
        try:
            saved = []
            multi = len(idxs) > 1
            p1m = self.meta.get("p1")
            for i1 in idxs:
                a, b = self.tres["alpha"][i1], self.tres["beta"][i1]
                if not (a == a and b == b):
                    self._log(f"skip {self.meta['x_ticks'][i1]}: no solution at this point")
                    continue
                v1 = self.meta["x_vals"][i1]
                for st in self.config:
                    if not st.get("use", True):
                        continue
                    ov = {}
                    if p1m is not None:
                        ov.update(param_overrides(st, p1m, v1))
                    ov["alpha"] = float(a)
                    ov["beta"] = float(b)
                    nst = apply_state_overrides(st, ov)
                    san = safe_name(st["name"])
                    tag = f"{safe_name(self.meta['x_ticks'][i1])}_" if multi else ""
                    fname = f"{base}_{tag}{san}.json"
                    nst["name"] = os.path.splitext(fname)[0]
                    with open(os.path.join(self.app._surf_dir(), fname), "w") as f:
                        json.dump(nst, f, indent=1)
                    saved.append(fname)
            if not saved:
                messagebox.showerror("Save surfaces", "nothing saved (no solution at these points)")
                return
            self._log("saved: " + ", ".join(saved))
        except Exception as ex:
            messagebox.showerror("Save surfaces", str(ex))

    def export_csv(self):
        if self.tres is None or self.meta is None:
            messagebox.showerror("Export CSV", "nothing to export (run a trim first)")
            return
        from tkinter import filedialog as _fd
        path = _fd.asksaveasfilename(defaultextension=".csv", initialdir=os.path.join(ROOT, "opencode"),
                                     initialfile="trim.csv", filetypes=[("CSV", "*.csv")])
        if not path:
            return
        try:
            m = self.meta
            rows = []
            for i, xv in enumerate(m["x_vals"]):
                rows.append([m["p1_label"], csv_val(xv, m["x_ticks"][i]),
                             self.tres["alpha"][i], self.tres["beta"][i],
                             self.tres["fz"][i], self.tres["fy"][i],
                             m["fz_t"][i], m["fy_t"][i]])
            write_csv(path, ["P1", "P1val", "alpha_deg", "beta_deg",
                             f"Fz[{m['funit']}]", f"Fy[{m['funit']}]",
                             f"Fz_target[{m['funit']}]", f"Fy_target[{m['funit']}]"], rows)
            self._log(f"exported {path}")
        except Exception as ex:
            messagebox.showerror("Export CSV", str(ex))


class VLMGui(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("VLM + ventilation")
        self.geometry("1280x780")
        self.solver = None
        self.msg_q = queue.Queue()
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True)

        # Tab surfaces
        self.tab_surf = ttk.Frame(self.nb)
        self.nb.add(self.tab_surf, text="Surfaces")
        toolbar = ttk.Frame(self.tab_surf)
        toolbar.pack(side="top", fill="x", padx=6, pady=4)
        ttk.Button(toolbar, text="+ New surface", command=self.add_surface).pack(side="left", padx=4)
        ttk.Button(toolbar, text="Delete active surface", command=self.del_surface).pack(side="left", padx=4)
        ttk.Button(toolbar, text="Save surface", command=self.save_surface).pack(side="left", padx=4)
        ttk.Button(toolbar, text="Load surface", command=self.load_surface).pack(side="left", padx=4)
        self.surf_nb = ttk.Notebook(self.tab_surf)
        self.surf_nb.pack(fill="both", expand=True)
        self.surf_tabs = []
        self.add_surface()

        # Tab solver
        self.tab_solv = ttk.Frame(self.nb)
        self.nb.add(self.tab_solv, text="VLMSolver")
        self._build_solver_tab()
        # Tab study
        self.tab_study = StudyTab(self.nb, self)
        self.nb.add(self.tab_study, text="Study")
        # Tab trim
        self.tab_trim = TrimTab(self.nb, self)
        self.nb.add(self.tab_trim, text="Trim")
        self.after(150, self._poll_queue)

    def _unique_tab_name(self, base="Surface"):
        names = {t.name for t in self.surf_tabs}
        i = 1
        while f"{base} {i}" in names:
            i += 1
        return f"{base} {i}"

    def add_surface(self):
        tab = SurfaceTab(self.surf_nb, self, self._unique_tab_name())
        self.surf_nb.add(tab, text=tab.name)
        self.surf_tabs.append(tab)
        self.surf_nb.select(tab)

    def del_surface(self):
        if len(self.surf_tabs) <= 1:
            messagebox.showinfo("Surfaces", "Keep at least one surface.")
            return
        cur = self.surf_nb.select()
        for tab in list(self.surf_tabs):
            if str(tab) == cur:
                self.surf_nb.forget(tab)
                self.surf_tabs.remove(tab)
                tab.destroy()
                break
        # tab names (possibly from loaded presets) are kept as-is
        self.refresh_solver_list()

    def built_surfaces(self):
        return [(tab.name, tab.built) for tab in self.surf_tabs if tab.built is not None]

    def _surf_dir(self):
        d = os.path.join(ROOT, "data")
        os.makedirs(d, exist_ok=True)
        return d

    def _active_tab(self):
        cur = self.surf_nb.select()
        for tab in self.surf_tabs:
            if str(tab) == cur:
                return tab
        return None

    def save_surface(self):
        tab = self._active_tab()
        if tab is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".json", initialdir=self._surf_dir(),
                                            initialfile=f"{tab.name}.json",
                                            filetypes=[("JSON", "*.json")])
        if not path:
            return
        # the preset name is the file name (the tab name is NOT the saved name)
        st = tab.form_state()
        st["name"] = os.path.splitext(os.path.basename(path))[0]
        with open(path, "w") as f:
            json.dump(st, f, indent=1)
        tab.name = st["name"]
        try:
            self.surf_nb.tab(tab, text=tab.name)
        except Exception:
            pass
        tab.status.config(text=f"saved to {path}")

    def load_surface(self):
        tab = self._active_tab()
        if tab is None:
            return
        path = filedialog.askopenfilename(initialdir=self._surf_dir(),
                                          filetypes=[("JSON", "*.json")])
        if not path:
            return
        try:
            state = load_preset(path)
            tab.set_form_state(state)
            self.refresh_solver_list()
        except Exception as ex:
            messagebox.showerror("Load surface", str(ex))

    # -- solver tab --
    def _entry(self, parent, row, label, default="", width=12):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=1)
        e = ttk.Entry(parent, width=width)
        e.insert(0, str(default))
        e.grid(row=row, column=1, sticky="ew", padx=4, pady=1)
        return e

    def _build_solver_tab(self):
        left = ttk.Frame(self.tab_solv, width=360)
        left.pack(side="left", fill="y", padx=6, pady=6)
        left.columnconfigure(1, weight=1)
        S = self.solv_entries = {}
        SV = self.solv_vars = {}
        r = 0
        ttk.Label(left, text="Built surfaces (selected = used):").grid(row=r, column=0, columnspan=2, sticky="w", padx=4, pady=2); r += 1
        self.surf_listbox = tk.Listbox(left, selectmode="multiple", height=5, exportselection=False)
        self.surf_listbox.grid(row=r, column=0, columnspan=2, sticky="ew", padx=4, pady=2); r += 1
        ttk.Label(left, text="flow speed U").grid(row=r, column=0, sticky="w", padx=4, pady=1)
        SV["speed_unit"] = tk.StringVar(value="m/s")
        ttk.Combobox(left, textvariable=SV["speed_unit"], values=["m/s", "kt"], width=6, state="readonly").grid(row=r, column=1, sticky="w", padx=4, pady=1); r += 1
        S["ux"] = self._entry(left, r, "U x", SOLVER_DEFAULTS["ux"]); r += 1
        S["uy"] = self._entry(left, r, "U y", SOLVER_DEFAULTS["uy"]); r += 1
        S["uz"] = self._entry(left, r, "U z", SOLVER_DEFAULTS["uz"]); r += 1
        ttk.Label(left, text="imposed").grid(row=r, column=0, sticky="w", padx=4, pady=1)
        SV["speed_mode"] = tk.StringVar(value="speed U")
        ttk.Combobox(left, textvariable=SV["speed_mode"], values=["speed U", "Froude Fn_h"], width=10, state="readonly").grid(row=r, column=1, padx=4, pady=1); r += 1
        S["fn"] = self._entry(left, r, "Fn_h (max immersion)", ""); r += 1
        self.speed_info = ttk.Label(left, text="U: -", wraplength=340)
        self.speed_info.grid(row=r, column=0, columnspan=2, sticky="w", padx=4, pady=1); r += 1
        SV["speed_mode"].trace_add("write", lambda *a: self._on_speed_mode_change())
        self._on_speed_mode_change()
        SV["speed_unit"].trace_add("write", lambda *a: self._on_speed_unit_change())
        self._speed_unit_current = "m/s"
        self.boundary_info = ttk.Label(left, text="boundary (from surfaces): -")
        self.boundary_info.grid(row=r, column=0, columnspan=2, sticky="w", padx=4, pady=1); r += 1
        S["t"] = self._entry(left, r, "t [s] (blank=auto)", ""); r += 1
        S["dt"] = self._entry(left, r, "dt [s] (blank=auto)", ""); r += 1
        ttk.Label(left, text="medium (scales loads)").grid(row=r, column=0, sticky="w", padx=4, pady=1)
        SV["medium"] = tk.StringVar(value="water")
        ttk.Combobox(left, textvariable=SV["medium"], values=["water", "air"], width=10, state="readonly").grid(row=r, column=1, padx=4, pady=1); r += 1
        ttk.Label(left, text="force unit").grid(row=r, column=0, sticky="w", padx=4, pady=1)
        SV["force_unit"] = tk.StringVar(value="N")
        ttk.Combobox(left, textvariable=SV["force_unit"], values=["N", "kgf"], width=10, state="readonly").grid(row=r, column=1, padx=4, pady=1); r += 1
        SV["mirror"] = tk.BooleanVar(value=False)
        ttk.Button(left, text="Advanced settings", command=self.open_advanced).grid(row=r, column=0, columnspan=2, pady=4); r += 1

        btns = ttk.Frame(left)
        btns.grid(row=r, column=0, columnspan=2, pady=6); r += 1
        self.btn_init = ttk.Button(btns, text="Init solver", command=self.on_init_solver)
        self.btn_init.pack(side="left", padx=3)
        self.btn_time = ttk.Button(btns, text="Run time_sim", command=lambda: self._run_async("time"))
        self.btn_time.pack(side="left", padx=3)
        self.btn_kutta = ttk.Button(btns, text="Run kutta loads", command=lambda: self._run_async("kutta"))
        self.btn_kutta.pack(side="left", padx=3)
        self.btn_vent = ttk.Button(btns, text="Run ventilation", command=lambda: self._run_async("vent"))
        self.btn_vent.pack(side="left", padx=3)

        erow = ttk.Frame(left)
        erow.grid(row=r, column=0, columnspan=2, pady=2); r += 1
        self.exp_dist_solv = tk.BooleanVar(value=False)
        ttk.Checkbutton(erow, text="distributions", variable=self.exp_dist_solv).pack(side="left", padx=2)
        ttk.Button(erow, text="Export CSV", command=self.export_solver_csv).pack(side="left", padx=2)

        self.solv_status = ttk.Label(left, text="solver not initialized", wraplength=340)
        self.solv_status.grid(row=r, column=0, columnspan=2, padx=4, pady=4); r += 1
        self.log = tk.Text(left, height=12, width=44)
        self.log.grid(row=r, column=0, columnspan=2, padx=4, pady=4)

        right = ttk.Frame(self.tab_solv)
        right.pack(side="right", fill="both", expand=True, padx=6, pady=6)
        self.fig3d = Figure(figsize=(14.0, 3), constrained_layout=False)
        # box maxed: legend lives outside right, data box uses the rest
        self.fig3d.subplots_adjust(left=0.0, right=0.80, top=1, bottom=0.00)
        self.ax3d = self.fig3d.add_subplot(111, projection="3d")
        self.cv3d = FigureCanvasTkAgg(self.fig3d, master=right)
        self.w3d = self.cv3d.get_tk_widget()
        self.w3d.pack(side="top", fill="x", padx=4)
        self._3d_shown = True
        ttk.Label(right, text="Per surface plots").pack(side="top", anchor="w", padx=4)
        self.scroll_wrap = ttk.Frame(right)
        self.scroll_wrap.pack(side="top", fill="both", expand=True)
        self.plot_scroll = tk.Canvas(self.scroll_wrap, highlightthickness=0)
        vscroll = ttk.Scrollbar(self.scroll_wrap, orient="vertical", command=self.plot_scroll.yview)
        hscroll = ttk.Scrollbar(self.scroll_wrap, orient="horizontal", command=self.plot_scroll.xview)
        self.plot_inner = ttk.Frame(self.plot_scroll)
        self.plot_inner.bind("<Configure>", lambda e: self.plot_scroll.configure(scrollregion=self.plot_scroll.bbox("all")))
        self.plot_scroll.create_window((0, 0), window=self.plot_inner, anchor="nw")
        self.plot_scroll.configure(yscrollcommand=vscroll.set, xscrollcommand=hscroll.set)
        vscroll.pack(side="right", fill="y")
        hscroll.pack(side="bottom", fill="x")
        self.plot_scroll.pack(side="left", fill="both", expand=True)
        self._bind_wheel(self.plot_scroll)
        self._bind_wheel(self.plot_inner)
        self.surf_plot_rows = []
        self._plot_sig = ([], [])
        bot = ttk.Frame(right)
        bot.pack(side="bottom", fill="x")
        ttk.Label(bot, text="Global CL / loads per surface + total").pack(anchor="w", padx=4)
        self.results_text = tk.Text(bot, height=8)
        self.results_text.pack(fill="x", padx=4, pady=4)
        self.results_text.insert("end", "Waiting for a run (time_sim + kutta loads).")
        self.results_text.config(state="disabled")
        self._build_advanced_window()

    def _build_advanced_window(self):
        """Hidden 'Advanced settings' window: plot toggles + viscous core ratios."""
        S, SV = self.solv_entries, self.solv_vars
        self.adv_win = tk.Toplevel(self)
        self.adv_win.title("Advanced settings")
        self.adv_win.protocol("WM_DELETE_WINDOW", self.adv_win.withdraw)
        adv = ttk.Frame(self.adv_win, padding=10)
        adv.pack(fill="both", expand=True)
        adv.columnconfigure(1, weight=1)
        r = 0
        ttk.Label(adv, text="viscous core ratios").grid(row=r, column=0, columnspan=2, sticky="w", padx=4, pady=2); r += 1
        S["ratio"] = self._entry(adv, r, "ratio (wake rollup)", SOLVER_DEFAULTS["ratio"]); r += 1
        S["a_ratio"] = self._entry(adv, r, "a_ratio (lin. system)", SOLVER_DEFAULTS["a_ratio"]); r += 1
        S["k_ratio"] = self._entry(adv, r, "k_ratio (kutta)", SOLVER_DEFAULTS["k_ratio"]); r += 1
        ttk.Label(adv, text="show plots").grid(row=r, column=0, sticky="w", padx=4, pady=(8, 1)); r += 1
        ttk.Checkbutton(adv, text="show mirror (3D)", variable=SV["mirror"],
                        command=self.refresh_results).grid(row=r, column=0, columnspan=2, sticky="w", padx=20, pady=0); r += 1
        for key, label, default in [("show_3d", "3D", True), ("show_cl", "Cl", True),
                                    ("show_gamma", "Gamma", False), ("show_vent", "Ventilation", True)]:
            SV[key] = tk.BooleanVar(value=default)
            ttk.Checkbutton(adv, text=label, variable=SV[key],
                            command=self.refresh_results).grid(row=r, column=0, columnspan=2, sticky="w", padx=20, pady=0); r += 1
        ttk.Label(adv, text="ventilation curves").grid(row=r, column=0, sticky="w", padx=4, pady=(8, 1)); r += 1
        for key, label, default in [("vent_lc", "Lc/c", True), ("vent_alpha", "alphaeff/alpha", True),
                                    ("vent_sigma", "sigma_c", True), ("vent_cl", "Cl_2d", False),
                                    ("vent_a0", "a0/2pi", False)]:
            SV[key] = tk.BooleanVar(value=default)
            ttk.Checkbutton(adv, text=label, variable=SV[key],
                            command=self.refresh_results).grid(row=r, column=0, columnspan=2, sticky="w", padx=20, pady=0); r += 1
        ttk.Label(adv, text="time distribution").grid(row=r, column=0, sticky="w", padx=4, pady=(8, 1)); r += 1
        SV["dist"] = tk.StringVar(value=SOLVER_DEFAULTS["distribution"])
        ttk.Combobox(adv, textvariable=SV["dist"], values=["classic", "cosine"], width=10, state="readonly").grid(row=r, column=1, padx=4, pady=1); r += 1
        self.adv_win.withdraw()

    def open_advanced(self):
        self.adv_win.deiconify()
        self.adv_win.lift()

    def _rebuild_surface_plots(self, named, kinds):
        for w in self.plot_inner.winfo_children():
            w.destroy()
        self.surf_plot_rows = []
        for name, _ in named:
            header = ttk.Label(self.plot_inner, text=name, font=("TkDefaultFont", 10, "bold"))
            header.pack(anchor="w", padx=4, pady=(6, 0))
            row = ttk.Frame(self.plot_inner)
            row.pack(fill="x", padx=4)
            plots = []
            for kind in kinds:
                fig = Figure(figsize=(3.2, 2.5), constrained_layout=True)
                ax = fig.add_subplot(111)
                cv = FigureCanvasTkAgg(fig, master=row)
                cv.get_tk_widget().pack(side="left", padx=4)
                self._bind_wheel(cv.get_tk_widget())
                plots.append((kind, ax, cv))
            self.surf_plot_rows.append({"name": name, "plots": plots})
        self._plot_sig = ([n for n, _ in named], list(kinds))

    def _bind_wheel(self, widget, canvas=None):
        target = canvas if canvas is not None else self.plot_scroll

        def _scroll(e):
            target.yview_scroll(int(-e.delta / 120), "units")
            return "break"

        widget.bind("<MouseWheel>", _scroll)
        widget.bind("<Button-4>", lambda e: target.yview_scroll(-1, "units"))
        widget.bind("<Button-5>", lambda e: target.yview_scroll(1, "units"))

    def _visible_kinds(self):
        SV = self.solv_vars
        return [k for k, flag in (("cl", "show_cl"), ("gamma", "show_gamma"), ("vent", "show_vent"))
                if SV[flag].get()]

    def _vent_curves(self):
        SV = self.solv_vars
        return {k: SV[f"vent_{k}"].get() for k in ("lc", "cl", "alpha", "a0", "sigma")}

    def refresh_solver_list(self):
        self.surf_listbox.delete(0, "end")
        for name, _ in self.built_surfaces():
            self.surf_listbox.insert("end", name)
            self.surf_listbox.select_set("end")

    def _selected_named(self):
        built = self.built_surfaces()
        sel = self.surf_listbox.curselection()
        if not built:
            raise ValueError("No built surface (Surfaces tab -> Build).")
        idx = sel if sel else tuple(range(len(built)))
        return [built[i] for i in idx]

    def _selected_surfaces(self):
        return [s for _, s in self._selected_named()]

    def _on_speed_unit_change(self):
        old, new = self._speed_unit_current, self.solv_vars["speed_unit"].get()
        if old == new:
            return
        try:
            for key in ("ux", "uy", "uz"):
                e = self.solv_entries[key]
                txt = e.get().strip()
                if txt:
                    e.delete(0, "end")
                    e.insert(0, f"{speed_from_si(speed_to_si(txt, old), new):.4g}")
        except ValueError:
            pass
        self._speed_unit_current = new

    def _on_speed_mode_change(self):
        fn_mode = self.solv_vars["speed_mode"].get() == "Froude Fn_h"
        self.solv_entries["uy"].config(state="disabled" if fn_mode else "normal")
        self.solv_entries["uz"].config(state="disabled" if fn_mode else "normal")
        self.solv_entries["fn"].config(state="normal" if fn_mode else "disabled")

    def snapshot_solver_form(self):
        """Solver form as plain strings (UI thread only)."""
        S, SV = self.solv_entries, self.solv_vars
        d = {k: S[k].get() for k in ("ux", "uy", "uz", "fn", "ratio", "a_ratio", "k_ratio", "t", "dt")}
        d.update(speed_unit=SV["speed_unit"].get(), speed_mode=SV["speed_mode"].get(),
                 dist=SV["dist"].get())
        return d

    def _solver_kwargs(self):
        return parse_solver_dict(self.snapshot_solver_form())

    def _flow_U(self, named):
        U, info = flow_U_from(self.snapshot_solver_form(), named)
        self.speed_info.config(text=info)
        return U

    def on_init_solver(self):
        try:
            named = self._selected_named()
            bc = common_boundary(named)
            U = self._flow_U(named)
            kw, _, _, _ = self._solver_kwargs()
            self.solver = VLMSolver([s for _, s in named], U, bc, ratio=kw["ratio"],
                                    a_ratio=kw["a_ratio"], k_ratio=kw["k_ratio"])
            self.boundary_info.config(text=f"boundary (from surfaces): {bc}")
            self.solv_status.config(text=f"solver OK: {len(named)} surface(s), N_tot={self.solver.N_tot}")
            self._log(f"Init solver: {len(named)} surface(s), boundary={bc}, N_tot={self.solver.N_tot}")
            self.refresh_results()
        except Exception as ex:
            messagebox.showerror("Init solver", str(ex))

    def _set_busy(self, busy):
        for b in (self.btn_init, self.btn_time, self.btn_kutta, self.btn_vent):
            b.config(state="disabled" if busy else "normal")

    def _run_async(self, kind):
        try:
            named = self._selected_named()
            bc = common_boundary(named)
            U = self._flow_U(named)
            kw, t, dt, dist = self._solver_kwargs()
        except Exception as ex:
            messagebox.showerror("Solver", str(ex))
            return
        surfs = [s for _, s in named]
        # Fresh wake state: _time_sim appends gamma_wake rows onto the existing
        # arrays, so re-running on dirty surfaces crashes with a reshape error.
        # _reset() (surface-side only) + solver rebuild fixes time/kutta/vent reruns.
        try:
            for s in surfs:
                s._reset()
            self.solver = VLMSolver(surfs, U, bc, ratio=kw["ratio"],
                                    a_ratio=kw["a_ratio"], k_ratio=kw["k_ratio"])
            self.boundary_info.config(text=f"boundary (from surfaces): {bc}")
        except Exception as ex:
            messagebox.showerror("Init solver", str(ex))
            return
        self._set_busy(True)
        self.solv_status.config(text=f"running {kind}...")
        th = threading.Thread(target=self._worker, args=(kind, t, dt, dist), daemon=True)
        th.start()

    def _worker(self, kind, t, dt, dist):
        try:
            if kind == "time":
                self.solver._time_sim(t=t, dt=dt, distribution=dist)
            elif kind == "kutta":
                # kutta needs a converged wake: always (re)run time_sim first
                # on the freshly reset surfaces, then compute the loads.
                self.solver._time_sim(t=t, dt=dt, distribution=dist)
                self.solver._kuttas_loads()
            elif kind == "vent":
                self.solver._ventilation()
            self.msg_q.put((kind, True, "done"))
        except Exception as ex:
            self.msg_q.put((kind, False, str(ex)))

    def _poll_queue(self):
        try:
            while True:
                kind, ok, msg = self.msg_q.get_nowait()
                if kind == "study_progress":
                    tab, done, total = msg
                    tab.study_bar["value"] = 100.0 * done / max(total, 1)
                    tab.set_status(f"run {done}/{total}")
                    continue
                if kind == "study_log":
                    tab, text = msg
                    tab._log(text)
                    continue
                if kind == "study_done":
                    if ok:
                        tab, text, names, per_surf, dists, posgrid, actual_times = msg
                        tab._study_finished(text, names, per_surf, dists, posgrid, actual_times)
                    else:
                        tab, text = msg
                        tab._study_finished(text, [], None, None, None)
                    continue
                if kind == "trim_progress":
                    tab, done, total = msg
                    tab.trim_bar["value"] = 100.0 * done / max(total, 1)
                    tab.set_status(f"trim {done}/{total}")
                    continue
                if kind == "trim_log":
                    tab, text = msg
                    tab._log(text)
                    continue
                if kind == "trim_done":
                    if ok:
                        tab, text, res = msg
                        tab._trim_finished(text, res)
                    else:
                        tab, text = msg
                        tab._trim_finished(text, None)
                    continue
                self._set_busy(False)
                if ok:
                    self.solv_status.config(text=f"{kind}: OK")
                    self._log(f"{kind}: done")
                    self.refresh_results()
                else:
                    self.solv_status.config(text=f"{kind}: ERROR")
                    self._log(f"{kind} ERROR: {msg}")
                    messagebox.showerror(f"Run {kind}", msg)
        except queue.Empty:
            pass
        self.after(150, self._poll_queue)

    def _log(self, txt):
        self.log.insert("end", txt + "\n")
        self.log.see("end")

    def refresh_results(self):
        if self.solver is None:
            return
        show_mirror = self.solv_vars["mirror"].get()
        rho = medium_density(self.solv_vars["medium"].get())
        funit = self.solv_vars["force_unit"].get()
        u_norm = float(np.linalg.norm(self.solver.U))
        try:
            named = self._selected_named()
        except ValueError:
            named = self.built_surfaces()
        ax3d = self.ax3d
        if self.solv_vars["show_3d"].get():
            draw_3d_panels(ax3d, named, show_mirror=show_mirror, with_wake=True,
                           title=None, equal=True, legend_out=True)
            self.cv3d.draw()
            if not self._3d_shown:
                self.w3d.pack(side="top", fill="x", padx=4, before=self.scroll_wrap)
                self._3d_shown = True
        elif self._3d_shown:
            self.w3d.pack_forget()
            self._3d_shown = False
        kinds = self._visible_kinds()
        if ([n for n, _ in named], kinds) != self._plot_sig:
            self._rebuild_surface_plots(named, kinds)
        vcurves = self._vent_curves()
        for row, (name, surf) in zip(self.surf_plot_rows, named):
            for kind, ax, cv in row["plots"]:
                if kind == "cl":
                    draw_cl_single(ax, name, surf)
                elif kind == "gamma":
                    draw_gamma_single(ax, name, surf)
                elif kind == "vent":
                    draw_vent_single(ax, name, surf, vcurves)
                cv.draw()
        # summary: Froude numbers, global CL + loads per surface + total
        lines = [f"medium={self.solv_vars['medium'].get()} rho={rho:.3f} kg/m3  |U|={u_norm:.2f} m/s  forces in {funit}"]
        load_lines = []
        total = np.zeros(3)
        any_load = False
        for name, surf in named:
            fr_c = froude_chord(surf, u_norm)
            fr_h = froude_depth(surf, u_norm)
            line = f"{name}: Fn chord={fr_c:.3f}" if fr_c is not None else f"{name}: Fn chord=n/a"
            line += f"  Fn depth={fr_h:.3f}" if fr_h is not None else "  Fn depth=n/a (not immersed)"
            lines.append(line)
            F = scaled_loads(surf, rho, funit)
            if F is None:
                lines.append(f"{name}: no loads yet (run kutta loads)")
                continue
            any_load = True
            total = total + F
            cl, cd = surf_coeffs(surf)
            cl_txt = f"{cl:.4f}" if cl is not None else "n/a"
            cd_txt = f"{cd:.4f} (induced)" if cd is not None else "n/a"
            lines.append(f"{name}: CL={cl_txt}  CD_induced={cd_txt}  loads [{funit}]=[{F[0]:.2f}, {F[1]:.2f}, {F[2]:.2f}]")
            load_lines.append(lines[-1])
        if any_load:
            lines.append(f"TOTAL: loads [{funit}]=[{total[0]:.2f}, {total[1]:.2f}, {total[2]:.2f}]")
            load_lines.append(lines[-1])
            self._log(" | ".join(load_lines))
        self.results_text.config(state="normal")
        self.results_text.delete("1.0", "end")
        self.results_text.insert("end", "\n".join(lines))
        self.results_text.config(state="disabled")

    def export_solver_csv(self):
        if self.solver is None:
            messagebox.showerror("Export CSV", "nothing to export (init the solver first)")
            return
        from tkinter import filedialog as _fd
        path = _fd.asksaveasfilename(defaultextension=".csv", initialdir=os.path.join(ROOT, "opencode"),
                                     initialfile="solver.csv", filetypes=[("CSV", "*.csv")])
        if not path:
            return
        try:
            named = self._selected_named()
        except ValueError as ex:
            messagebox.showerror("Export CSV", str(ex))
            return
        try:
            rho = medium_density(self.solv_vars["medium"].get())
            funit = self.solv_vars["force_unit"].get()
            u_norm = float(np.linalg.norm(self.solver.U))
            rows, total = [], np.zeros(3)
            any_load = False
            for name, surf in named:
                F = scaled_loads(surf, rho, funit)
                if F is None:
                    rows.append([name, surf.area, None, None, None, None, None, None])
                    continue
                any_load = True
                total = total + F
                cl, cd = surf_coeffs(surf)
                rows.append([name, surf.area, cl, cd, F[0], F[1], F[2],
                             froude_chord(surf, u_norm), froude_depth(surf, u_norm)])
            if not any_load:
                raise ValueError("no loads yet (run kutta loads)")
            area_tot = sum((r[1] or 0) for r in rows)
            rows.append(["TOTAL", area_tot, None, None, total[0], total[1], total[2], None, None])
            write_csv(path, ["surface", "area_m2", "CL", "CD_induced", f"Fx[{funit}]", f"Fy[{funit}]",
                             f"Fz[{funit}]", "Fn_chord", "Fn_depth"], rows)
            self._log(f"exported {path}")
            if self.exp_dist_solv.get():
                self._export_solver_dist(path, named, rho, funit)
        except Exception as ex:
            messagebox.showerror("Export CSV", str(ex))

    def _export_solver_dist(self, base_path, named, rho, funit):
        rows = []
        for name, surf in named:
            d = surface_dist(surf)
            s_list, g_list, c_list = d["s"] or [], d["gamma"] or [], d["cl"] or []
            if not s_list and not g_list and not c_list:
                continue
            lo = d["loads"] or [None, None, None]
            F = ([v * rho * force_factor(funit) for v in lo]
                 if all(v is not None for v in lo) else [None, None, None])
            for k in range(max(len(s_list), len(g_list), len(c_list))):
                rows.append([name,
                             s_list[k] if k < len(s_list) else None,
                             g_list[k] if k < len(g_list) else None,
                             c_list[k] if k < len(c_list) else None,
                             F[0], F[1], F[2]])
        root, _ = os.path.splitext(base_path)
        dist_path = root + "_dist.csv"
        write_csv(dist_path, ["surface", "s", "gamma", "Cl_2d",
                              f"Fx[{funit}]", f"Fy[{funit}]", f"Fz[{funit}]"], rows)
        self._log(f"exported {dist_path}")


def _fix_windows_dpi():
    """Declare the process DPI-aware before Tk starts, otherwise Windows
    bitmap-scales the whole window and the interface looks blurry."""
    try:
        if sys.platform == "win32":
            import ctypes
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(1)
            except Exception:
                ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


def main():
    _fix_windows_dpi()
    app = VLMGui()
    app.mainloop()


if __name__ == "__main__":
    main()
