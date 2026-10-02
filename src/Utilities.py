import copy
import sys
import numpy             as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import MaxNLocator

def chord_fn(n, ar, b, sym, space, shape, lam=1.8):
    """
    Return chord length distribution along spanwise direction according to the
    desired planform shape.

    Input:
        n     -> number of spanwise stations
        ar    -> aspect ratio
        b     -> wing span
        sym   -> symmetric or not
        space -> uniform or cosine spanwise spacing
        shape -> wing planform shape, can be "rectangular", "elliptical" or "tapered"
        lam   -> taper ratio, only needed for tapered shape

    Output:
        c     -> chord length distribution along spanwise direction, shape (n,)
    """
    if sym :
        le = (0.5 * b)
    else :
         le = b
    s_min, s_max = 0.0, le
    if space :
        if sym :
            theta = np.linspace(np.pi/2, np.pi, n)
            s = le*(-np.cos(theta))
        else :
            theta = np.linspace(0, np.pi, n)
            s = le*(1-np.cos(theta))/2
    else :
        s = np.linspace(s_min, s_max, n)
    match shape:
        case "rectangular":
            c = b/ar * np.ones(n)
        case "elliptical":
            c = 4*b/(np.pi*ar) * np.sqrt(1 - (0.999*s/le)**2)
        case "tapered":
            c = - s*4*(lam - 1)/(ar*(lam + 1)) + 4*le*lam/(ar*(lam + 1))
    return c

def plot3D(surfaces, show_mirror=False, show_norm=False):
    """ 
    Plot all the surface in 3D

    Input
        surfaces    -> a list of VLMSurface
        show_mirror -> enable the display of boundary conditions 
    """
    fig_3d = plt.figure(figsize=(14, 8),constrained_layout=True)
    ax4 = fig_3d.add_subplot(111, projection='3d')
    first = {"wing":True, "wake":True, "mir_wing":True, "mir_wake":True}             # to only have one legend per item
    for surface in surfaces:
        normals = surface.normals
        for i, panel in enumerate(surface.wing_panels["real"]):
                pnt    = copy.copy(panel.pnt)
                vrt    = copy.copy(panel.vrt)
                # Close the polygon shape
                pnt.append(pnt[0])
                pnt_plt = np.array(pnt)
                vrt.append(vrt[0])
                vrt_plt = np.array(vrt)
                
                if show_norm :
                    ax4.quiver(panel.ctr[1], panel.ctr[0], panel.ctr[2], normals[i][1], normals[i][0], normals[i][2], length=np.linalg.norm(panel.chord), color='blue')

                if first["wing"] :               # In order to have just one legend
                        ax4.plot(pnt_plt[:, 1], pnt_plt[:, 0], pnt_plt[:, 2], color='black', label='Wing')
                        ax4.plot(vrt_plt[:, 1], vrt_plt[:, 0], vrt_plt[:, 2], color='red', lw=0.8, label='Vortex rings')
                        first["wing"] = False
                else :
                        ax4.plot(pnt_plt[:, 1], pnt_plt[:, 0], pnt_plt[:, 2], color='black')
                        ax4.plot(vrt_plt[:, 1], vrt_plt[:, 0], vrt_plt[:, 2], color='red', lw=0.8)
        for i, panel in enumerate(surface.wake_panels["real"]):
            pnt    = copy.copy(panel.pnt)
            # Close the polygon shape
            pnt.append(pnt[0])
            pnt_plt = np.array(pnt)
            if first["wake"]:               # In order to have just one legend
                    ax4.plot(pnt_plt[:, 1], pnt_plt[:, 0], pnt_plt[:, 2], color='blue', lw=0.6, label='Wake')
                    first["wake"] = False
            else :
                    ax4.plot(pnt_plt[:, 1], pnt_plt[:, 0], pnt_plt[:, 2], color='blue', lw=0.6)
        if show_mirror:
            for i, panel in enumerate(surface.wing_panels["mirror"]):
                    pnt    = copy.copy(panel.pnt)
                    vrt    = copy.copy(panel.vrt)
                    # Close the polygon shape
                    pnt.append(pnt[0])
                    pnt_plt = np.array(pnt)
                    vrt.append(vrt[0])
                    vrt_plt = np.array(vrt)
                    if first["mir_wing"] :               # In order to have just one legend
                            ax4.plot(pnt_plt[:, 1], pnt_plt[:, 0], pnt_plt[:, 2], color='dimgray', label='Mirrored wing')
                            ax4.plot(vrt_plt[:, 1], vrt_plt[:, 0], vrt_plt[:, 2], color='forestgreen', lw=0.8, label='Mirrored vortex rings')
                            first["mir_wing"] = False
                    else :
                            ax4.plot(pnt_plt[:, 1], pnt_plt[:, 0], pnt_plt[:, 2], color='dimgray')
                            ax4.plot(vrt_plt[:, 1], vrt_plt[:, 0], vrt_plt[:, 2], color='forestgreen', lw=0.8)
            for i, panel in enumerate(surface.wake_panels["mirror"]):
                pnt    = copy.copy(panel.pnt)
                # Close the polygon shape
                pnt.append(pnt[0])
                pnt_plt = np.array(pnt)
                if first["mir_wake"]:               # In order to have just one legend
                        ax4.plot(pnt_plt[:, 1], pnt_plt[:, 0], pnt_plt[:, 2], color='darkviolet', lw=0.6, label='Mirrored wake')
                        first["mir_wake"] = False
                else :
                        ax4.plot(pnt_plt[:, 1], pnt_plt[:, 0], pnt_plt[:, 2], color='darkviolet', lw=0.6)
            
    ax4.view_init(elev=30, azim=40)
    x_lim = ax4.get_xlim3d()
    y_lim = ax4.get_ylim3d()
    z_lim = ax4.get_zlim3d()
    ax4.set_xlim(x_lim[0],x_lim[1])
    ax4.set_ylim(y_lim[0],y_lim[1]) 
    ax4.set_zlim(z_lim[1],z_lim[0])
    ax4.set_box_aspect([-x_lim[0]+x_lim[1],-y_lim[0]+y_lim[1],-z_lim[0]+z_lim[1]])
    ax4.set_xlabel("y", fontstyle='italic')
    ax4.set_ylabel("x", fontstyle='italic')
    ax4.set_zlabel("z", fontstyle='italic')
    ax4.xaxis.set_major_locator(MultipleLocator(0.5))
    ax4.yaxis.set_major_locator(MultipleLocator(1))
    ax4.zaxis.set_major_locator(MultipleLocator(0.5))

    ax4.set_title('3D view')
    ax4.legend()

    
def plot2D(surfaces, show_mirror, axis):
    """ 
    Plot all the surfaces in 2D

    Input
        surfaces    -> a list of VLMSurface
        show_mirror -> enable the display of boundary conditions 
        axis        -> axis that goes out - 0 for x ; 1 for y ; 2 for z
    """
    fig_2d = plt.figure(figsize=(14/1.5, 8/1.5),constrained_layout=True)
    ax4 = fig_2d.add_subplot()
    first = {"wing":True, "wake":True, "mir_wing":True, "mir_wake":True}             # to only have one legend per item
    for surface in surfaces:
        for i, panel in enumerate(surface.wing_panels["real"]):
                pnt    = copy.copy(panel.pnt)
                vrt    = copy.copy(panel.vrt)
                ctr    = copy.copy(panel.ctr)
                # Close the polygon shape
                pnt.append(pnt[0])
                pnt_plt = np.array(pnt)
                vrt.append(vrt[0])
                vrt_plt = np.array(vrt)
                if first["wing"] :               # In order to have just one legend
                        ax4.plot(pnt_plt[:, (axis+1)%3], pnt_plt[:, (axis+2)%3], color='black', label='Wing panels')
                        ax4.plot(vrt_plt[:, (axis+1)%3], vrt_plt[:, (axis+2)%3], color='red', lw=0.8, label='Vortex rings')
                        #ax4.plot(ctr[(axis+1)%3], ctr[(axis+2)%3], "o", ms=5, markerfacecolor='None', markeredgecolor='red',  label='Control points')
                        first["wing"] = False
                else :
                        ax4.plot(pnt_plt[:, (axis+1)%3], pnt_plt[:, (axis+2)%3], color='black')
                        ax4.plot(vrt_plt[:, (axis+1)%3], vrt_plt[:, (axis+2)%3], color='red', lw=0.8)
                        #ax4.plot(ctr[(axis+1)%3], ctr[(axis+2)%3], "o", ms=5, markerfacecolor='None', markeredgecolor='red')
        for i, panel in enumerate(surface.wake_panels["real"]):
            pnt    = copy.copy(panel.pnt)
            # Close the polygon shape
            pnt.append(pnt[0])
            pnt_plt = np.array(pnt)
            if first["wake"]:               # In order to have just one legend
                    ax4.plot(pnt_plt[:, (axis+1)%3], pnt_plt[:, (axis+2)%3], color='blue', lw=0.6, label='Wake panels')
                    first["wake"] = False
            else :
                    ax4.plot(pnt_plt[:, (axis+1)%3], pnt_plt[:, (axis+2)%3], color='blue', lw=0.6)
        if show_mirror:
            for i, panel in enumerate(surface.wing_panels["mirror"]):
                    pnt    = copy.copy(panel.pnt)
                    vrt    = copy.copy(panel.vrt)
                    # Close the polygon shape
                    pnt.append(pnt[0])
                    pnt_plt = np.array(pnt)
                    vrt.append(vrt[0])
                    vrt_plt = np.array(vrt)
                    if first["mir_wing"] :               # In order to have just one legend
                            ax4.plot(pnt_plt[:, (axis+1)%3], pnt_plt[:, (axis+2)%3], color='dimgray', label='Mirrored wing')
                            ax4.plot(vrt_plt[:, (axis+1)%3], vrt_plt[:, (axis+2)%3], color='forestgreen', lw=0.8, label='Mirrored vortex rings')
                            first["mir_wing"] = False
                    else :
                            ax4.plot(pnt_plt[:, (axis+1)%3], pnt_plt[:, (axis+2)%3], color='dimgray')
                            ax4.plot(vrt_plt[:, (axis+1)%3], vrt_plt[:, (axis+2)%3], color='forestgreen', lw=0.8)
            for i, panel in enumerate(surface.wake_panels["mirror"]):
                pnt    = copy.copy(panel.pnt)
                # Close the polygon shape
                pnt.append(pnt[0])
                pnt_plt = np.array(pnt)
                if first["mir_wake"]:               # In order to have just one legend
                        ax4.plot(pnt_plt[:, (axis+1)%3], pnt_plt[:, (axis+2)%3], color='darkviolet', lw=0.6, label='Mirrored wake')
                        first["mir_wake"] = False
                else :
                        ax4.plot(pnt_plt[:, (axis+1)%3], pnt_plt[:, (axis+2)%3], color='darkviolet', lw=0.6)
            
    if axis == 0 :
        ax4.invert_yaxis()
        ax4.set_xlabel(r'$y$' + " [m]" )
        ax4.set_ylabel(r'$z$' + " [m]")
        ax4.set_title('Back view')
    elif axis == 1 :
        for line in ax4.lines:
            xdata, ydata = line.get_xdata(), line.get_ydata()
            line.set_data(ydata, xdata)
        ax4.relim()
        ax4.invert_yaxis()
        ax4.set_xlabel(r'$x$' + " [m]")
        ax4.set_ylabel(r'$z$' + " [m]")
        ax4.set_title('Lateral view')
    else :
        for line in ax4.lines:
            xdata, ydata = line.get_xdata(), line.get_ydata()
            line.set_data(ydata, xdata)
        ax4.set_xlabel(r'$y$' + " [m]")
        ax4.set_ylabel(r'$x$' + " [m]")
        ax4.relim()
        ax4.invert_yaxis()
        ax4.set_title('Top view')

    x_lim = ax4.get_xlim()
    y_lim = ax4.get_ylim()
    ax4.set_xlim(x_lim[0],x_lim[1])
    ax4.set_ylim(y_lim[0],y_lim[1]) 
    ax4.set_aspect('equal')
    
    ax4.grid(True)
    ax4.legend()
    # =======================================================
    # FIN DE LA FONCTION : Placement robuste sans collision
    # =======================================================
    # 'adjustable=box' est crucial pour garder tes chiffres et ton échelle 100% exacts
    ax4.set_aspect('equal', adjustable='box') 
    
    ax4.xaxis.set_major_locator(MultipleLocator(0.1))
    ax4.yaxis.set_major_locator(MultipleLocator(0.1))

    # On force le titre à monter légèrement pour ne jamais être collé au cadre
    ax4.set_title(ax4.get_title(), y=1.08)

    # 1. On place la légende à l'extérieur droit de la figure
    # On utilise loc="center left" et on la décale à X = 1.05 (juste après la fin du graphique)
    ax4.legend(
        loc="center left", 
        bbox_to_anchor=(1.05, 0.5), # Centrée verticalement à droite
        borderaxespad=0,
        ncol=1 # En colonne simple pour rester propre à droite
    )

    # 2. On ajuste manuellement les marges de la figure globale (14, 8)
    # On laisse 25% d'espace vide à droite (right=0.75) pour que la légende y respire 
    # sans jamais compresser le graphique ou toucher le titre.
    fig_2d.subplots_adjust(left=0.08, right=0.75, top=0.85, bottom=0.15)
    ax4.xaxis.set_major_locator(MaxNLocator(nbins=8, prune='both'))
    ax4.yaxis.set_major_locator(MaxNLocator(nbins=6, prune='both'))

def load_csv(filepath):
    """
    Lit le fichier depuis le disque, remplace les virgules par des points 
    et sépare les colonnes X et Y.
    """
    x_list = []
    y_list = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                # On ignore les lignes vides
                line = line.strip()
                if not line:
                    continue
                
                # Nettoyage : virgule décimale -> point
                clean_line = line.replace(',', '.')
                
                # Séparation par le point-virgule (vu dans ton exemple précédent)
                if ';' in clean_line:
                    parts = clean_line.split(';')
                    x_list.append(float(parts[0].strip()))
                    y_list.append(float(parts[1].strip()))
        
        return np.array(x_list), np.array(y_list)
        
    except FileNotFoundError:
        print(f"Erreur : Le fichier est introuvable à l'adresse : {filepath}")
        return None, None
    except Exception as e:
        print(f"Erreur lors du traitement : {e}")
        return None, None


def plot_ventilation(surface, harwood):
    """ 
    Plot the distributions of the ventilation parameters along the spanwise direction 
    
    """
    fig = plt.figure(figsize=(7/1, 4/1),constrained_layout=True)
    ax  = fig.add_subplot()

    if harwood == True :
        x, y = load_csv("data/cavity.csv")
        sorted_ind = np.argsort(x)
        x, y_bottom = x[sorted_ind], y[sorted_ind]
        y_top = np.zeros_like(y_bottom)
        ax.fill_between(x, y_top, y_bottom, 
                    facecolor='#90EE90',  # Vert clair (LightGreen)
                    alpha=0.6, 
                    edgecolor='none')
        ax.plot(x, y_bottom, lw=0.8, color='black', label=r"$L_c/c$ experimental")

        x, y = load_csv("data/Lc.csv")
        sorted_ind = np.argsort(x)
        x, y = x[sorted_ind], y[sorted_ind]
        ax.plot(x, y, "--", color="gray", label=r'$L_c/c$'+"  LLM")
        x, y = load_csv("data/cl.csv")
        sorted_ind = np.argsort(y)
        x, y = x[sorted_ind], y[sorted_ind]
        ax.plot(x, y, "--", color="black", label=r'$C_{l_\text{2D}}$'+"  LLM")
        x, y = load_csv("data/alpha.csv")
        sorted_ind = np.argsort(y)
        x, y = x[sorted_ind], y[sorted_ind]
        ax.plot(x, y, "--", color="red", label=r'$\alpha_{\text{eff}}/\alpha$'+"  LLM")
        x, y = load_csv("data/a0.csv")
        sorted_ind = np.argsort(y)
        x, y = x[sorted_ind], y[sorted_ind]
        ax.plot(x, y, "--", color="magenta", label=r'$a_0/(2\pi)$'+"  LLM")
        x, y = load_csv("data/sigma.csv")
        sorted_ind = np.argsort(y)
        x, y = x[sorted_ind], y[sorted_ind]
        ax.plot(x, y, "--", color="blue", label=r'$\sigma_c$')


    M, N, B = surface.M, surface.N, surface.span
    ctrl    = np.array([p.ctr for p in surface.wing_panels["real"]])
    z       = np.sum(ctrl.reshape((M,N,3)), axis=0)[:,2]/M
    z       = z/B
    vent    = surface.ventilation

    ax.plot(vent["Lc/c"], z, color="gray", label=r'$L_c/c$'+"  VLM")
    ax.plot(vent["a0"]*np.sin(vent["alpha_eff"]),z, color="black", label=r'$C_{l_\text{2D}}$'+"  VLM")
    ax.plot(vent["alpha_eff"]/(vent["avg_aoa"]), z, color="red", label=r'$\alpha_{\text{eff}}/\alpha$'+"  VLM")
    ax.plot(vent["a0"]/(2*np.pi), z, color="magenta", label=r'$a_0/(2\pi)$'+"  VLM")
    ax.plot(vent["sigma_c"], z, color="blue", label=r'$\sigma_c$'+"  VLM")
    ax.plot()
    ax.set_xlim(-0.2, 1.9)
    ax.set_ylim(1.2,-0.3)
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.set_xlabel(r'$x/c$')
    ax.set_ylabel(r'$z/h$')
    #ax.grid()
    ax.legend(loc="lower right")


def trim_foils(build, fly_heights, Fz_target=160.0, csv_path=None, shape=None, Fy_targets=None,
               alpha0=6.8, beta0=1.22, dalpha=0.5, dbeta=0.5,
               tol=0.5, tol_h=None, maxit=15, damping=0.7,
               t=None, dt=None, distribution="classic", verbose=True):
    """
    Trim alpha (vertical force) and beta (horizontal force) to the operating
    point at each flying height. Vertical and horizontal are treated as
    decoupled : alpha is corrected on the vertical error, beta on the
    horizontal one. No ventilation coupling.

    Input:
        build       -> callable (alpha_deg, beta_deg, fly_height) returning a VLMSolver
        fly_heights -> array of flying heights [m]
        Fz_target   -> vertical force target [kgf]
        csv_path    -> csv file with (flying height [m] ; horizontal target [kgf]) columns,
                       read with load_csv and interpolated at fly_heights
        shape       -> foil shape letter ("L", "C", ...) : loads data/<LETTER>_h_load.csv
                       (takes precedence over Fy_targets, csv_path wins if both given)
        Fy_targets  -> horizontal targets [kgf], scalar or array (alternative to csv_path)
        alpha0/beta0-> initial guesses [deg], warm-started from the previous height after that
        dalpha/dbeta-> finite-difference steps [deg] used once per height to measure dF/dalpha, dF/dbeta
        tol/tol_h   -> convergence tolerances [kgf] on vertical / horizontal (tol_h defaults to tol)
        maxit       -> max correction iterations per height
        damping     -> under-relaxation factor on the corrections
        t/dt        -> time-simulation parameters (solver defaults when None)

    Output:
        alpha_deg, beta_deg -> trimmed angles [deg] at each flying height (input order)
    """
    H = np.atleast_1d(np.asarray(fly_heights, dtype=float))
    if tol_h is None:
        tol_h = tol
    if csv_path is None and shape is not None:
        csv_path = f"data/{str(shape).strip()[0].upper()}_h_load.csv"
    if csv_path is not None:
        x, y = load_csv(csv_path)
        if x is None:
            raise ValueError(f"could not load horizontal targets from {csv_path}")
        o = np.argsort(x)
        Fy_t = np.interp(H, x[o], y[o])
    elif Fy_targets is not None:
        Fy_t = np.full_like(H, float(Fy_targets)) if np.ndim(Fy_targets) == 0 else np.asarray(Fy_targets, dtype=float)
    else:
        raise ValueError("provide horizontal targets via csv_path or Fy_targets")

    def forces(vlm):
        F = 1025*np.sum([s.loads for s in vlm.surfaces], axis=0)/9.81   # [kgf], z positive downwards
        return -F[2], F[1]                                              # vertical (up), horizontal

    def solve(a, b, h):
        vlm = build(a, b, h)
        if t is None:
            vlm._time_sim()
        else:
            vlm._time_sim(t=t, dt=dt, distribution=distribution)
        vlm._kuttas_loads()
        return forces(vlm)

    order = np.argsort(H)
    alpha_sorted = np.zeros_like(H)
    beta_sorted = np.zeros_like(H)
    a, b = alpha0, beta0
    for rank, idx in enumerate(order):
        h, fy_t = H[idx], Fy_t[idx]
        Fz, Fy = solve(a, b, h)
        s_va = (solve(a + dalpha, b, h)[0] - Fz)/dalpha     # dFz/dalpha [kgf/deg]
        s_hb = (solve(a, b + dbeta, h)[1] - Fy)/dbeta       # dFy/dbeta  [kgf/deg]
        if abs(s_va) < 1e-9 or abs(s_hb) < 1e-9:
            raise ValueError(f"null slope at H={h} m (dFz/dalpha={s_va:.3f}, dFy/dbeta={s_hb:.3f})")
        k = 0
        while k < maxit and (abs(Fz_target - Fz) > tol or abs(fy_t - Fy) > tol_h):
            k += 1
            a += damping*(Fz_target - Fz)/s_va
            b += damping*(fy_t - Fy)/s_hb
            Fz, Fy = solve(a, b, h)
            if verbose:
                print(f"H={h:.2f} m it={k} alpha={a:.3f} beta={b:.3f} Fz={Fz:.1f}/{Fz_target:.1f} Fy={Fy:.1f}/{fy_t:.1f}")
        if k == maxit and (abs(Fz_target - Fz) > tol or abs(fy_t - Fy) > tol_h):
            print(f"Warning : did not converge at H={h} m (Fz={Fz:.1f}/{Fz_target:.1f} Fy={Fy:.1f}/{fy_t:.1f})")
        alpha_sorted[rank] = a
        beta_sorted[rank] = b
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    return alpha_sorted[inv], beta_sorted[inv]


def _study_cell(raw):
    """Parse one GUI-export cell: "" -> None, number (comma decimals) ->
    float, anything else -> stripped string."""
    t = str(raw).strip()
    if t == "":
        return None
    try:
        return float(t.replace(",", "."))
    except ValueError:
        return t


def _study_lines(path):
    """Non-empty lines of a GUI CSV split on ';' (header included)."""
    with open(path, "r", encoding="utf-8") as f:
        rows = []
        for line in f:
            if line.strip():
                rows.append([c.strip() for c in line.rstrip("\n").split(";")])
    if not rows:
        raise ValueError(f"{path}: empty file")
    return rows


def _study_header(rows):
    """True if the first row looks like a header (contains P1/CL/...)."""
    first = {c.strip().lower() for c in rows[0]}
    return bool(first & {"surface", "p1", "p1val", "cl", "alpha_deg"})


def _study_col(rows, header, names):
    """Column index from a header row (case-insensitive, Fx[N] matches fx)."""
    norm = [h.strip().lower().split("[")[0] for h in header]
    for n in names:
        if n in norm:
            return norm.index(n)
    raise ValueError(f"column {names[0]!r} not found in header {header}")


def load_study_csv(path):
    """
    Load a Study `study.csv` export (CL vs sweep) for hand plotting.

    Input:
        path -> file written by StudyTab.export_csv, header
                surface;P1;P1val;P2;P2val;CL (";" + comma decimals)

    Output:
        dict(p1, p2, surfaces, data) with
        data[surface][p2_raw] = dict(x, CL, x_vals, x_raw, p2val).
        x/Cl are float arrays (nan when a cell is not numeric),
        x_vals the parsed values (float/str/None), p2_raw the exact
        P2val cell text ("" when P2 is "none", i.e. one curve/surface).
        Curves are sorted on x when fully numeric, else kept in file order.

    Example:
        d = load_study_csv("opencode/study.csv")
        c = d["data"]["foil"][""]
        plt.plot(c["x"], c["CL"])
    """
    rows = _study_lines(path)
    body = rows[1:] if _study_header(rows) else rows
    if body and len(body[0]) < 6:
        raise ValueError(f"{path}: need 6 columns (surface;P1;P1val;P2;P2val;CL)")
    p1 = p2 = None
    data, order = {}, []
    for r in body:
        if len(r) < 6:
            continue
        surf, p1lab, p1raw, p2lab, p2raw, clraw = (c.strip() for c in r[:6])
        if p1 is None:
            p1, p2 = p1lab or None, p2lab or None
        cl = _study_cell(clraw)
        if cl is None or (isinstance(cl, float) and np.isnan(cl)):
            continue                                        # same skip as export
        xv = _study_cell(p1raw)
        curves = data.setdefault(surf, {})
        if surf not in order:
            order.append(surf)
        curve = curves.setdefault(p2raw, {"x_raw": [], "x_vals": [],
                                          "cl": [], "p2val": _study_cell(p2raw)})
        curve["x_raw"].append(p1raw)
        curve["x_vals"].append(xv)
        curve["cl"].append(float(cl) if isinstance(cl, (int, float)) else np.nan)
    out = {}
    for surf, curves in data.items():
        out[surf] = {}
        for key, c in curves.items():
            x = np.array([v if isinstance(v, (int, float)) else np.nan
                          for v in c["x_vals"]], dtype=float)
            cl = np.array(c["cl"], dtype=float)
            if len(x) and np.all(~np.isnan(x)):
                o = np.argsort(x, kind="stable")
                x, cl = x[o], cl[o]
                c["x_raw"] = [c["x_raw"][i] for i in o]
                c["x_vals"] = [c["x_vals"][i] for i in o]
            out[surf][key] = {"x": x, "CL": cl, "x_vals": c["x_vals"],
                              "x_raw": c["x_raw"], "p2val": c["p2val"]}
    return {"p1": p1, "p2": p2, "surfaces": order, "data": out}


def load_study_dist_csv(path):
    """
    Load a Study `study_dist.csv` export (spanwise distributions).

    Input:
        path -> file written by StudyTab._export_dist, header
                surface;P1;P1val;P2val;CL;s;gamma;Cl_2d;Fx[u];Fy[u];Fz[u]

    Output:
        dict(p1, entries), one entry per (surface, P1val, P2val, CL)
        block (station rows are consecutive in the file):
        dict(surface, p1_raw, p1val, p2_raw, p2val, CL,
             s, gamma, cl, Fx, Fy, Fz, funit).
        s/gamma/cl are float arrays in file order (nan for empty cells),
        Fx/Fy/Fz the (constant) total loads of the point, funit the unit
        read from the Fx header ("N"/"kgf", None without header).

    Example:
        d = load_study_dist_csv("opencode/study_dist.csv")
        e = d["entries"][0]
        plt.plot(e["cl"], e["s"])
    """
    rows = _study_lines(path)
    if _study_header(rows):
        header, body = rows[0], rows[1:]
        i_surf = _study_col(rows, header, ["surface"])
        i_p1 = _study_col(rows, header, ["p1"])
        i_p1v = _study_col(rows, header, ["p1val"])
        i_p2v = _study_col(rows, header, ["p2val"])
        i_cl = _study_col(rows, header, ["cl"])
        i_s = _study_col(rows, header, ["s"])
        i_g = _study_col(rows, header, ["gamma"])
        i_c = _study_col(rows, header, ["cl_2d", "cl_2d".replace("_", "")])
        i_fx = _study_col(rows, header, ["fx"])
        i_fy = _study_col(rows, header, ["fy"])
        i_fz = _study_col(rows, header, ["fz"])
        funit = None
        hfx = header[i_fx]
        if "[" in hfx and "]" in hfx:
            funit = hfx.split("[", 1)[1].split("]")[0].strip() or None
    else:
        body = rows
        i_surf, i_p1, i_p1v, i_p2v, i_cl = 0, 1, 2, 3, 4
        i_s, i_g, i_c, i_fx, i_fy, i_fz = 5, 6, 7, 8, 9, 10
        funit = None
    need = max(i_surf, i_p1, i_p1v, i_p2v, i_cl,
               i_s, i_g, i_c, i_fx, i_fy, i_fz) + 1
    if body and len(body[0]) < need:
        raise ValueError(f"{path}: need {need} columns (surface;P1;P1val;..."
                         "P2val;CL;s;gamma;Cl_2d;Fx;Fy;Fz)")
    p1 = None
    entries, cur, curkey = [], None, None
    def _f(raw):
        v = _study_cell(raw)
        return float(v) if isinstance(v, (int, float)) else np.nan
    for r in body:
        if len(r) < need:
            continue
        get = lambda i: r[i].strip() if i < len(r) else ""
        surf, p1lab, p1raw, p2raw, clraw = (get(i) for i in
                                            (i_surf, i_p1, i_p1v, i_p2v, i_cl))
        if p1 is None:
            p1 = p1lab or None
        key = (surf, p1raw, p2raw, clraw)
        if key != curkey:
            curkey = key
            cur = {"surface": surf, "p1_raw": p1raw,
                   "p1val": _study_cell(p1raw), "p2_raw": p2raw,
                   "p2val": _study_cell(p2raw),
                   "CL": _study_cell(clraw),
                   "s": [], "gamma": [], "cl": [],
                   "Fx": None, "Fy": None, "Fz": None, "funit": funit}
            entries.append(cur)
        cur["s"].append(_f(get(i_s)))
        cur["gamma"].append(_f(get(i_g)))
        cur["cl"].append(_f(get(i_c)))
        for k, i in (("Fx", i_fx), ("Fy", i_fy), ("Fz", i_fz)):
            v = _study_cell(get(i))
            if isinstance(v, (int, float)) and cur[k] is None:
                cur[k] = float(v)
    for e in entries:
        e["s"] = np.array(e["s"], dtype=float)
        e["gamma"] = np.array(e["gamma"], dtype=float)
        e["cl"] = np.array(e["cl"], dtype=float)
        if isinstance(e["CL"], (int, float)):
            e["CL"] = float(e["CL"])
        else:
            e["CL"] = None
    return {"p1": p1, "entries": entries}


def load_trim_csv(path):
    """
    Load a Trim `trim.csv` export for hand plotting.

    Input:
        path -> file written by TrimTab.export_csv, header
                P1;P1val;alpha_deg;beta_deg;Fz[u];Fy[u];Fz_target[u];Fy_target[u]

    Output:
        dict(p1, funit, x, x_vals, x_raw,
             alpha, beta, fz, fy, fz_target, fy_target).
        x/alpha/... are float arrays in file order (nan for empty cells),
        x_vals the parsed sweep values, funit the unit read from the
        Fz header ("N"/"kgf", None without header).

    Example:
        d = load_trim_csv("opencode/trim.csv")
        plt.plot(d["x"], d["alpha"], label="alpha")
        plt.plot(d["x"], d["beta"], label="beta")
    """
    rows = _study_lines(path)
    if _study_header(rows):
        header, body = rows[0], rows[1:]
        i_p1 = _study_col(rows, header, ["p1"])
        i_p1v = _study_col(rows, header, ["p1val"])
        i_a = _study_col(rows, header, ["alpha_deg", "alpha"])
        i_b = _study_col(rows, header, ["beta_deg", "beta"])
        i_fz = _study_col(rows, header, ["fz"])
        i_fy = _study_col(rows, header, ["fy"])
        i_fzt = _study_col(rows, header, ["fz_target"])
        i_fyt = _study_col(rows, header, ["fy_target"])
        funit = None
        hfz = header[i_fz]
        if "[" in hfz and "]" in hfz:
            funit = hfz.split("[", 1)[1].split("]")[0].strip() or None
    else:
        body = rows
        i_p1, i_p1v, i_a, i_b = 0, 1, 2, 3
        i_fz, i_fy, i_fzt, i_fyt = 4, 5, 6, 7
        funit = None
    need = max(i_p1, i_p1v, i_a, i_b, i_fz, i_fy, i_fzt, i_fyt) + 1
    if body and len(body[0]) < need:
        raise ValueError(f"{path}: need {need} columns (P1;P1val;alpha_deg;..."
                         "beta_deg;Fz;Fy;Fz_target;Fy_target)")
    p1 = None
    xr, xv, a, b, fz, fy, fzt, fyt = [], [], [], [], [], [], [], []
    def _f(raw):
        v = _study_cell(raw)
        return float(v) if isinstance(v, (int, float)) else np.nan
    for r in body:
        if len(r) < need:
            continue
        get = lambda i: r[i].strip() if i < len(r) else ""
        if p1 is None:
            p1 = get(i_p1) or None
        xr.append(get(i_p1v))
        xv.append(_study_cell(get(i_p1v)))
        a.append(_f(get(i_a)))
        b.append(_f(get(i_b)))
        fz.append(_f(get(i_fz)))
        fy.append(_f(get(i_fy)))
        fzt.append(_f(get(i_fzt)))
        fyt.append(_f(get(i_fyt)))
    x = np.array([v if isinstance(v, (int, float)) else np.nan for v in xv],
                 dtype=float)
    return {"p1": p1, "funit": funit, "x": x, "x_vals": xv, "x_raw": xr,
            "alpha": np.array(a, dtype=float), "beta": np.array(b, dtype=float),
            "fz": np.array(fz, dtype=float), "fy": np.array(fy, dtype=float),
            "fz_target": np.array(fzt, dtype=float),
            "fy_target": np.array(fyt, dtype=float)}