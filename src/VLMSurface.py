import numpy as np
import copy as copy
import sys

sys.path.append("./src/")
from src.VLMPanel import VLMPanel

class VLMSurface:
    """ 
    Vortex Lattice Methode lifting surface
    Contains all the information about a lifting surface and its sheded wake, including its boundary condition
    """
    #--------------------#
    #   Initialization   #
    #--------------------#
    def __init__(self, origin, plan, boundary, shedding, b, c, alpha, beta, lamb, delta, phi, sym, space, n, m, vent, root_ang=0, tip_ang=0, radius=1, center=1, fly_height=0.0):
        
        # Wing geometry
        self.origin    = origin     # starting point of the wing - will end at the middle if it's a symmetric wing
        self.plan      = plan       # plan in wich the wing is based - 0 for z-x, 1 for y-x
        self.span      = b          # wing span
        self.chord     = c          # chord law distribution
        self.sweep     = lamb       # middle chord sweep angle  [rad]
        self.dihedron  = delta      # dihedral angle            [rad]
        self.twist     = phi        # quarter chord twist angle [rad]
        self.symmetric = sym        # True for a symmetric wing
        self.root_ang  = root_ang   
        self.tip_ang   = tip_ang
        self.radius    = radius
        self.center    = center
        self.fly_height = fly_height  # flight height [m] - part of the foil above the free surface (z<0) is cut off, 0 = no cut
        # Discretization parameters
        self.N         = n          # number of spanwise pannels before an eventual symmetry
        self.M         = m          # number of chordwise pannels
        self.spacing   = space      # True for a cosine spacing at the tips

        # Simulation parameters
        self.boundary  = boundary   # True create a mirror image to model the free surface on the x-y plan
        self.aoa       = alpha      # geometric angle of attack [rad]   -> axe = -y
        self.drift     = beta       # geometric angle of drift  [rad]   -> axe = -z
        self.tip_shed  = shedding   # enable the tip shedding - both -> both tips ; right -> only right tip ; left -> only left tip ; none -> both disable
                                    # left and right are defined for an horizontal wing, for a vertical one left <-> top and right <-> bottom
        self.ventilate = vent       # True enables the ventilation on the surface
        
        # Mesh
        self.wing        = {"real":None}    # wing - and mirrored wing - corner points
        self.wing_panels = {"real":None}    # wing - and mirrored wing - panels
        self.edges       = {}               # trailing edge, left tip and right tip
        self.orientation = None             # orientation of the normals / 1 for upward -1 for downward
        self.wake        = {"real":{"middle":np.array([])},"mirror":{}}    # wake - and mirrored wake - corner points, divided in middle, left and right
        self.wake_panels = {"real":[], "mirror":[]}                        # wake - and mirrored wake - panels, divided in middle, left and right
        self.normals     = None             # normal vector of each panel

        # vortex strengths
        self.gamma       = np.array([])
        self.gamma_wake  = {"middle":np.array([]), "right":np.array([]), "left":np.array([])}

        # secondary computations
        self.loads       = None             # 3D loads 
        self.Cl_2d       = None             # normal lift coef at each section
        self.area        = None             # total area of the surface

        # ventilation 
        self.ventilation = {}               # ventilation output : a0, alpha_eff, sigma_c, Lc/c 
        self.max_depth   = None

        self._build_wing()

    #-------------#
    #   Methods   #
    #-------------#

    def _reset(self):
        """
        Reset the surface to its state just after _build_wing, before any use in VLMSolver
        """
        self.wake        = {"real":{"middle":np.array([])},"mirror":{}}    # wake - and mirrored wake - corner points, divided in middle, left and right
        self.wake_panels = {"real":[], "mirror":[]}                        # wake - and mirrored wake - panels, divided in middle, left and right
        self.normals     = None             # normal vector of each panel

        # vortex strengths
        self.gamma = np.empty(0)
        self.gamma_wake = {"middle": np.empty(0),  "right":  np.empty(0),  "left":   np.empty(0)}

        # secondary computations
        self.loads       = None             # 3D loads 
        self.Cl_2d       = None             # normal lift coef at each section
        self.Cl_old      = None 
        self.ventilation = {}

        self.wake = {
            "real": {
                "middle": np.empty((0, 3)),
                "left":   np.empty((0, 3)),
                "right":  np.empty((0, 3))
            },
            "mirror": {}
        }
        self.wake_panels = {"real": [], "mirror": []}
        

        


    def _set_normals(self, thetas):
        """
        Gets the normal vectors taking into account the correction of the ventilation algorithm.

        Input:
            thetas -> rotation angles - shape (N+1,) / if thetas isn't an array no correction (thetas = 0)
        """
        m, n = self.M, self.N
        normals = np.zeros((m*n, 3))
        for k,panel in enumerate(self.wing_panels["real"]):
            normals[k] = panel.normal
        normals_rot = copy.copy(normals)    
        if isinstance(thetas, np.ndarray):
            theta  = np.tile(thetas, m)     # broadcast thetas to match the number of panels
            x = np.array([1, 0, 0])
            K = np.cross(x, normals)
            norm_K = np.linalg.norm(K, axis=1, keepdims=True)
            K = np.where(norm_K > 1e-8, K / norm_K, np.array([[0.0, 1.0, 0.0]]))        # if the normal is aligned with x we choose to rotate around y, but it should not happen
            theta_col = np.array(theta).reshape(-1, 1)
            normals_rot = normals * np.cos(theta_col) + np.cross(K, normals) * np.sin(theta_col)
        self.normals = normals_rot
        
    def _unit_rot(self, v, theta, ax):
        """
        Elemental clockwise rotation of the reference system axes.
        
        Input:
            v     -> vector componets - shape: (N, 3)
            theta -> rotation angle - clockwise rotation: theta>0
            ax    -> rotation axis index

        Output:
            v_rot -> rotated vector components - shape: (N, 3)
        """
        # Assemble rotation matrix
        rot = np.zeros((3, 3))  # rotation matrix, v[0
        # diagonal elements
        rot[ax, ax] = 1.0
        rot[ax-1, ax-1] = np.cos(theta)
        rot[ax-2, ax-2] = np.cos(theta)
        # extra-diagonal elements
        rot[ax-2, ax-1] = -np.sin(theta)
        rot[ax-1, ax-2] = np.sin(theta)

        # Apply rotation
        v_rot = np.sum(rot[None, :, :] * v[:, None, :], axis=2)
        return v_rot

    def _symmetry(self, grid, merge, norm, p, n):
        """
        Symmetry of the geometry with respect to a plane, and merge of the two sides if needed.
        If we merge we assume that the plane pass by the origin and that the grid start from the origin

        Input:
            grid  -> grid points 
            merge -> boolean, if True the two sides are merged together, otherwise they are kept separate
            norm  -> normal vector of the symmetry plane
            p     -> point on the symmetry plane
            n     -> number of panels in spanwise direction, only needed for the merge

        Output:
            grid_sym -> symmetric grid points or the merged grid points if merge is True
        """
        S = np.eye(3) - 2*np.outer(norm, norm)/(norm @ norm)    # symmetry matrix
        grid_off = grid - p                                     # grid with the plane as origin
        grid_sym = grid_off@(S.T)                               # symmetric grid
        grid_sym = grid_sym + p                                 # symmetric grid with the original plane position
        if merge :
            m = round(np.size(grid)/(3*(n+1)))-1       # number of panels in chordwise direction
            new_grid = np.empty((0, 3))
            for i in range(m+1):
                new_grid = np.concatenate([new_grid, grid_sym[i*(n+1):(i+1)*(n+1),:][::-1][:-1], grid[i*(n+1):(i+1)*(n+1),:]])  # merge the two sides
            grid_sym = new_grid
        return grid_sym

    def _paneling(self, p, r, w, n, b):
        """
        Build a list of panels from a grid

        Input:
            p      -> panels corner points
            r      -> ring corner points - same than panel if it's a wake
            w      -> 0 for wing - 1 for wake
            n      -> number of panels in the spanwise direction
            b      -> wing span
            
        Output:
            panels -> list of panels
        """
        m = round(np.size(p)/(3*(n+1)))-1       # number of panels in chordwise direction
        panels = []    
        for i in range(m):
            for j in range(n):
                pnt = [p[i*(1+n)+j],
                    p[i*(n+1)+j+1],
                    p[(i+1)*(n+1)+j+1],
                    p[(i+1)*(n+1)+j]]    # ordered panel vertices    
                vtx = [r[i*(1+n)+j],
                    r[i*(n+1)+j+1],
                    r[(i+1)*(n+1)+j+1],
                    r[(i+1)*(n+1)+j]]    # ordered ring vertices
                panels.append(VLMPanel(pnt, vtx, w, b))
        return panels

    def _build_wing(self):
        """
        Build the wing panels and rings, will create self.wing and self.wing_panels
        """
        m = self.M
        n = self.N
        b     = self.span
        alpha = self.aoa
        beta  = self.drift
        lamb  = self.sweep
        delta = self.dihedron
        c     = self.chord
        org   = self.origin

        # Leading-edge length of the semi-wing
        le = (0.5 * b)
        s_min, s_max = 0.0, le

        # Spanwise wing discretization, taking the symmetry and spacing parameters into account
        if self.spacing :
            if self.symmetric :                                 # if it's symmetric, we cluster only at the wing tip, otherwise we cluster at both the root and the tip
                theta = np.linspace(np.pi/2, np.pi, (n+1))
                s = le*(-np.cos(theta))                         # leading-edge coordinate 
            else :
                theta = np.linspace(0, np.pi, (n+1))
                s = le*(1-np.cos(theta))/2                      # leading-edge coordinate
        else :
            s = np.linspace(s_min, s_max, (n+1))                # leading-edge coordinate

        y, z = self._generate_foil(s, self.root_ang, self.tip_ang, self.radius, self.center )
        y = np.tile(y,m+1)                                      # grid points y components
        z = np.tile(z,m+1)                                      # grid points z components

        # Chordwise wing panels discretiration
        i = np.arange(m+1)[:, None]
        j = c[None, :]
        ch = (i/m) * j                                          # chordwise discretization at each spanwise station
        one = np.ones_like(i)
        ch = ch - one*(j/2) + c[0]/2                            # grid points x components, the chord law is centered  
        ch = ch.reshape(-1)
        p_wing = np.column_stack((ch, y, z)) 

        # apply the geometrical transformations
        p_swept = np.column_stack((p_wing[:,2]*np.tan(lamb)+p_wing[:,0],p_wing[:,1],p_wing[:,2]))     # apply sweep
        p_yaw   = self._unit_rot(p_swept, -delta, 0)                                                        # apply dihedron
        
        # take care of the symmetry
        if self.symmetric :
            p_yaw  = self._symmetry(p_yaw, merge=True, norm=np.array([0,0,1]), p=np.array([0,0,0]), n=n)
            self.N = 2*n
            n      = self.N           # double n for the rest of the code

        self.wing["real"] = p_yaw
        self._apply_twist()
        p_twist = copy.copy(self.wing["real"])

        # build the rings
        self._build_ring()
        r_yaw = self.wing["real"]

        # change of reference plan
        p_geo = self._change_plan(p_twist)      # fully buildt geometrical wing panel corner points
        r_geo = self._change_plan(r_yaw)        # fully buildt geometrical wing ring corner points

        # translate to the origin
        p_geo = p_geo + org
        r_geo = r_geo + org

        # cut the emerged part above the free surface (flight height) before the
        # global rotations - remaining stations keep the reference mesh size
        p_geo, r_geo, self.chord, self.span, n = self._truncate_emerged(p_geo, r_geo, m, n)
        self.N = n
        b = self.span

        # global rotations
        p_attac = self._unit_rot(p_geo,  -alpha, 1)   # incidence rotation                                  
        p_earth = self._unit_rot(p_attac,  -beta, 2)  # drift rotation
        r_attac = self._unit_rot(r_geo,  -alpha, 1)                                                   
        r_earth = self._unit_rot(r_attac,  -beta, 2)

        self.wing_panels["real"] = self._paneling(p_earth, r_earth, 0, n, b)
        self.wing["real"]        = r_earth
        self.edges["middle"] = r_earth[-(n+1):]                                    # get the trailing edge for the shedding
        self.edges["left"]   = r_earth.reshape(m+1, n+1, 3)[:,0,:].reshape(m+1,3)  # get the left tip
        self.edges["right"]  = r_earth.reshape(m+1, n+1, 3)[:,n,:].reshape(m+1,3)  # get the left tip

        # max depth
        ctrl     = np.array([p.ctr for p in self.wing_panels["real"]])
        z        = np.sum(ctrl.reshape((m,n,3)), axis=0)[:,2]/m  
        self.max_depth = np.max(z)

        # boundary condition
        if self.boundary=="symmetric" or self.boundary=="antisymmetric" :
            p_mir = self._symmetry(p_earth, False, np.array([0,0,1]),np.array([0,0,0]),n)
            r_mir = self._symmetry(r_earth, False, np.array([0,0,1]),np.array([0,0,0]),n)
            self.wing_panels["mirror"] = self._paneling(p_mir, r_mir, 0, n, b)
            self.wing["mirror"]        = r_mir

    def _truncate_emerged(self, p_in, r_in, m, n):
        """
        Cut the part of the wing above the free surface (z < 0, z positive
        downwards), applied on the translated geometry before the global
        rotations. The immersed stations are kept as-is (no resampling), so N
        decreases with fly_height while the panel size stays the one of the
        fly_height=0 reference - the first panel is partial (cut through it),
        or absorbs the next station if the cut leaves less than 81% of a
        reference panel (sliver merge, to avoid degenerate panels).
        No-op when fly_height <= 0. The downstream shape (elbow, tip, twist)
        is preserved exactly since the cut is applied on the built geometry.

        Input:
            p_in, r_in -> panel and ring corner grids after translation to the
                          origin, shape ((m+1)*(n+1), 3), root (j=0) to tip (j=n)
            m, n       -> chordwise / spanwise panel counts

        Output:
            p_cut, r_cut -> truncated grids, shape ((m+1)*(n_new+1), 3)
            chord_new    -> chord law of the remaining stations, shape (n_new+1,)
            span_new     -> immersed span (2 x immersed leading-edge length)
            n_new        -> remaining spanwise panel count (<= n)
        """
        h = self.fly_height
        if h is None or h <= 0:
            return p_in, r_in, self.chord, self.span, n
        if self.symmetric:
            raise ValueError("fly_height truncation is not supported with symmetric=True (root merged at the middle).")
        p_grid = p_in.reshape(m+1, n+1, 3)
        # shallowest corner of each station : the new waterline row is fully
        # immersed, so no panel corner ends up above water at this stage
        z_stat = p_grid[:, :, 2].min(axis=0)      # depth of each station
        if np.all(z_stat < 0):
            raise ValueError(f"fly_height={h} m fully emerges the wing (deepest station at z={z_stat.max():.4f} m).")
        if z_stat[0] >= 0:
            return p_in, r_in, self.chord, self.span, n  # already fully immersed
        j0 = int(np.argmax(z_stat >= 0))          # first immersed station
        denom = z_stat[j0] - z_stat[j0-1]
        f = (0.0 - z_stat[j0-1]) / (denom if abs(denom) > 1e-12 else 1e-12)
        f = float(np.clip(f, 0.0, 1.0))
        if (1.0 - f) < 0.81 and j0 < n:
            # the waterline panel would be a sliver (degenerate influence
            # coefficients) : absorb station j0, merging it with the first
            # fully immersed panel instead of cutting
            j1 = j0 + 1
        else:
            j1 = j0
        r_grid = r_in.reshape(m+1, n+1, 3)
        # cut grids : interpolated waterline row + immersed stations j1..n (not resampled)
        p_cut = np.concatenate([((1-f)*p_grid[:, j0-1, :] + f*p_grid[:, j0, :])[:, None, :], p_grid[:, j1:, :]], axis=1)
        r_cut = np.concatenate([((1-f)*r_grid[:, j0-1, :] + f*r_grid[:, j0, :])[:, None, :], r_grid[:, j1:, :]], axis=1)
        n_new = p_cut.shape[1] - 1                # remaining panels (<= n)
        c = np.asarray(self.chord)
        chord_new = np.concatenate([np.array([(1-f)*c[j0-1] + f*c[j0]]), c[j1:]])
        span_new = 2*float(np.sum(np.linalg.norm(np.diff(p_cut[0], axis=0), axis=1)))
        return p_cut.reshape(-1, 3), r_cut.reshape(-1, 3), chord_new, span_new, n_new

    def _generate_foil(self, s, root_angle_deg, tip_angle_deg, radius, curve_pos):
        """
        Generates the Y and Z coordinates of a parametric foil (C, J, L).
        Assumes Z is positive downwards. 
        Strictly preserves the curve length (span).
        
        Parameters:
            s              : 1D array of the curvilinear coordinate (e.g., from 0 to s_max).
            root_angle_deg : Entry angle at the hull (0° = perfectly vertical downwards, 90° = horizontal).
            tip_angle_deg  : Exit angle at the tip (0° = vertical downwards, 90° = perfectly horizontal).
            radius         : Bend radius of the curve in meters.
            curve_pos      : Position of the curve on the remaining straight length (0.0 to 1.0).
                            - 1.0 = L-Foil (curve pushed maximally towards the tip)
                            - 0.5 = J-Foil (curve right in the middle)
                            - 0.0 = Curve immediately starts at the hull
        Returns:
            y, z : Arrays of the 2D coordinates.
        """
        d_root = np.deg2rad(root_angle_deg)
        d_tip  = np.deg2rad(tip_angle_deg)
        
        # 1. Length of the circular arc
        L_arc = radius * abs(d_tip - d_root)
        S_total = s[-1]
        
        # Safety check: the radius cannot be so large that the arc exceeds the total span
        if L_arc > S_total:
            raise ValueError(f"Radius too large! Arc length ({L_arc:.2f}m) exceeds the span ({S_total:.2f}m).")
            
        # 2. Distribution of the remaining straight length
        L_rem = S_total - L_arc
        L_shaft = L_rem * curve_pos        # Length of the upper straight section (shaft)
        L_tip_straight = L_rem - L_shaft   # Length of the lower straight section (tip)

        # Rate of angle change along the arc (defined upfront : no discrete
        # station may fall inside a small arc, but k is still needed below)
        k = (d_tip - d_root) / L_arc if L_arc > 0 else 0.0
        
        # Initialize coordinates
        y = np.zeros_like(s)
        z = np.zeros_like(s)
        
        # --- MASKS FOR THE 3 SECTIONS ---
        m_shaft = s <= L_shaft
        m_arc   = (s > L_shaft) & (s <= L_shaft + L_arc)
        m_tip   = s > L_shaft + L_arc
        
        # --- SECTION 1: The shaft ---
        y[m_shaft] = s[m_shaft] * np.sin(d_root)
        z[m_shaft] = s[m_shaft] * np.cos(d_root)
        
        # Connection points (end of the shaft)
        y_shaft_end = L_shaft * np.sin(d_root)
        z_shaft_end = L_shaft * np.cos(d_root)
        
        # --- SECTION 2: The circular arc (elbow) ---
        if np.any(m_arc):
            s_arc = s[m_arc] - L_shaft
            
            # Perfect analytical integration for the arc to guarantee exact curvilinear length
            delta_s = d_root + k * s_arc
            y[m_arc] = y_shaft_end - (np.cos(delta_s) - np.cos(d_root)) / k
            z[m_arc] = z_shaft_end + (np.sin(delta_s) - np.sin(d_root)) / k
            
        # Connection points (end of the arc)
        if L_arc > 0:
            y_arc_end = y_shaft_end - (np.cos(d_tip) - np.cos(d_root)) / k
            z_arc_end = z_shaft_end + (np.sin(d_tip) - np.sin(d_root)) / k
        else:
            y_arc_end, z_arc_end = y_shaft_end, z_shaft_end

        # --- SECTION 3: The tip ---
        if np.any(m_tip):
            s_tip = s[m_tip] - (L_shaft + L_arc)
            y[m_tip] = y_arc_end + s_tip * np.sin(d_tip)
            z[m_tip] = z_arc_end + s_tip * np.cos(d_tip)
            
        return y, z

    def _apply_twist(self):
        """
        Add the twist to the wing
        If phi is a number it apply the twist linearly to have a twist phi at the tip (classic wing construction)
        If phi is an array we assumed it's the value wanted at each spanwise section, we also assume it is well broadcasted
        """
        m    = self.M
        n    = self.N
        phi  = self.twist
        grid = self.wing["real"]
        if isinstance(phi, np.ndarray) :        # phi is already discretized at each station
            phi_j = phi                                                         # spanwise twist distribution
        else :
            phi_j = np.abs(grid[:,2])*phi/grid[-1,2]                                                   # spanwise twist distribution
        p_middle = np.zeros_like(grid[:n+1])
        for j in range(n+1):    
            p_middle[j] = (grid[j] + grid[j+m*(n+1)])/2                        # getting the middle chord point of each section
        p_middle = np.tile(p_middle, (m+1,1))                      # broadcasting, each j section got its middle chord position
        grid_off = grid - p_middle                                 # grid where each j section is in the local frame with the middle chord point as x origin
        idx = np.arange((m+1)*(n+1)).reshape(m+1, n+1)
        for j in range(n+1):    
            grid_off[idx[:,j]] = self._unit_rot(grid_off[idx[:,j]], -phi_j[j], 2)                         # twisted frame for each section
        self.wing["real"] = grid_off + p_middle                                                     # twisted wing corner points

    def _build_ring(self):
        """ 
        Build the ring starting from the panels geometry
        self.wing is now the ring corner points
        """
        r_wing = self.wing["real"]
        m      = self.M
        n      = self.N
        for j in range(n+1):
            for i in range(m):
                r_wing[i*(n+1)+j] = r_wing[i*(n+1)+j] + (r_wing[(i+1)*(n+1)+j]-r_wing[i*(n+1)+j]) * 0.25        # one quarter chord offset
            r_wing[m*(n+1)+j] = r_wing[m*(n+1)+j] + (r_wing[m*(n+1)+j]-r_wing[(m-1)*(n+1)+j])/3
        self.wing["real"] = r_wing
        
    def _change_plan(self, grid):
        """ 
        Change the reference plan (only vertical z-x to horizontal y-x for now)

        Input:
        -> grid: grid that you want to rebase

        Output:
        -> new_grid: the grid based in the good reference plan
        """
        if self.plan==1:
            new_grid = np.column_stack([grid[:,0], -grid[:,2], grid[:,1]])   # z-x plan become y-x plan
            return new_grid
        return grid

    def _update_wake(self,wake):
        """ 
        Update the wake geometry, usefull to avoid this lines in _time_sim

        Input:
            wake -> the new wake - in a list [mid_wake, left_wake, right_wake]
        """
        self.wake["real"]["middle"] = wake[0]
        self.wake["real"]["left"]   = wake[1]
        self.wake["real"]["right"]  = wake[2]
        if self.boundary=="symmetric" or self.boundary=="antisymmetric" :                             # updating the mirrored wake
            self.wake["mirror"]["middle"] = self._symmetry(wake[0], False, np.array([0,0,1]), np.array([0,0,0]), 0)
            self.wake["mirror"]["left"]   = self._symmetry(wake[1], False, np.array([0,0,1]), np.array([0,0,0]), 0)
            self.wake["mirror"]["right"]  = self._symmetry(wake[2], False, np.array([0,0,1]), np.array([0,0,0]), 0)
        
    def _ventilation_coef(self, aoa, u):
        """
        Compute the correction coefficent which model the ventilation

        Input :
            aoa -> effective angle of attack
            u   -> velocity of the flow
        """
        m, n     = self.M, self.N
        h        = self.max_depth
        aoa_safe = np.maximum(aoa, 0.001)
        ctrl     = np.array([p.ctr for p in self.wing_panels["real"]])
        z        = np.sum(ctrl.reshape((m,n,3)), axis=0)[:,2]/m      # mean depth of each spanwise station
        froude   = u/np.sqrt(9.81*h)
        sigma_c  = 2*z/((froude**2)*h)                          # cavitation number at each spanwise station
        psi      = sigma_c/(2*aoa)
        Lc       = (2.67*psi + 96.62)/(psi**3 - 7.1*psi**2 + 49.42*psi + 0.961)      # cavity lentgh divided by the chord lentgh
        a0       = (np.pi/2*Lc**3 - 2*Lc**2 +4.5*Lc + 1)/(Lc**3 - Lc**2 + 0.75*Lc + 1/(2*np.pi))
        self.ventilation["a0"]        = a0
        self.ventilation["alpha_eff"] = aoa
        self.ventilation["sigma_c"]   = sigma_c
        self.ventilation["Lc/c"]      = Lc
        self.ventilation["froude"]    = froude

