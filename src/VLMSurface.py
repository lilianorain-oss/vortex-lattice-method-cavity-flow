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
    def __init__(self, origin, plan, boundary, shedding, b, c, alpha, beta, lamb, delta, phi, sym, space, n, m):
        
        # Wing geometry
        self.origin    = origin     # starting point of the wing - will end at the middle if it's a symmetric wing
        self.plan      = plan       # plan in wich the wing is based - 0 for z-x, 1 for y-x
        self.span      = b          # wing span
        self.chord     = c          # chord law distribution
        self.sweep     = lamb       # middle chord sweep angle  [rad]
        self.dihedron  = delta      # dihedral angle            [rad]
        self.twist     = phi        # quarter chord twist angle [rad]
        self.symmetric = sym        # True for a symmetric wing

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
        # Mesh
        self.wing        = {"real":None}    # wing - and mirrored wing - corner points
        self.wing_panels = {"real":None}    # wing - and mirrored wing - panels
        self.edges       = {}               # trailing edge, left tip and right tip
        self.wake        = {"real":{"middle":np.array([])},"mirror":{}}    # wake - and mirrored wake - corner points, divided in middle, left and right
        self.wake_panels = {"real":[], "mirror":[]}                        # wake - and mirrored wake - panels, divided in middle, left and right

        # vortex strengths
        self.gamma       = np.array([])
        self.gamma_wake  = {"middle":np.array([]), "right":np.array([]), "left":np.array([])}

        # secondary computations
        self.loads       = None             # 3D loads 
        self.Cl_2d       = None             # lift coef at each section

    #-------------#
    #   Methods   #
    #-------------#

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

    def _paneling(self,p, r, w, n, b):
        """
        Build a list of panels from a grid

        Input:
            p      -> panels corner points
            r      -> ring corner points - same than panel if it's a wake
            w      -> 0 for wing - 1 for wake
            n      -> number of panels in the spanwise direction            
        Output:
            panels -> list of panels
        """
        b = self.span
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
        s = np.tile(s,m+1)                                      # grid points z components

        # Chordwise wing panels discretiration
        i = np.arange(m+1)[:, None]
        j = c[None, :]
        ch = (i/m) * j                                          # chordwise discretization at each spanwise station
        one = np.ones_like(i)
        ch = ch - one*(j/2) + c[0]/2                            # grid points x components, the chord law is centered  
        ch = ch.reshape(-1)
        p_wing = np.column_stack((ch, np.zeros_like(s), s)) 

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

        # boundary condition
        if self.boundary:
            p_mir = self._symmetry(p_earth, False, np.array([0,0,1]),np.array([0,0,0]),n)
            r_mir = self._symmetry(r_earth, False, np.array([0,0,1]),np.array([0,0,0]),n)
            self.wing_panels["mirror"] = self._paneling(p_mir, r_mir, 0, n, b)
            self.wing["mirror"]        = r_mir


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
        if self.boundary :                             # updating the mirrored wake
            self.wake["mirror"]["middle"] = self._symmetry(wake[0], False, np.array([0,0,1]), np.array([0,0,0]), 0)
            self.wake["mirror"]["left"]   = self._symmetry(wake[1], False, np.array([0,0,1]), np.array([0,0,0]), 0)
            self.wake["mirror"]["right"]  = self._symmetry(wake[2], False, np.array([0,0,1]), np.array([0,0,0]), 0)

