## Packages Import
import numpy as np

from VLMPanel import VLMPanel

## Class Definition
class VLMGrid:
    """
    Vortex Lattice Method grid object
    """
    #--------------------#
    #   Initialization   #
    #--------------------#
    def __init__(self, b, c, theta, gamma, lamb, sym,
                s_type, u_inf, n):
        """
        Wing is discretized into a number of panel elements both in spanwise and
        chordwise direction.
        """
        # Wing geometry
        self.span     = b       # wing span                 [m]
        self.chord    = c       # chord lengths             [m]
        self.incid    = theta   # angle of incidence        [rad]
        self.dihedron = gamma   # dihedral angle            [rad]
        self.sweep    = lamb    # quarter-chord sweep angle [rad]
        self.symmetric = sym    # symmetric wing

        # Vortex parameters
        self.sing = s_type      # type of singularity element

        # Flow properties
        self.U = u_inf      # inflow velocity [m/s]

        # Discretization
        self.N = n              # number of panels
        self.panels = None      # panel objects

        # Linear system

    
    #-------------#
    #   Methods   #
    #-------------#
    def _unit_rot(self, v, theta, ax, inverse=False):
        """
        Elemental clockwise rotation of the reference system axes.
        
        Input:
        v     -> vector componets - shape: (N, 3)
        theta -> rotation angle - clockwise rotation: theta>0
        ax    -> rotation axis index
        """
        # Assemble rotation matrix
        rot = np.zeros((3, 3))  # rotation matrix
        # diagonal elements
        rot[ax, ax] = 1.0
        rot[ax-1, ax-1] = np.cos(theta)
        rot[ax-2, ax-2] = np.cos(theta)
        # extra-diagonal elements
        rot[ax-2, ax-1] = -np.sin(theta)
        rot[ax-1, ax-2] = np.sin(theta)

        if inverse:
            # Apply inverse rotation
            rot = rot.T

        # Apply rotation
        v_rot = np.sum(rot[None, :, :] * v[:, None, :], axis=2)
        return v_rot


    def _build_mesh(self):
        """
        Discretization of the wing by a finite number of lattices.
        """
        n      = self.N
        b      = self.span
        theta  = self.incid
        lamb   = self.sweep
        gamma  = self.dihedron
        c      = self.chord
        s_type = self.sing

        # Leading-edge length of the semi-wing
        le = (0.5 * b) / (np.cos(lamb) * np.cos(gamma))
        if self.symmetric:
            s_min, s_max = 0.0, le
        else:
            s_min, s_max = -le, le

        # Spanwise wing discretization
        s = np.linspace(s_min, s_max, (n+1))    # leading-edge coordinates

        # Change reference system for leading-edge coordinates
        le_wing = np.column_stack((np.zeros_like(s), np.zeros_like(s), s))      # wing frame
        le_i  = self._unit_rot(le_wing, (-theta * np.cos(lamb)), 2, True)   # incidence rotation (about spanwise axis)
        le_ii = self._unit_rot(le_i, (gamma * np.cos(lamb)), 0, True)       # dihedral rotation (about chordwise axis)
        le_body = self._unit_rot(le_ii, -lamb, 1, True)                         # body frame
        
        # Translate leading-edge coordinates into trailing-edge points
        c_wing = np.column_stack((c, np.zeros_like(c), np.zeros_like(c)))       # wing frame
        c_i  = self._unit_rot(c_wing, (-theta * np.cos(lamb)), 2, True)     # incidence rotation (about spanwise axis)
        c_ii = self._unit_rot(c_i, (gamma * np.cos(lamb)), 0, True)         # dihedral rotation (about chordwise axis)
        c_body = self._unit_rot(c_ii, -lamb, 1, True)                           # body frame
        # Rescale coordinates with projected points on horizontal plane to ensure chord length norm
        k = c / np.linalg.norm(c_body[:, :2], axis=1)   # scale factor
        c_body *= k[:, None]
        te_body = le_body + np.column_stack((c_body[:, :2], np.zeros_like(c)))  # rigid translation of leading-edge coordinates
    
        panels = []     # list of panel objects
        for i in range(n):
            pnt = [le_body[i],
                le_body[i+1],
                te_body[i+1],
                te_body[i]]    # ordered panel vertices

            panels.append(VLMPanel(p=pnt, b=b, s_type=s_type))

        self.panels = panels