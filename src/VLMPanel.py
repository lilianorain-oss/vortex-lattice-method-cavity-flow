## Packages Import
import numpy as np

## Class Definition
class VLMPanel:
    """
    Vortex Lattice Method panel object with ring vortex
    """
    #--------------------#
    #   Initialization   #
    #--------------------#
    def __init__(self, p, b, s_type):
        """
        Panel is of quadrilateral shape and vertices must be ordered from root
        position on leading-edge proceeding in counterclockwise direction.
        Ring vortex is placed on top of the panel; horseshoe vortex is approximated
        by a ring vortex whose starting vortex extends far back.
        """
        # Wing geometry
        self.span = b       # wing span [m]

        # Panel geometry
        self.pnt    = p        # panel vertices
        self.ctr    = None     # control point
        self.normal = None     # normal direction
        self.chord  = None     # chord length [m]
        self.width  = None     # pannel width [m]
        self.area   = None     # panel area   [m**2]

        # Vortex parameters
        self.sing = s_type      # type of singularity element
        self.vrt  = None        # ring vortex points
        
        self._get_panel_geom()
        self._get_vortex_geom(20*b)


    #-------------#
    #   Methods   #
    #-------------#
    def _get_panel_geom(self):
        """
        Compute panel geometric parameters such as control point position, normal
        versor, chord length, width and area, starting from panel's vertices.
        """
        p = self.pnt

        # Compute representative panel's geometric parameters
        c_avg = 0.5 * (np.linalg.norm(p[2] - p[1]) + np.linalg.norm(p[3] - p[0]))  # average chord length
        w_avg = 0.5 * (np.linalg.norm(p[2] - p[3]) + np.linalg.norm(p[1] - p[0]))  # average width

        # Compute position of control point
        cp = (p[0] + p[1] + 3*p[3] + 3*p[2]) / 8    # three-quarter line

        # Compute normal versor and panel surface
        ai = np.zeros(3)
        for i, pi in enumerate(p):
            qi = p[(i + 1) % len(p)]
            ai += np.cross(pi, qi)

        av = 0.5 * ai
        a = np.linalg.norm(av)
        if a > 0.0:
            n = av / a
        else:
            # Degenerate cells with zero area
            print(f"Degenerate panel {i} with {len(p)} vertices")

        self.ctr    = cp
        self.normal = n
        self.chord  = c_avg
        self.width  = w_avg
        self.area   = a

    def _get_vortex_geom(self, d):
        """ 
        Compute ring vortex's vertices position starting from panel's vertices: the
        leading segment is placed on the panel's quarter chord line. Horseshoe vortex
        is approximated by a ring vortex with negligible contribution by the wake
        vortex.

        Input:
        d -> wake vortex distance from wing
        """
        p = self.pnt

        v = []
        for i, pi in enumerate(p):
            vi = np.copy(pi)

            if self.sing == "horseshoe":
                # Horseshoe vortex approximation
                if i in [2, 3]:
                    # Wake vortex lies far behind the wing
                    vi[0] += d   # HP: trailing vortices parallel to x-axis
                    v.append(vi)
                    continue

            # Compute offset based on position of point
            if i in [0, 3]:
                offset = 0.25 * (p[3] - p[0])   # panel's root
            elif i in [1, 2]:
                offset = 0.25 * (p[2] - p[1])   # panel's tip
            # Apply offset to panel's vertices
            vi += offset
            v.append(vi)

        self.vrt = v