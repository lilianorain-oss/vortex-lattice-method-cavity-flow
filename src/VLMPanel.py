import numpy as np

class VLMPanel:
    """
    Vortex Lattice Method panel with ring vortex
    """
    #--------------------#
    #   Initialization   #
    #--------------------#
    def __init__(self, p, v, w, b):
        """
        Panel is of quadrilateral shape and vertices must be ordered from root
        position on leading-edge proceeding in counterclockwise direction.
        The pannel can be a wing pannel (with an offset ring) or a wake pannel (with the ring being the pannel)
        """
        # Wing geometry
        self.span = b       # wing span [m]

        # nature of the pannel (wing or wake)
        self.nature = w        # 0 for wing, 1 for wake

        # Panel geometry
        self.pnt    = p        # panel vertices
        self.ctr    = None     # control point
        self.normal = None     # normal direction
        self.chord  = None     # chord length [m]
        self.width  = None     # pannel width [m]
        self.area   = None     # panel area   [m**2]
        
        # Vortex parameters
        self.vrt = v            # ring vortex points

        if self.nature == 0 :
            self._get_panel_geom()
      
    #-------------#
    #   Methods   #
    #-------------#
    def _get_panel_geom(self):
        """
        Compute panel geometric parameters such as control point position, normal
        versor, chord length, width and area, starting from panel's vertices.
        """
        p = self.pnt
        v = self.vrt
        # Compute representative panel's geometric parameters
        c_avg = 0.5 * (p[2] - p[1] + p[3] - p[0])  # average chord vector
        w_avg = 0.5 * (p[2] - p[3] + p[1] - p[0])  # average span vector
        
        # Compute position of control point
        cp = (p[0] + p[1] + 3*p[3] + 3*p[2])/8    # three-quarter line

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
            #Degenerate cells with zero area
            print(f"Degenerate panel {i} with {len(p)} vertices")


        self.ctr    = cp
        self.normal = n
        self.chord  = c_avg
        self.width  = w_avg
        self.area   = a
