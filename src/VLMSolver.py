import numpy as np
import copy as copy
import numba as nb


class VLMSolver:
    """
    Vortex Lattice Method solver
    """
    #--------------------#
    #   Initialization   #
    #--------------------#
    def __init__(self, surfaces, u_inf, boundary):

        # Flow properties
        self.U        = u_inf      # inflow velocity [m/s]
        self.boundary = boundary   # enable the free surface condition

        # surfaces
        self.surfaces = surfaces   # list of the VLMSurface

        # global parameters
        self.N_tot    = None       # total number of panels 
        self.controls = None       # control points of all the real wings
        self.normals  = None       # normals of all the real wings

        # Linear system
        self.A = None       # influence coefficient matrix
        self.b = None       # right hand side
    
        self.time=0

        self._get_parameters()

    #-------------#
    #   Methods   #
    #-------------#
 
    def _get_parameters(self):
        """ 
        get the global parameters from each surface
        """
        ctrl  = []
        norm  = []
        N_tot = 0
        for surface in self.surfaces :
            N_tot += surface.N * surface.M
            for panel in surface.wing_panels["real"] :
                ctrl.append(panel.ctr)     # we compute the normals and control points vector only once
                norm.append(panel.normal)
        self.N_tot    = N_tot
        self.controls = np.array(ctrl)
        self.normals  = np.array(norm)

    @staticmethod
    @nb.njit(parallel=True, fastmath=True)
    def _induced_velocity(p, c1, c2, gamma):
        N = p.shape[0]
        M = c1.shape[0]
        v_total = np.zeros((N, 3), dtype=np.float64)
        eps = 1e-4

        # Parallélisation sur les points d'évaluation
        for i in nb.prange(N):
            px, py, pz = p[i, 0], p[i, 1], p[i, 2]
            
            for j in range(M):
                # Vecteurs r1 et r2
                rx1 = px - c1[j, 0]
                ry1 = py - c1[j, 1]
                rz1 = pz - c1[j, 2]
                
                rx2 = px - c2[j, 0]
                ry2 = py - c2[j, 1]
                rz2 = pz - c2[j, 2]
                
                # Vecteur r0 (c2 - c1)
                r0x = c2[j, 0] - c1[j, 0]
                r0y = c2[j, 1] - c1[j, 1]
                r0z = c2[j, 2] - c1[j, 2]
                
                # Produit vectoriel r1 x r2
                cx = ry1 * rz2 - rz1 * ry2
                cy = rz1 * rx2 - rx1 * rz2
                cz = rx1 * ry2 - ry1 * rx2
                
                cross_norm2 = cx*cx + cy*cy + cz*cz
                r1_norm = np.sqrt(rx1*rx1 + ry1*ry1 + rz1*rz1)
                r2_norm = np.sqrt(rx2*rx2 + ry2*ry2 + rz2*rz2)
                
                # Vérification de la singularité (mask)
                if r1_norm > eps and r2_norm > eps and cross_norm2 > eps**4:
                    # Calcul du terme scalaire
                    dot_term = r0x * (rx1/r1_norm - rx2/r2_norm) + \
                            r0y * (ry1/r1_norm - ry2/r2_norm) + \
                            r0z * (rz1/r1_norm - rz2/r2_norm)
                    
                    # Assemblage final
                    coeff = gamma[j] / (12.566370614359172) # 4 * pi
                    factor = coeff * dot_term / cross_norm2
                    
                    v_total[i, 0] += cx * factor
                    v_total[i, 1] += cy * factor
                    v_total[i, 2] += cz * factor
                    
        return v_total
    
    def _induced_velocityp (self,p,c1,c2,gamma):
        """
        Return the velocity induced by the vortex segments [c1, c2], of strentgh gamma,
        at the points p

        Input :
        p     -> points where the velocity is computed, shape (N,3)
        c1    -> starting points of the segments, shape (M,3)
        c2    -> ending points of the segments, shape (M,3)
        gamma -> circulation of each segment, shape (M,)

        Output :
        v     -> velocity induced at each point by each segment, shape (N,3)
        """
        r1 = p[:, None, :] - c1[None, :, :]    # (N,M,3)
        r2 = p[:, None, :] - c2[None, :, :]    # (N,M,3)
        r0 = c2[None, :, :] - c1[None, :, :]   # (1,M,3)
        r1_norm = np.linalg.norm(r1, axis=2)   # (N,M)
        r2_norm = np.linalg.norm(r2, axis=2)
        cross = np.cross(r1, r2)                # (N,M,3)
        cross_norm2 = np.sum(cross**2, axis=2)  # (N,M)
        # The safe are needed to avoid division by zero, the value of 1.0 is arbitrary since we will set the velocity to zero in these case
        eps = 1e-5
        r2_norm_safe = np.where(r2_norm < eps, 1.0, r2_norm)    
        r1_norm_safe = np.where(r1_norm < eps, 1.0, r1_norm)    
        cross_norm2_safe = np.where(cross_norm2 < eps, 1.0, cross_norm2)
        mask = ((r1_norm > eps) &(r2_norm > eps) &(cross_norm2 > eps))                               # points on or aligned with the segment
        term = np.sum(r0 * (r1 / r1_norm_safe[:, :, None] - r2 / r2_norm_safe[:, :, None]), axis=2)  # (N,M)
        coeff = gamma[None, :] / (4 * np.pi)
        v = coeff[:, :, None] * (cross / cross_norm2_safe[:, :, None]) * term[:, :, None]
        v[~mask] = 0                    # if one point is on or aligned with the segment, its induced velocity is 0
        return np.sum(v, axis=1)  # (N,3)
    

    def _vectorize (self, grid, gamma, n) :
        """
        Transform the grid corner points into two array of points that define all
        the segments, needed for the _induced_velocity function, and the local circulation
        of each segment

        Input :
        grid  -> 2D grid of the points that defines the segments which induce the velocity
        gamma -> circulation of each vortex ring
        n     -> number of panels in spanwise direction, needed to reshape the grid and gamma in the right way

        Output :
        c1, c2    -> starting and ending points of each segment
        gamma_seg -> local circulation of each segment
        """
        m = round(np.size(grid)/(3*(n+1)))-1                    # number of panels in chordwise direction
        idx = np.arange((m+1)*(n+1)).reshape(m+1, n+1)          # reshape for building the segments points
        c1_j = grid[idx[:, :-1]]                                # spanwise segments
        c2_j = grid[idx[:, 1:]]
        c1_i = grid[idx[:-1, :]]                                # chordwise segments
        c2_i = grid[idx[1:, :]]
        c1 = np.concatenate([c1_j.reshape(-1,3), c1_i.reshape(-1,3)])
        c2 = np.concatenate([c2_j.reshape(-1,3), c2_i.reshape(-1,3)])
        gamma = gamma.reshape(m, n)
        gamma_seg_j =  gamma[1:, :] - gamma[:-1, :]                                     # local circulation at each spanwise segment except the first and last one
        gamma_seg_j_full = np.vstack([gamma[0, :], gamma_seg_j, -gamma[-1, :]])         # all the local circulations of spanwise segments
        gamma_seg_i =  -gamma[:, 1:] + gamma[:, :-1]                                    # local circulation at each chordwise segment except the first and last one        
        gamma_seg_i_full = np.hstack([-gamma[:, [0]], gamma_seg_i, gamma[:, [-1]]])     # all the local circulations of chordwise segments
        gamma_seg = np.concatenate([gamma_seg_j_full.reshape(-1), gamma_seg_i_full.reshape(-1)])    # local circulation of each segment
        return c1, c2, gamma_seg
    
    def _full_vectorize(self, wing):
        """
        Build the segments needed for using _induced_velocity from all the surfaces
        Basically do _vectorize at each surface needed and return the same type of results, directly usable for _induced_velocity

        Input :
            wing      -> True if the wing influence needs to be counted (for the local speed for instance) - False if it doesn't (for the RHS...)

        Output :
            c1, c2    -> starting and ending points of each segment
            gamma_seg -> local circulation of each segment
        """
        surfaces  = self.surfaces
        c1_tot = np.empty((0, 3))
        c2_tot = np.empty((0, 3))
        gamma_tot = np.empty(0)
        for surface in surfaces:
            for k in surface.wing.keys():
                if wing :                               # taking the wing into consideration or not
                    mid_grid  = np.concatenate([surface.wing[k][:surface.M*(surface.N+1)], surface.wake[k]["middle"]])
                    mid_gamma = np.concatenate([surface.gamma, surface.gamma_wake["middle"]])
                else :
                    mid_grid  = surface.wake[k]["middle"]
                    mid_gamma = surface.gamma_wake["middle"]
                c1, c2, gamma_v      = self._vectorize(mid_grid, mid_gamma, surface.N)

                if ( surface.tip_shed == "left" ) or ( surface.tip_shed == "both" ) :
                    c1_l, c2_l, gamma_vl = self._vectorize(surface.wake[k]["left"], surface.gamma_wake["left"], round(len(surface.wake[k]["left"])/(surface.M+1)-1))
                else :
                    c1_l, c2_l, gamma_vl = np.empty((0, 3)), np.empty((0, 3)), np.empty(0)   # in order to let the concatenation work and to be able to modifie the gamma for boundary condition
                if ( surface.tip_shed == "right" ) or ( surface.tip_shed == "both" ) :
                    c1_r, c2_r, gamma_vr = self._vectorize(surface.wake[k]["right"], surface.gamma_wake["right"], round(len(surface.wake[k]["right"])/(surface.M+1)-1))
                else :
                    c1_r, c2_r, gamma_vr = np.empty((0, 3)), np.empty((0, 3)), np.empty(0)
                c1_tot    = np.concatenate([c1_tot, c1, c1_l, c1_r])  
                c2_tot    = np.concatenate([c2_tot, c2, c2_l, c2_r])  
                if k=="real":           # used to put a - before the mirrored gamma if we want a pure symmetric boundary condition (wall), without a minus its a negative image
                    gamma_tot = np.concatenate([gamma_tot, gamma_v, gamma_vl, gamma_vr]) 
                else :
                    gamma_tot = np.concatenate([gamma_tot, gamma_v, gamma_vl, gamma_vr]) 
        return c1_tot, c2_tot, gamma_tot
            
    def _build_A (self):
        """
        Construct the influence coefficients matrix A by computing the velocity induced at each control point by each vortex ring,
        and projecting it on the normal direction of the panel
        """
        n    = self.N_tot
        ctrl = self.controls
        norm = self.normals
        A = np.zeros((n,n))
        j = 0
        for surface in self.surfaces:
            for i in range(surface.N*surface.M):
                v = surface.wing_panels["real"][i].vrt
                c1 = np.array([v[0], v[1],v[2], v[3]]).reshape(-1,3)                
                c2 = np.array([v[1], v[2],v[3], v[0]]).reshape(-1,3)
                v_ring = self._induced_velocity(ctrl, c1, c2, np.array([1,1,1,1]))  # velocity induced at each control point by the vortex ring of panel j (global) / i (local)
                if self.boundary :
                    v = surface.wing_panels["mirror"][i].vrt
                    c1 = np.array([v[0], v[1],v[2], v[3]]).reshape(-1,3)                
                    c2 = np.array([v[1], v[2],v[3], v[0]]).reshape(-1,3)
                    v_ring = v_ring + self._induced_velocity(ctrl, c1, c2, -np.array([1,1,1,1]))    # velocity induced at each control point by the vortex mirror ring of panel j (global) / i (local)
                A[:,j] = np.sum(v_ring * norm, axis=1)                                           # projection of the induced velocity on the normal direction of each panel
                j += 1
        self.A = A


    def _build_b (self):
        """ 
        Construct the RHS b
        """
        n        = self.N_tot
        u_inf    = self.U
        ctrl     = self.controls
        normal   = self.normals
        surfaces = self.surfaces
        u_inf  = np.tile(u_inf, (n,1))   # n would return a 1D array [u_x, u_y, u_z, u_x, u_y, u_z, ...] the tuple parameters is needed to reshape it in a (n, 3) array
        b = np.zeros(n)
        if np.size(surfaces[0].wake["real"]["middle"]) == 0 :
            v = np.tile(np.array([0,0,0]), (n,1))
        else :
            c1, c2, gamma_v = self._full_vectorize(False)
            v = self._induced_velocity(ctrl, c1, c2, gamma_v)                 # velocity induced at each control point by the wake vortex rings
        b = -np.sum((v + u_inf) * normal, axis=1)
        self.b = b


    def _kuttas_loads(self):
        """
        Compute the loads by applying the Kutta-Joukowski theorem to each vortex segment, and summing up all the contributions
        Each surface ends up whith its loads computed
        
        """
        u        = self.U
        surfaces = self.surfaces
        for surface in surfaces:
            m, n        = surface.M, surface.N
            wing_panels = surface.wing_panels
            gamma       = surface.gamma
            u_inf    = np.tile(u, (m*n, 1))          # m*n would return a 1D array [u_x, u_y, u_z, u_x, u_y, u_z, ...] the tuple parameters is needed to reshape it in a (m*n, 3) array
            u_inf_t  = np.tile(u, ((n-1)*m, 1))      # n-1 because at both tip the local circulation is none (we assume steady state for the loads computation)
            circ     = gamma - np.concatenate([np.zeros(n), gamma[:n*(m-1)]])                            # circulation at each bound segment
            circ_t   = gamma.reshape(m, n)[:,1:].reshape(-1) - gamma.reshape(m, n)[:,:-1].reshape(-1)    # circulation at each trailing segment
            width    = []         
            points   = []         
            chords   = []         
            points_t = []        
            for k,p in enumerate(wing_panels["real"]):
                width.append(p.vrt[1]-p.vrt[0])             # width of each bound segment
                points.append((p.vrt[0] + p.vrt[1])/2)      # mid bound segment points
                if k%n != 0 :                                   # tips are not taken into account TODO tip shedding
                    chords.append(p.vrt[0] - p.vrt[3])          # length of each trailing segment
                    points_t.append((p.vrt[0] + p.vrt[3])/2)    # mid trailing segment points
            width    = np.array(width)
            points   = np.array(points)
            chords   = np.array(chords)
            points_t = np.array(points_t)
            c1, c2, gamma_v = self._full_vectorize(True)
            F_trailing = np.cross(u_inf_t + self._induced_velocity(points_t, c1, c2, gamma_v), circ_t[:,None]*chords)          # loads of the trailing segments by Kutta-Joukowski
            F_bound    = np.cross(u_inf + self._induced_velocity(points, c1, c2, gamma_v), circ[:,None]*width)                 # loads of the bound segments by Kutta-Joukowski                                  
            F_tot      = np.sum( np.concatenate([F_bound, F_trailing]), axis = 0 ) 
            S = np.sum(np.array([p.area for p in surface.wing_panels["real"]]).reshape(m,n), axis=0)
            surface.Cl_2d = 2*np.sum(F_bound[:,1].reshape(m, n), axis=0)/S
            surface.loads = F_tot
        
    def _secondary_computation (self):
        """ 
        Compute the loads using the pressure difference between the two sides of each panel, and summing up all the contributions


        """
        u_inf = self.U
        for surface in self.surfaces :
            m, n    = surface.M, surface.N
            u_inf   = self.U
            gamma   = surface.gamma
            gamma_w = surface.gamma_wake
            panels  = surface.wing_panels["real"]
            ctrl    = np.array([p.ctr for p in panels])
            normals = np.array([p.normal for p in panels])
            u_inf   = np.tile(u_inf, (n*m, 1))
            dp      = []
            print("gamma :", gamma)
            d_gam_i = np.concatenate([gamma[n:],gamma_w["middle"][:n]]) - np.concatenate([np.zeros_like(gamma[:n]), gamma[:n*(m-1)]])     # delta circulation at each control point in the chordwise direction   
            # delta circulation at each control point in the spanwise direction
            if surface.tip_shed == "both":  
                n_w = round(len(gamma_w["left"])/m)                            # number of panels in chordwise direction for the tip wake                
                d_gam_j = np.hstack([gamma.reshape(m, n)[:,1:],gamma_w["right"].reshape(m,n_w)[:,0][:,None]]).reshape(-1) - np.hstack([gamma_w["left"].reshape(m,n_w)[:,-1][:,None], gamma.reshape(m, n)[:,:-1]]).reshape(-1)  
            elif surface.tip_shed == "right":
                n_w = round(len(gamma_w["right"])/m)                           # number of panels in chordwise direction for the tip wake
                d_gam_j = np.hstack([gamma.reshape(m, n)[:,1:],gamma_w["right"].reshape(m,n_w)[:,0][:,None]]).reshape(-1) - np.hstack([gamma.reshape(m, n)[:,0][:,None], gamma.reshape(m, n)[:,:-1]]).reshape(-1)
            elif surface.tip_shed == "left": 
                n_w = round(len(gamma_w["left"])/m )                           # number of panels in chordwise direction for the tip wake
                d_gam_j = np.hstack([gamma.reshape(m, n)[:,1:],gamma.reshape(m, n)[:,-1][:,None]]).reshape(-1) - np.hstack([gamma_w["left"].reshape(m,n_w)[:,-1][:,None], gamma.reshape(m, n)[:,:-1]]).reshape(-1)
            else :
                d_gam_j = np.hstack([gamma.reshape(m, n)[:,1:],gamma.reshape(m, n)[:,-1][:,None]]).reshape(-1) - np.hstack([gamma.reshape(m, n)[:,0][:,None], gamma.reshape(m, n)[:,:-1]]).reshape(-1)
            print("dekta gam I : ",d_gam_i)
            print("delta gam J : ",d_gam_j)
            
            taux_i  = np.array([p.chord/(np.linalg.norm(p.chord)**2) for p in panels])          # chordwise unit vector divided by the mean chord length, for each panel
            taux_j  = np.array([p.width/(np.linalg.norm(p.width)**2) for p in panels])          # spanwise unit vector divided by the mean width, for each panel
            S       = np.array([p.area for p in panels])                                        # area of each panel
            c1, c2, gamma_v = self._full_vectorize(False)
            V       = u_inf + self._induced_velocity(ctrl, c1, c2, gamma_v)
            dp      = np.sum(V * (taux_i*d_gam_i[:,None]/2 + taux_j*d_gam_j[:,None]/2), axis=1) 
            dF      = -dp[:,None]*S[:,None]*normals                                  
            F_tot   = np.sum(dF, axis = 0)
            surface.loads = F_tot

    def _time_sim(self, t, dt, distribution):
        """ 
        Do the time stepping simulation with the wake relaxation

        Input :
            t            -> total simulation time
            dt           -> time step length
            distribution -> type of time step distribution
        """
        # Initialization
        u_inf    = self.U
        surfaces = self.surfaces
        boundary = self.boundary
        self._build_A()
        A = self.A
        inv_A = np.linalg.inv(A)
        self._build_b()
        b = self.b
        Gamma_tot = inv_A @ b               # first solve without wake
        n_gamma   = 0                       # give the start of Gamma in Gamma_tot for each surface
        for surface in surfaces:            # wake initialization by getting the edges
            n, m   = surface.N, surface.M  
            surface.gamma = Gamma_tot[n_gamma:n_gamma+n*m] 
            surface.wake["real"]["middle"] = np.copy(surface.edges["middle"])
            surface.wake["real"]["left"]   = np.copy(surface.edges["left"])   
            surface.wake["real"]["right"]  = np.copy(surface.edges["right"])
            n_gamma += n*m
        match distribution :
            case "classic" :
                dta = np.tile(dt, round(t/dt))
            case "cosine" :         # more pannels at the beginning of the simulation, to better capture the starting vortex
                theta = np.linspace(0, np.pi/2, round(t/dt))  
                dta = t*(1-np.cos(theta))
                dta = dta - np.concatenate([np.array([0]), dta[:-1]])
        for s in range(round(t/dt)):
            n_gamma = 0                 # give the start of Gamma in Gamma_tot for each surface
            for surface in surfaces:
                n, m   = surface.N, surface.M 
                wake   = surface.wake["real"]["middle"]
                l_wake = surface.wake["real"]["left"]
                r_wake = surface.wake["real"]["right"]
                # simulate the wing advancement
                wake   = wake + dta[s]*u_inf
                l_wake = l_wake + dta[s]*u_inf
                r_wake = r_wake + dta[s]*u_inf
                # shed one row of ring
                wake   = np.concatenate([surface.edges["middle"], wake])                       # for each part of the wake we add its shedding edge to the newly convect wake
                l_wake = np.hstack([l_wake.reshape(m+1, s+1, 3), surface.edges["left"].reshape(m+1, 1, 3)]).reshape(-1, 3)          # vertical concatenation
                r_wake = np.hstack([surface.edges["right"].reshape(m+1, 1, 3), r_wake.reshape(m+1, s+1, 3)]).reshape(-1, 3)         
                # store the vorteces strentgh of the wake
                Gamma = Gamma_tot[n_gamma:n_gamma+n*m]              # Gamma is the circulation of each panel of this surface
                surface.gamma                = Gamma
                surface.gamma_wake["middle"] = np.concatenate([Gamma[-(n):], surface.gamma_wake["middle"]])         # the newly shed wake panels take the circulation of the previous edge's panels
                surface.gamma_wake["left"]   = np.hstack([surface.gamma_wake["left"].reshape(m, s), Gamma.reshape(m, n)[:,0].reshape(m, 1)]).reshape(-1)            # same with vertical concatenation
                surface.gamma_wake["right"]  = np.hstack([Gamma.reshape(m, n)[:,n-1].reshape(m, 1), surface.gamma_wake["right"].reshape(m, s)]).reshape(-1)
                n_gamma += n*m                                      # updating the position in the total circulation
                # update the wake geometry
                surface._update_wake([wake, l_wake, r_wake])        # updating the wake for the next b computation
            # update the right hand side and gamma copmutation
            self._build_b()
            Gamma_tot = inv_A @ self.b                              # solving the new situation
            n_gamma = 0
            for surface in surfaces:                                # we update the circulations
                surface.gamma = Gamma_tot[n_gamma:n_gamma+surface.N*surface.M]      
                n_gamma += surface.M*surface.N  
            # simulate the wake rollup
            c1, c2, gamma_v = self._full_vectorize(True)            # getting all the segments (wing + wake) that induce velocity
            for surface in surfaces:
                n, m   = surface.N, surface.M 
                wake   = surface.wake["real"]["middle"]
                l_wake = surface.wake["real"]["left"]
                r_wake = surface.wake["real"]["right"]
                wake   = wake + np.concatenate([np.zeros((n+1,3)), dta[s]*self._induced_velocity(wake[n+1:], c1, c2, gamma_v)])    # each corner point except at the edges are convect by the induced velocity
                l_wake = l_wake + np.hstack([
                    dta[s]*self._induced_velocity(l_wake.reshape(m+1, s+2, 3)[:,:s+1].reshape(-1, 3), c1, c2, gamma_v).reshape(m+1, s+1, 3),
                    np.zeros((m+1,1,3))
                    ]).reshape(-1, 3)                               # reshape in 2D grid to not take into account the tip edge
                r_wake = r_wake + np.hstack([
                    np.zeros((m+1,1,3)), 
                    dta[s]*self._induced_velocity(r_wake.reshape(m+1, s+2, 3)[:,1:].reshape(-1, 3), c1, c2, gamma_v).reshape(m+1, s+1, 3),
                    ]).reshape(-1, 3)
                surface._update_wake([wake, l_wake, r_wake])
                # at the last step we build the wake panels, usefull for plotting
                if s == round(t/dt)-1 :
                    span   = surface.span
                    panels = []
                    panels = panels + surface._paneling(surface.wake["real"]["middle"], surface.wake["real"]["middle"], 1, n, span)
                    if ( surface.tip_shed == "left" ) or ( surface.tip_shed == "both" ) :
                        panels = panels + surface._paneling(surface.wake["real"]["left"], surface.wake["real"]["left"], 1, s+1, span)
                    if ( surface.tip_shed == "right" ) or ( surface.tip_shed == "both" ) :
                        panels = panels + surface._paneling(surface.wake["real"]["right"], surface.wake["real"]["right"], 1, s+1, span)
                    surface.wake_panels["real"] = panels
                    if boundary:
                        panels = []
                        panels = panels + surface._paneling(surface.wake["mirror"]["middle"], surface.wake["mirror"]["middle"], 1, n, span)
                        if ( surface.tip_shed == "left" ) or ( surface.tip_shed == "both" ) :
                            panels = panels + surface._paneling(surface.wake["mirror"]["left"], surface.wake["mirror"]["left"], 1, s+1, span)
                        if ( surface.tip_shed == "right" ) or ( surface.tip_shed == "both" ) :
                            panels = panels + surface._paneling(surface.wake["mirror"]["right"], surface.wake["mirror"]["right"], 1, s+1, span)
                        surface.wake_panels["mirror"] = panels

