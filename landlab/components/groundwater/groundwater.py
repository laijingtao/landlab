'''
Solve water table
Laplace equation

Created by JL Apr. 2016
'''

import numpy as np
from landlab import Component, FieldError
from landlab.grid.base import BAD_INDEX_VALUE

class WaterTableSolver(Component):

    def __init__(self, input_grid):
        self._grid = input_grid
        self._n = self._grid.number_of_nodes
        self._neighbors = self._build_neighbors_list()
        self._nrows = self._grid.number_of_node_rows
        self._ncols = self._grid.number_of_node_columns

        right_bdry = np.array(range(2*self._ncols-1, self._n-1, self._ncols))
        top_bdry = np.array(range((self._nrows-1)*self._ncols+1, self._n-1))
        left_bdry = np.array(range(self._ncols, self._n-self._ncols, self._ncols))
        bottom_bdry = np.array(range(1, self._ncols-1))
        self._boundaries = np.array([right_bdry, top_bdry, left_bdry, bottom_bdry])
        self._fixed_boundary_flux = np.zeros(4, dtype=bool)

        next_to_right = right_bdry - 1
        next_to_top = top_bdry - self._ncols
        next_to_left = left_bdry + 1
        next_to_bottom = bottom_bdry + self._ncols
        self._next_to_boundaries = np.array([next_to_right, next_to_top, next_to_left, next_to_bottom])

    def _build_neighbors_list(self):
        (nrows, ncols) = self._grid.shape
        neighbor_dR = np.array([0, 0, 1, -1])
        neighbor_dC = np.array([1, -1, 0, 0])
        neighbors = np.zeros(shape=(self._n, 4), dtype=int)
        neighbors[:] = -1

        for node in range(self._n):
            r = self._grid.node_y[node]/self._grid.dx
            c = self._grid.node_x[node]/self._grid.dx
            for i in range(4):
                neighbor_r = r+neighbor_dR[i]
                neighbor_c = c+neighbor_dC[i]
                if neighbor_r<0 or neighbor_c<0 or neighbor_r>=nrows or neighbor_c>=ncols:
                    continue
                neighbors[node][i] = neighbor_r*ncols+neighbor_c

        return neighbors

    def set_boundary_fixed(self, right_fixed=False, top_fixed=False, left_fixed=False, bottom_fixed=False):
        #set up grid's boundary conditions (right, top, left, bottom)

        try:
            self._grid.at_node['water_table_fixed']
        except FieldError:
            self._grid.at_node['water_table_fixed'] = np.zeros(self._n, dtype=bool)

        if right_fixed:
            self._grid.at_node['water_table_fixed'][self._boundaries[0]] = True

        if top_fixed:
            self._grid.at_node['water_table_fixed'][self._boundaries[1]] = True

        if left_fixed:
            self._grid.at_node['water_table_fixed'][self._boundaries[2]] = True

        if bottom_fixed:
            self._grid.at_node['water_table_fixed'][self._boundaries[3]] = True

    def set_boundary_flux(self, right_flux=None, top_flux=None, left_flux=None, bottom_flux=None):
        #set up grid's boundary flux conditions (right, top, left, bottom)
        if (right_flux is None) and (top_flux is None) and (left_flux is None) and (bottom_flux is None):
            return

        #check 'groundwater_flux_fixed' and 'groundwater_flux'
        try:
            self._grid.at_node['groundwater_flux_fixed']
        except FieldError:
            self._grid.at_node['groundwater_flux_fixed'] = np.zeros(self._n, dtype=bool)
        try:
            self._grid.at_node['groundwater_flux']
        except FieldError:
            self._grid.at_node['groundwater_flux'] = np.zeros(self._n, dtype=float)

        if not(right_flux is None):
            right_bdry = self._boundaries[0]
            self._grid.at_node['groundwater_flux_fixed'][right_bdry] = True
            self._grid.at_node['groundwater_flux'][right_bdry] = right_flux
            self._fixed_boundary_flux[0] = True

        if not(top_flux is None):
            top_bdry = self._boundaries[1]
            self._grid.at_node['groundwater_flux_fixed'][top_bdry] = True
            self._grid.at_node['groundwater_flux'][top_bdry] = top_flux
            self._fixed_boundary_flux[1] = True

        if not(left_flux is None):
            left_bdry = self._boundaries[2]
            self._grid.at_node['groundwater_flux_fixed'][left_bdry] = True
            self._grid.at_node['groundwater_flux'][left_bdry] = left_flux
            self._fixed_boundary_flux[2] = True

        if not(bottom_flux is None):
            bottom_bdry = self._boundaries[3]
            self._grid.at_node['groundwater_flux_fixed'][bottom_bdry] = True
            self._grid.at_node['groundwater_flux'][bottom_bdry] = bottom_flux
            self._fixed_boundary_flux[3] = True

    def solve(self, tolerance=0.05, k_gw=5.):
        h = self._grid.at_node['water_table_elevation']
        fixed_depth = self._grid.at_node['water_table_fixed']

        try:
            self._grid.at_node['groundwater_flux_fixed']
        except FieldError:
            fixed_flux = np.zeros(self._n, dtype=bool)
        else:
            fixed_flux = self._grid.at_node['groundwater_flux_fixed']

        try:
            self._grid.at_node['groundwater_surface_input']
        except FieldError:
            surface_input = False
        else:
            surface_input = True

        active, = np.where(np.logical_and(fixed_depth==False, fixed_flux==False))
        fixed_flux_index, = np.where(fixed_flux==True)
        random_seed = ((h.max()+0.1) - (h.min()-0.1)) * np.random.random(self._n) + h.min()-0.1
        h[active] = random_seed[active]
        count = 1
        flag = False
        while count<100000:
            count += 1
            h_old = h.copy()
            #update water table
            if surface_input:
                h[active] = np.sum(h[self._neighbors[active]], axis=1)/4 + \
                            self._grid.at_node['groundwater_surface_input'][active]/(4.*k_gw)
            else:
                h[active] = np.sum(h[self._neighbors[active]], axis=1)/4
            #boundary flux
            for i in range(4):
                if self._fixed_boundary_flux[i]:
                    h[self._boundaries[i]] = h[self._next_to_boundaries[i]] - \
                                             self._grid.at_node['groundwater_flux'][self._boundaries[i]]*self._grid.dx/k_gw
            #check convergence
            if np.nanmax(np.absolute((h-h_old)/h_old))<tolerance:
                flag = True
                break

        if flag==False:
            print 'Didn\'t meet tolerance!!!'

        self._grid.at_node['water_table_elevation'] = h
        return self._grid
