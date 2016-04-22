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

    def solve(self, tolerance=0.05):
        h = self._grid.at_node['water_table_elevation']
        fixed_depth = self._grid.at_node['water_table_fixed']

        try:
            self._grid.at_node['groundwater_flux_fixed']
        except FieldError:
            fixed_flux = np.zeros(len(h), dtype=bool)
        else:
            fixed_flux = self._grid.at_node['groundwater_flux_fixed']

        active, = np.where(np.logical_and(fixed_depth==False, fixed_flux==False))
        fixed_flux_index, = np.where(fixed_flux==True)
        random_seed = np.random.randint(h.min(), h.max(), len(h))
        h[active] = random_seed[active]
        #relax_factor = 4/(2+np.sqrt(4-(np.cos(3.1415/(self._grid.shape[0]-1))+np.cos(3.1415/(self._grid.shape[1]-1)))**2))
        i = 1
        flag = False
        while i<100000:
            #print i
            i += 1
            h_old = h.copy()
            #h[active] = (h[self._neighbors[active, 0]]+h[self._neighbors[active, 1]]+\
            #             h[self._neighbors[active, 2]]+h[self._neighbors[active, 3]])/4
            h[active] = np.sum(h[self._neighbors[active]], axis=1)/4
            #h[fixed_flux_index] = h[np.where(self._grid.node_x==1.*self._grid.dx)]-self._grid.at_node['groundwater_flux'][fixed_flux_index]*self._grid.dx/5.
            if np.nanmax(np.absolute((h-h_old)/h_old))<tolerance:
                flag = True
                break

        if flag==False:
            print 'Didn\'t meet tolerance!!!'

        self._grid.at_node['water_table_elevation'] = h
        return self._grid
