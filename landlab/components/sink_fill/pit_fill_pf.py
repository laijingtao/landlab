"""
pit_fill_pf.py
Priority-Flood algorithm (Barnes et al. 2014)

Created by JL, Nov. 2015
"""
import numpy as np

import Queue

import landlab
from landlab import Component, FieldError
from landlab.grid.base import BAD_INDEX_VALUE

class PitFiller(Component):

    def __init__(self, input_grid):

        self._grid = input_grid
        self._n = self._grid.number_of_nodes
        (self._boundary, ) = np.where(self._grid.status_at_node!=0)
        (self._open_boundary, ) = np.where(np.logical_or(self._grid.status_at_node==1, self._grid.status_at_node==2))
        (self._close_boundary, ) = np.where(self._grid.status_at_node==4)

        #self._neighbors = np.concatenate((self._grid.neighbors_at_node, self._grid.diagonals_at_node), axis=1)
        #self._neighbors[self._neighbors == BAD_INDEX_VALUE] = -1
        self._build_neighbors_list()

    def _build_neighbors_list(self):

        (nrows, ncols) = self._grid.shape
        neighbor_dR = np.array([0, 0, 1, -1, 1, 1, -1, -1])
        neighbor_dC = np.array([1, -1, 0, 0, 1, -1, 1, -1])
        self._neighbors = np.zeros(shape=(self._n, 8), dtype=int)
        self._neighbors[self._neighbors==0] = -1
        for node in range(self._n):
            r = self._grid.node_y[node]/self._grid.dx
            c = self._grid.node_x[node]/self._grid.dx
            for i in range(8):
                neighbor_r = r+neighbor_dR[i]
                neighbor_c = c+neighbor_dC[i]
                if neighbor_r<0 or neighbor_c<0 or neighbor_r>=nrows or neighbor_c>=ncols:
                    continue
                self._neighbors[node][i] = neighbor_r*ncols+neighbor_c


    def pit_fill(self):

        dem = self._grid.at_node['topographic__elevation']
        neighbors = self._neighbors
        closed = np.zeros(self._n, dtype=bool)
        depressions = np.zeros(self._n, dtype=bool)
        raised_node = Queue.Queue(maxsize=self._n)
        priority_queue = Queue.PriorityQueue(maxsize=self._n)

        raised_put = raised_node.put
        raised_get = raised_node.get
        priority_put = priority_queue.put
        priority_get = priority_queue.get
        raised_empty = raised_node.empty
        priority_empty = priority_queue.empty

        for i in self._open_boundary:
            priority_put((dem[i], i))
            closed[i] = True
        for i in self._close_boundary:
            closed[i] = True

        while not(raised_node.empty()) or not(priority_empty()):
            if not(raised_node.empty()):
                node = raised_get()
            else:
                elev, node = priority_get()
            for neighbor_node in neighbors[node]:
                if neighbor_node==-1:
                    continue
                if closed[neighbor_node]:
                    continue
                closed[neighbor_node] = True
                if dem[neighbor_node]<=dem[node]:
                    dem[neighbor_node] = dem[node]
                    depressions[neighbor_node] = True
                    raised_put(neighbor_node)
                else:
                    priority_put((dem[neighbor_node], neighbor_node))

        self._grid.at_node['topographic__elevation'] = dem
        self._grid.at_node['topographic__depressions'] = depressions
        return self._grid
