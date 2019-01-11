"""
flow_direction_over_flat.py
Implementation of Barnes et al.(2014)

Created by J. Lai at UIUC (Nov 2015)
"""

import numpy as np
import time
#import Queue
from collections import deque

import landlab
from landlab import Component, FieldError
from landlab.grid.base import BAD_INDEX_VALUE

from landlab.components.flow_routing.flow_direction_DN import grid_flow_directions


class FlowRouterOverFlat(Component):
    """
    This class calculates the flow direction across topography with flats
    created by the PitFiller module. This approach is presented by 
    Barnes et al. (2014), and it operates by calculating a gradient away
    from higher edges of depressions as well as a gradient towards lower
    edges of depressions, resulting in a convergent flow field without
    changing the actual topography.

    The primary function of this class is :func:`route_flow`.

    Construction:

        PitFiller(grid)

    Parameters
    ----------
    grid : RasterModelGrid
        A grid.

    Returns
    -------
    receiver : 
        Node array of receivers (node that receives flow from current node)


    Examples
    --------
    This class is working together with 
    :func:`landlab.components.flow_routing.route_flow_dn_JL.FlowRouter.route_flow`.
    It is turned on/off by setting the value of 'routing_flat' in the FlowRouter
    module.

    Note: the standand landlab FlowRouter in route_flow_dn.py does not support 
    flow routing over flat. Users should use the FlowRouter in route_flow_dn_JL.py.

    >>> import numpy as np
    >>> from landlab import RasterModelGrid
    >>> from landlab.components.flow_routing.route_flow_dn_JL import FlowRouter
    >>> mg = RasterModelGrid((5, 5), spacing=(1, 1))
    >>> elev = np.array([0.,  0.,  0.,  0.,
    ...                  0.,  0.,  0.,  0.,
    ...                  0.,  0.,  0.,  0.,
    ...                  0.,  0.,  0.,  0.,
    ...                  0.,  0.,  0.,  0.])
    >>> _ = mg.add_field('node','topographic__elevation', elev)
    >>> mg.set_closed_boundaries_at_grid_edges(True, True, True, False)
    >>> fr = FlowRouter(mg, routing_flow=True)
    >>> mg = fr.route_flow()
    
    """

    _name = 'FlowRouterOverFlat'

    _input_var_names = (
        'topographic__elevation'
    )

    _output_var_names = (
        'topographic__elevation',
    )

    _var_units = {
        'topographic__elevation': 'm',
    }

    _var_mapping = {
        'topographic__elevation': 'node',
    }

    _var_doc = {
        'topographic__elevation': 'Land surface topographic elevation',
    }

    def __init__(self, input_grid):

        self._grid = input_grid

        self._n = self._grid.number_of_nodes
        (self._boundary, ) = np.where(self._grid.status_at_node!=0)
        (self._open_boundary, ) = np.where(np.logical_or(self._grid.status_at_node==1, self._grid.status_at_node==2))
        (self._close_boundary, ) = np.where(self._grid.status_at_node==4)

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


    def route_flow(self, receiver, dem='topographic__elevation'):
        """
        Route surface water flow over a landscape with flats.

        Parameters:
        -------
        receiver : 
            Node array of receivers (node that receives flow from current node)

        Returns
        -------
        receiver : 
            Node array of receivers (node that receives flow from current node)
            with flow direction updated.

        """

        self._dem = self._grid['node'][dem]
        """
        if receiver==None:
            self._flow_receiver = self._flow_dirs_d8(self._dem)
        else:
            self._flow_receiver = receiver
        """
        self._flow_receiver = receiver
        #(self._flow_receiver, ss) = grid_flow_directions(self._grid, self._dem)

        # This class also supports cython, but python version is recommended because
        # using cython does not increase the performance significantly.
        method = 'python'
        if method=='cython':
            from flow_direction_over_flat_cython import adjust_flow_direction
            self._flow_receiver = adjust_flow_direction(self._dem, np.intc(self._flow_receiver),
                                                        np.intc(self._boundary), np.intc(self._close_boundary),
                                                        np.intc(self._open_boundary), np.intc(self._neighbors))
        else:
            flat_mask, labels = self._resolve_flats()
            self._flow_receiver = self._flow_dirs_over_flat_d8(flat_mask, labels)

        return self._flow_receiver


    def _resolve_flats(self):

        is_flat = np.zeros(self._n, dtype=bool)
        node_id = np.arange(self._n, dtype=int)
        (sink, ) = np.where(node_id==self._flow_receiver)
        for node in sink:
            if node in self._close_boundary:
                continue
            if not(is_flat[node]):
                is_flat = self._identify_flats(is_flat, node)

        high_edges, low_edges = self._flat_edges(is_flat)

        labels = np.zeros(self._n, dtype='int')
        labelid = 1
        for node in low_edges:
            if labels[node]==0:
                labels = self._label_flats(labels, node, labelid)
                labelid += 1

        flat_mask = np.zeros(self._n, dtype='float')
        flat_height = np.zeros(labelid, dtype='float')

        #this part is bottleneck
        flat_mask, flat_height = self._away_from_higher(flat_mask, labels, flat_height, high_edges)
        flat_mask, flat_height = self._towards_lower(flat_mask, labels, flat_height, low_edges)

        return flat_mask, labels


    def _identify_flats(self, is_flat, node):

        flow_receiver = self._flow_receiver
        neighbors = self._neighbors
        boundary = self._boundary
        dem = self._dem

        to_fill = deque(maxlen=self._n*2)
        to_fill_put = to_fill.append
        to_fill_get = to_fill.popleft

        closed = np.zeros(self._n, dtype=bool)
        closed[node] = True
        to_fill_put(node)
        elev = dem[node]
        while not(len(to_fill)==0):
            node = to_fill_get()
            if is_flat[node]:
                continue
            is_flat[node] = True

            for neighbor_node in neighbors[node]:
                if neighbor_node==-1:
                    continue
                if dem[neighbor_node]!=elev:
                    continue
                if neighbor_node in boundary:
                    continue
                if is_flat[neighbor_node] or closed[neighbor_node]:
                    continue
                closed[neighbor_node] = True
                to_fill_put(neighbor_node)

        return is_flat


    def _flat_edges(self, is_flat):

        flow_receiver = self._flow_receiver
        neighbors = self._neighbors
        boundary = self._boundary
        open_boundary = self._open_boundary
        close_boundary = self._close_boundary
        dem = self._dem

        low_edges = deque(maxlen=self._n*2)
        high_edges = deque(maxlen=self._n*2)
        high_put = high_edges.append
        low_put = low_edges.append

        for node in range(self._n):
            if node in boundary:
                continue
            if not(is_flat[node]):
                continue
            for neighbor_node in neighbors[node]:
                if neighbor_node==-1:
                    continue
                if neighbor_node in boundary:
                    if flow_receiver[node]==node and (neighbor_node in close_boundary):
                        high_put(node)
                        break
                    continue
                if flow_receiver[node]!=node and flow_receiver[neighbor_node]==neighbor_node and dem[node]==dem[neighbor_node]:
                    low_put(node)
                    break
                elif flow_receiver[node]==node and dem[node]<dem[neighbor_node]:
                    high_put(node)
                    break

        return high_edges, low_edges


    def _label_flats(self, labels, node, labelid):

        flow_receiver = self._flow_receiver
        neighbors = self._neighbors
        boundary = self._boundary
        dem = self._dem

        to_fill = deque(maxlen=self._n*2)
        to_fill_put = to_fill.append
        to_fill_get = to_fill.popleft

        closed = np.zeros(self._n, dtype=bool)
        closed[node] = True
        to_fill_put(node)
        elev = dem[node]
        while not(len(to_fill)==0):
            node = to_fill_get()
            if labels[node]!=0:
                continue
            labels[node] = labelid

            for neighbor_node in neighbors[node]:
                if neighbor_node==-1:
                    continue
                if neighbor_node in boundary:
                    continue
                if dem[neighbor_node]!=elev:
                    continue
                if labels[neighbor_node]!=0 or closed[neighbor_node]:
                    continue
                closed[neighbor_node] = True
                to_fill_put(neighbor_node)

        return labels


    def _away_from_higher(self, flat_mask, labels, flat_height, high_edges):

        flow_receiver = self._flow_receiver
        neighbors = self._neighbors
        boundary = self._boundary

        k = 1
        MARKER = -100

        high_put = high_edges.append
        high_get = high_edges.popleft

        closed = np.zeros(self._n, dtype=bool)
        #closed[high_edges.queue] = True
        for i in high_edges:
            closed[i] = True

        high_put(MARKER)
        while len(high_edges)>1:
            node = high_get()

            if node==MARKER:
                k += 1
                high_put(MARKER)
                continue
            if flat_mask[node]>0:
                continue

            flat_mask[node] = k
            flat_height[labels[node]] = k

            for neighbor_node in neighbors[node]:
                if neighbor_node==-1:
                    continue
                if neighbor_node in boundary:
                    continue
                if flat_mask[neighbor_node]>0:
                    continue
                if closed[neighbor_node]:
                    continue
                if labels[neighbor_node]==labels[node] and flow_receiver[neighbor_node]==neighbor_node:
                    closed[neighbor_node] = True
                    high_put(neighbor_node)

        return flat_mask, flat_height


    def _towards_lower(self, flat_mask, labels, flat_height, low_edges):

        flow_receiver = self._flow_receiver
        neighbors = self._neighbors
        boundary = self._boundary

        flat_mask = 0-flat_mask
        k = 1
        MARKER = -100

        low_put = low_edges.append
        low_get = low_edges.popleft

        closed = np.zeros(self._n, dtype=bool)
        #closed[low_edges.queue] = True
        for i in low_edges:
            closed[i] = True

        low_put(MARKER)
        while len(low_edges)>1:
            node = low_get()

            if node==MARKER:
                k += 1
                low_put(MARKER)
                continue
            if flat_mask[node]>0:
                continue

            if flat_mask[node]<0:
                flat_mask[node] = flat_height[labels[node]]+flat_mask[node]+2*k
            else:
                flat_mask[node] = 2*k

            for neighbor_node in neighbors[node]:
                if neighbor_node==-1:
                    continue
                if neighbor_node in boundary:
                    continue
                if flat_mask[neighbor_node]>0:
                    continue
                if closed[neighbor_node]:
                    continue
                if labels[neighbor_node]==labels[node] and flow_receiver[neighbor_node]==neighbor_node:
                    closed[neighbor_node] = True
                    low_put(neighbor_node)
            
        return flat_mask, flat_height


    def _flow_dirs_d8(self, dem):

        flow_receiver = np.arange(self._n)

        for node in range(self._n):
            if node in self._boundary:
                continue
            min_elev = dem[node]
            receiver = node
            for neighbor_node in self._neighbors[node]:
                if neighbor_node==-1:
                    continue
                if neighbor_node in self._open_boundary:
                    receiver = neighbor_node
                    break
                if neighbor_node in self._close_boundary:
                    continue
                if dem[neighbor_node]<min_elev:
                    min_elev = dem[neighbor_node]
                    receiver = neighbor_node
            flow_receiver[node] = receiver

        return flow_receiver


    def _flow_dirs_over_flat_d8(self, flat_mask, labels):

        flow_receiver = self._flow_receiver
        neighbors = self._neighbors
        boundary = self._boundary
        for node in range(self._n):
            if flow_receiver[node]!=node:
                continue
            if node in boundary:
                continue
            potential_receiver = neighbors[node]
            potential_receiver = potential_receiver[np.where(potential_receiver!=-1)]
            potential_receiver = potential_receiver[np.where(labels[potential_receiver]==labels[node])]
            receiver = potential_receiver[np.argmin(flat_mask[potential_receiver])]
            flow_receiver[node] = receiver

        return flow_receiver
