import numpy as np
cimport numpy as np
cimport cython
from libcpp.deque cimport deque

DTYPE_FLOAT = np.double
ctypedef np.double_t DTYPE_FLOAT_t

DTYPE_INT = np.int
ctypedef np.int_t DTYPE_INT_t

import time

cdef class FlowRouterOverFlat_Cy:

    cdef int _n
    cdef double[:] _dem
    cdef int[:] _flow_receiver, _boundary, _close_boundary, _open_boundary
    cdef int[:, :] _neighbors

    def __init__(self, np.ndarray[DTYPE_FLOAT_t, ndim=1] dem,
                 np.ndarray[DTYPE_INT_t, ndim=1] receiver,
                 np.ndarray[DTYPE_INT_t, ndim=1] bdry,
                 np.ndarray[DTYPE_INT_t, ndim=1] c_bdry,
                 np.ndarray[DTYPE_INT_t, ndim=1] o_bdry,
                 np.ndarray[DTYPE_INT_t, ndim=2] neighbors):
        start_time = time.time()
        self._n = len(dem)
        self._dem = dem
        self._flow_receiver = receiver
        self._boundary = bdry
        self._close_boundary = c_bdry
        self._open_boundary = o_bdry
        self._neighbors = neighbors
        print "__init__", time.time()-start_time


    cdef np.ndarray route_flow(self):

        cdef np.ndarray[DTYPE_INT_t, ndim=1] flow_receiver = np.asarray(self._flow_receiver)

        flat_mask, labels = self._resolve_flats()
        flow_receiver = self._flow_dirs_over_flat_d8(flow_receiver, flat_mask, labels)

        return flow_receiver


    cdef _resolve_flats(self):

        cdef int i
        cdef np.ndarray[DTYPE_INT_t, ndim=1] flow_receiver = np.asarray(self._flow_receiver)
        cdef np.ndarray[DTYPE_INT_t, ndim=1] close_boundary = np.asarray(self._close_boundary)

        cdef np.ndarray[DTYPE_INT_t, ndim=1] is_flat = np.zeros(self._n, dtype=DTYPE_INT)
        cdef np.ndarray[DTYPE_INT_t, ndim=1] node_id = np.arange(self._n, dtype=DTYPE_INT)
        cdef DTYPE_INT_t node

        (temp, ) = np.where(node_id==flow_receiver)
        cdef np.ndarray[DTYPE_INT_t, ndim=1] sink = np.zeros(len(temp), dtype=DTYPE_INT)
        for i in range(len(temp)):
            sink[i] = temp[i]

        start_time = time.time()
        for node in sink:
            if node in close_boundary:
                continue
            if not(is_flat[node]):
                is_flat = self._identify_flats(is_flat, node)
        print '_identify_flats', time.time()-start_time

        start_time = time.time()
        cdef deque[int] high_edges, low_edges
        high_edges, low_edges = self._flat_edges(is_flat)
        print '_flat_edges', time.time()-start_time

        start_time = time.time()
        cdef np.ndarray[DTYPE_INT_t, ndim=1] labels = np.zeros(self._n, dtype=DTYPE_INT)
        cdef int labelid = 1
        for i in range(low_edges.size()):
            node = low_edges.at(i)
            if labels[node]==0:
                labels = self._label_flats(labels, node, labelid)
                labelid += 1
        print '_label_flats', time.time()-start_time

        cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] flat_mask = np.zeros(self._n, dtype=DTYPE_FLOAT)
        cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] flat_height = np.zeros(labelid, dtype=DTYPE_FLOAT)
        #this part is bottleneck
        start_time = time.time()
        self._away_from_higher(flat_mask, labels, flat_height, high_edges)
        print '_away_from_higher', time.time()-start_time
        start_time = time.time()
        self._towards_lower(flat_mask, labels, flat_height, low_edges)
        print '_towards_lower', time.time()-start_time

        return flat_mask, labels


    cdef _identify_flats(self, np.ndarray[DTYPE_INT_t, ndim=1] is_flat, int node_arg):

        cdef np.ndarray[DTYPE_INT_t, ndim=1] flow_receiver = np.asarray(self._flow_receiver)
        cdef np.ndarray[DTYPE_INT_t, ndim=2] neighbors = np.asarray(self._neighbors)
        cdef np.ndarray[DTYPE_INT_t, ndim=1] boundary = np.asarray(self._boundary)
        cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] dem = np.asarray(self._dem)

        cdef deque[int] to_fill
        cdef np.ndarray[DTYPE_INT_t, ndim=1] closed = np.zeros(self._n, dtype=DTYPE_INT)
        cdef DTYPE_INT_t neighbor_node
        cdef int node = node_arg

        closed[node] = 1
        to_fill.push_back(node)
        cdef DTYPE_FLOAT_t elev = dem[node]
        while not(to_fill.size()==0):
            node = to_fill.front()
            to_fill.pop_front()
            if is_flat[node]:
                continue
            is_flat[node] = 1

            for neighbor_node in neighbors[node]:
                if neighbor_node==-1:
                    continue
                if dem[neighbor_node]!=elev:
                    continue
                if neighbor_node in boundary:
                    continue
                if is_flat[neighbor_node] or closed[neighbor_node]:
                    continue
                closed[neighbor_node] = 1
                to_fill.push_back(neighbor_node)


    cdef (deque[int], deque[int]) _flat_edges(self, np.ndarray[DTYPE_INT_t, ndim=1] is_flat):

        cdef np.ndarray[DTYPE_INT_t, ndim=1] flow_receiver = np.asarray(self._flow_receiver)
        cdef np.ndarray[DTYPE_INT_t, ndim=2] neighbors = np.asarray(self._neighbors)
        cdef np.ndarray[DTYPE_INT_t, ndim=1] boundary = np.asarray(self._boundary)
        cdef np.ndarray[DTYPE_INT_t, ndim=1] open_boundary = np.asarray(self._open_boundary)
        cdef np.ndarray[DTYPE_INT_t, ndim=1] close_boundary = np.asarray(self._close_boundary)
        cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] dem = np.asarray(self._dem)

        cdef deque[int] low_edges
        cdef deque[int] high_edges
        cdef int node
        cdef DTYPE_INT_t neighbor_node

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
                        high_edges.push_back(node)
                        break
                    continue
                if flow_receiver[node]!=node and flow_receiver[neighbor_node]==neighbor_node and dem[node]==dem[neighbor_node]:
                    low_edges.push_back(node)
                    break
                elif flow_receiver[node]==node and dem[node]<dem[neighbor_node]:
                    high_edges.push_back(node)
                    break

        return high_edges, low_edges


    cdef _label_flats(self, np.ndarray[DTYPE_INT_t, ndim=1] labels, int node_arg, int labelid_arg):

        cdef np.ndarray[DTYPE_INT_t, ndim=1] flow_receiver = np.asarray(self._flow_receiver)
        cdef np.ndarray[DTYPE_INT_t, ndim=2] neighbors = np.asarray(self._neighbors)
        cdef np.ndarray[DTYPE_INT_t, ndim=1] boundary = np.asarray(self._boundary)
        cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] dem = np.asarray(self._dem)

        cdef deque[int] to_fill
        cdef DTYPE_INT_t neighbor_node
        cdef int node = node_arg
        cdef int labelid = labelid_arg

        cdef np.ndarray[DTYPE_INT_t, ndim=1] closed = np.zeros(self._n, dtype=DTYPE_INT)
        closed[node] = 1
        to_fill.push_back(node)
        cdef DTYPE_FLOAT_t elev = dem[node]
        while not(to_fill.size()==0):
            node = to_fill.front()
            to_fill.pop_front()
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
                closed[neighbor_node] = 1
                to_fill.push_back(neighbor_node)


    cdef _away_from_higher(self, np.ndarray[DTYPE_FLOAT_t, ndim=1] flat_mask,
                          np.ndarray[DTYPE_INT_t, ndim=1] labels,
                          np.ndarray[DTYPE_FLOAT_t, ndim=1] flat_height,
                          deque[int] high_edges):

        cdef np.ndarray[DTYPE_INT_t, ndim=1] flow_receiver = np.asarray(self._flow_receiver)
        cdef np.ndarray[DTYPE_INT_t, ndim=2] neighbors = np.asarray(self._neighbors)
        cdef np.ndarray[DTYPE_INT_t, ndim=1] boundary = np.asarray(self._boundary)
        cdef int k = 1
        cdef int MARKER = -100
        cdef int node
        cdef DTYPE_INT_t neighbor_node

        cdef np.ndarray[DTYPE_INT_t, ndim=1] closed = np.zeros(self._n, dtype=DTYPE_INT)
        cdef int i
        cdef int tempn
        for i in range(high_edges.size()):
            tempn = high_edges.at(i)
            closed[tempn] = 1

        high_edges.push_back(MARKER)
        while high_edges.size()>1:
            node = high_edges.front()
            high_edges.pop_front()

            if node==MARKER:
                k += 1
                high_edges.push_back(MARKER)
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
                    closed[neighbor_node] = 1
                    high_edges.push_back(neighbor_node)


    cdef _towards_lower(self, np.ndarray[DTYPE_FLOAT_t, ndim=1] flat_mask,
                       np.ndarray[DTYPE_INT_t, ndim=1] labels,
                       np.ndarray[DTYPE_FLOAT_t, ndim=1] flat_height,
                       deque[int] low_edges):

        cdef np.ndarray[DTYPE_INT_t, ndim=1] flow_receiver = np.asarray(self._flow_receiver)
        cdef np.ndarray[DTYPE_INT_t, ndim=2] neighbors = np.asarray(self._neighbors)
        cdef np.ndarray[DTYPE_INT_t, ndim=1] boundary = np.asarray(self._boundary)
        cdef int k = 1
        cdef int MARKER = -100
        cdef int node
        cdef DTYPE_INT_t neighbor_node

        flat_mask = 0-flat_mask

        cdef np.ndarray[DTYPE_INT_t, ndim=1] closed = np.zeros(self._n, dtype=DTYPE_INT)
        cdef int i
        cdef int tempn
        for i in range(low_edges.size()):
            tempn = low_edges.at(i)
            closed[tempn] = 1

        low_edges.push_back(MARKER)
        while low_edges.size()>1:
            node = low_edges.front()
            low_edges.pop_front()

            if node==MARKER:
                k += 1
                low_edges.push_back(MARKER)
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
                    closed[neighbor_node] = 1
                    low_edges.push_back(neighbor_node)


    cdef np.ndarray[DTYPE_INT_t, ndim=1] _flow_dirs_over_flat_d8(self, np.ndarray[DTYPE_INT_t, ndim=1] flow_receiver,
                                np.ndarray[DTYPE_FLOAT_t, ndim=1] flat_mask,
                                np.ndarray[DTYPE_INT_t, ndim=1] labels):

        cdef np.ndarray[DTYPE_INT_t, ndim=2][DTYPE_INT_t, ndim=1] neighbors = np.asarray(self._neighbors)
        cdef np.ndarray[DTYPE_INT_t, ndim=1] boundary = np.asarray(self._boundary)

        cdef int node

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


def adjust_flow_direction(np.ndarray[DTYPE_FLOAT_t, ndim=1] dem,
                          np.ndarray[DTYPE_INT_t, ndim=1] receiver,
                          np.ndarray[DTYPE_INT_t, ndim=1] bdry,
                          np.ndarray[DTYPE_INT_t, ndim=1] c_bdry,
                          np.ndarray[DTYPE_INT_t, ndim=1] o_bdry,
                          np.ndarray[DTYPE_INT_t, ndim=2] neighbors):

    fr = FlowRouterOverFlat_Cy(dem, receiver, bdry, c_bdry, o_bdry, neighbors)
    receiver = fr.route_flow()

    return receiver
