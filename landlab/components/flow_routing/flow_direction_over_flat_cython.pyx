cimport numpy as np
cimport cython
from libcpp.queue cimport queue

DTYPE_FLOAT = np.double
ctypedef np.double_t DTYPE_FLOAT_t

DTYPE_INT = np.int
ctypedef np.int_t DTYPE_INT_t


cdef class FlowRouterOverFlat_Cy():

    def __init__(self, np.ndarray[DTYPE_FLOAT_t, ndim=1] dem,
                 np.ndarray[DTYPE_INT_t, ndim=1] receiver
                 np.ndarray[DTYPE_INT_t, ndim=1] bdry
                 np.ndarray[DTYPE_INT_t, ndim=1] c_bdry
                 np.ndarray[DTYPE_INT_t, ndim=1] o_bdry
                 np.ndarray[DTYPE_INT_t, ndim=2] neighbors):

        self._n = len(dem)
        self._dem = dem
        self._flow_receiver = receiver
        self._boundary = bdry
        self._close_boundary = c_bdry
        self._open_boundary = o_bdry
        self._neighbors = neighbors


    def route_flow(self):

        flat_mask, labels = self._resolve_flats()
        receiver = self._flow_dirs_over_flat_d8(flat_mask, labels)

        return receiver


    def _resolve_flats(self):

        cdef np.ndarray is_flat = np.zeros(self._n, dtype=bool)
        cdef np.ndarray node_id = np.arange(self._n, dtype=DTYPE_INT)
        cdef np.ndarray sink = np.zeros(self._n, dtype=bool)
        (sink, ) = np.where(node_id==self._flow_receiver)
        for node in sink:
            if node in self._close_boundary:
                continue
            if not(is_flat[node]):
                is_flat = self._identify_flats(is_flat, node)

        high_edges, low_edges = self._flat_edges(is_flat)

        cdef np.ndarray labels = np.zeros(self._n, dtype=DTYPE_INT)
        cdef int labelid = 1
        for node in low_edges:
            if labels[node]==0:
                labels = self._label_flats(labels, node, labelid)
                labelid += 1

        cdef np.ndarray flat_mask = np.zeros(self._n, dtype='float')
        cdef np.ndarray flat_height = np.zeros(labelid, dtype='float

        #this part is bottleneck
        flat_mask, flat_height = self._away_from_higher(flat_mask, labels, flat_height, high_edges)
        flat_mask, flat_height = self._towards_lower(flat_mask, labels, flat_height, low_edges)

        return flat_mask, labels


    def _identify_flats(self, is_flat, node):

        cdef np.ndarray flow_receiver = self._flow_receiver
        cdef np.ndarray neighbors = self._neighbors
        cdef np.ndarray boundary = self._boundary
        cdef np.ndarray dem = self._dem

        cdef queue to_fill

        cdef np.ndarray closed = np.zeros(self._n, dtype=bool)
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




def adjust_flow_direction(np.ndarray[DTYPE_FLOAT_t, ndim=1] dem,
                          np.ndarray[DTYPE_INT_t, ndim=1] receiver
                          np.ndarray[DTYPE_INT_t, ndim=1] bdry
                          np.ndarray[DTYPE_INT_t, ndim=1] c_bdry
                          np.ndarray[DTYPE_INT_t, ndim=1] o_bdry
                          np.ndarray[DTYPE_INT_t, ndim=2] neighbors):

    fr = FlowRouterOverFlat_Cy(dem, receiver, bdry, c_bdry, o_bdry, neighbors)
    receiver = fr.route_flow()

    return receiver
