cimport numpy as np
cimport cython
from libcpp.queue cimport queue

DTYPE_FLOAT = np.double
ctypedef np.double_t DTYPE_FLOAT_t

DTYPE_INT = np.int
ctypedef np.int_t DTYPE_INT_t

def adjust_flow_direction(np.ndarray[DTYPE_FLOAT_t, ndim=1] dem,
                          np.ndarray[DTYPE_INT_t, ndim=1] receiver
                          np.ndarray[DTYPE_INT_t, ndim=1] bdry
                          np.ndarray[DTYPE_INT_t, ndim=1] c_bdry
                          np.ndarray[DTYPE_INT_t, ndim=1] o_bdry
                          np.ndarray[DTYPE_INT_t, ndim=2] neighbors):

    resolve_flats(flat_mask, labels)
    receiver = flow_dirs_over_flat_d8(flat_mask, labels)
