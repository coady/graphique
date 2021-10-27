# distutils: language=c++
# cython: language_level=3, boundscheck=False, wraparound=False
import numpy as np
from cython.operator cimport postincrement
from libcpp.unordered_map cimport unordered_map


def asiarray(array):
    return np.asarray(array).astype(np.intp, casting='safe', copy=False)


def group_indices(array):
    """Return unique counts and grouped index array."""
    indices = np.empty(len(array), np.intp)
    values, counts = array.value_counts().flatten()
    cdef:
        Py_ssize_t [:] indices_view = indices
        const Py_ssize_t [:] array_view = asiarray(array)
        const Py_ssize_t [:] values_view = asiarray(values)
        const Py_ssize_t [:] counts_view = asiarray(counts)
        unordered_map[Py_ssize_t, Py_ssize_t] offsets
        Py_ssize_t offset = 0
    with nogil:
        offsets.reserve(values_view.shape[0])
        for i in range(values_view.shape[0]):
            offsets[values_view[i]] = offset
            offset += counts_view[i]
        for i in range(array_view.shape[0]):
            indices_view[postincrement(offsets[array_view[i]])] = i
    return counts, indices
