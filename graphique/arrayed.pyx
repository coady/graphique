# distutils: language=c++
# cython: language_level=3, boundscheck=False, wraparound=False
import numpy as np
import pyarrow as pa
from cython.operator cimport postincrement
from libcpp.unordered_map cimport unordered_map


def asiarray(array):
    return np.asarray(array).astype(np.intp, casting='safe', copy=False)


def arggroupby(array):
    """Return unique keys with corresponding index arrays."""
    cdef const Py_ssize_t [:] array_view = asiarray(array)
    indices = np.empty(len(array), np.intp)
    cdef Py_ssize_t [:] indices_view = indices
    values, counts = array.value_counts().flatten()
    cdef const Py_ssize_t [:] values_view = asiarray(values)
    cdef const Py_ssize_t [:] counts_view = asiarray(counts)
    cdef unordered_map[Py_ssize_t, Py_ssize_t] offsets
    cdef Py_ssize_t offset = 0
    with nogil:
        offsets.reserve(values_view.shape[0])
        for i in range(values_view.shape[0]):
            offsets[values_view[i]] = offset
            offset += counts_view[i]
        for i in range(array_view.shape[0]):
            indices_view[postincrement(offsets[array_view[i]])] = i
    return values, np.split(indices, np.cumsum(counts))


def argunique(array):
    """Return index array of first occurrences.

    Relies on `Array.unique` having stable ordering.
    """
    cdef const Py_ssize_t [:] array_view = asiarray(array)
    cdef const Py_ssize_t [:] values_view = asiarray(array.unique())
    indices = np.empty(values_view.size, np.intp)
    cdef Py_ssize_t [:] indices_view = indices
    cdef Py_ssize_t j = 0
    with nogil:
        for i in range(values_view.shape[0]):
            for j in range(j, array_view.shape[0]):
                if values_view[i] == array_view[j]:
                    break
            indices_view[i] = postincrement(j)
    return indices
