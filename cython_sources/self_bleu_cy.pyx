# distutils: language=c++

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool
from cython_sources.self_bleu_cpp cimport SELF_BLEU_CPP
from cpython.mem cimport PyMem_Malloc, PyMem_Free

cdef class SelfBleu:
    cdef SELF_BLEU_CPP* self_bleu_instance
    cdef int num_refs
    cdef float* temp_weights

    def __cinit__(self, references, weights, int smoothing_function=0, bool auto_reweigh=False,
                  SelfBleu other_instance=None):
        self.num_refs = len(references)
        cdef vector[vector[string]] temp_refs = vector[vector[string]]()
        self.temp_weights = <float*> PyMem_Malloc(len(weights) * sizeof(float))
        for ref in references:
            temp_refs.push_back(vector[string]())
            for token in ref:
                temp_refs.back().push_back(token.encode())
        for i, w in enumerate(weights):
            self.temp_weights[i] = w
        cdef SELF_BLEU_CPP* other_cpp_instance = NULL
        if other_instance is not None:
            other_cpp_instance = other_instance.self_bleu_instance
        self.self_bleu_instance = new SELF_BLEU_CPP(temp_refs, self.temp_weights, len(weights), smoothing_function,
                                                    auto_reweigh, other_cpp_instance)

    def __dealloc__(self):
        del self.self_bleu_instance
        PyMem_Free(self.temp_weights)

    cpdef list get_score(self):
        cdef double* results = <double*> PyMem_Malloc(self.num_refs * sizeof(double))
        self.self_bleu_instance.get_score(results)
        cdef list py_results = []
        for i in range(self.num_refs):
            py_results.append(results[i])
        PyMem_Free(results)
        return py_results
