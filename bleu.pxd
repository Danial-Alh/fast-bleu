from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool


cdef extern from "cpp_sources/bleu.cpp":
    pass

cdef extern from "cpp_sources/bleu.h":
    cdef cppclass BLEU_CPP:
        BLEU_CPP() nogil except +
        BLEU_CPP(vector[vector[string]], float *, int , int , bool) nogil except +
        void get_score(vector[vector[string]], double*) nogil except +
