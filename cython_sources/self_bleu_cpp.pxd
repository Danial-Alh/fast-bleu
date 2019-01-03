from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool


cdef extern from "./../cpp_sources/sources/self_bleu.cpp":
    pass

cdef extern from "./../cpp_sources/headers/self_bleu.h":
    cdef cppclass SELF_BLEU_CPP:
        SELF_BLEU_CPP() nogil except +
        SELF_BLEU_CPP(vector[vector[string]], float *, int , int , bool, SELF_BLEU_CPP*) nogil except +
        void get_score(double*) nogil except +
