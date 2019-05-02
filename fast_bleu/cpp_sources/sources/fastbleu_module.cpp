#include <Python.h>
#include <iostream>
#include <string>
#include <vector>
#include "bleu.h"
#include "self_bleu.h"
using namespace std;

vector<float> list2vector_float(PyObject* incoming){
    vector<float> data;
    if (PyList_Check(incoming)) {
        for(Py_ssize_t i = 0; i < PyList_Size(incoming); i++) {
            PyObject *value = PyList_GetItem(incoming, i);
            data.push_back((float)PyFloat_AS_DOUBLE(value));
        }
    } else {
        throw logic_error("Passed PyObject pointer was not a float list!");
    }
	return data;
}

vector<vector<float>> listoflist2vectorofvector_float(PyObject* incoming){
    vector<vector<float>> data;
    if (PyList_Check(incoming)) {
        for(Py_ssize_t i = 0; i < PyList_Size(incoming); i++) {
            PyObject *value = PyList_GetItem(incoming, i);
            data.push_back(list2vector_float(value));
        }
    } else {
        throw logic_error("Passed PyObject pointer was not a float list of list!");
    }
	return data;
}

vector<string> list2vector_str(PyObject* incoming){
    vector<string> data;
    if (PyList_Check(incoming)) {
        for(Py_ssize_t i = 0; i < PyList_Size(incoming); i++) {
            PyObject *value = PyList_GetItem(incoming, i);
            data.push_back(PyBytes_AsString(value));
        }
    } else {
        throw logic_error("Passed PyObject pointer was not a string list!");
    }
	return data;
}

vector<vector<string>> listoflist2vectorofvector_str(PyObject* incoming){
    vector<vector<string>> data;
    if (PyList_Check(incoming)) {
        for(Py_ssize_t i = 0; i < PyList_Size(incoming); i++) {
            PyObject *value = PyList_GetItem(incoming, i);
            data.push_back(list2vector_str(value));
        }
    } else {
        throw logic_error("Passed PyObject pointer was not a string list of list!");
    }
	return data;
}

PyObject* vector2list_double(const vector<double> &data)
{
    PyObject* listObj = PyList_New( data.size() );
	if (!listObj) throw logic_error("Unable to allocate memory for Python list");
	for (unsigned int i = 0; i < data.size(); i++) {
		PyObject *num = PyFloat_FromDouble( (double) data[i]);
		if (!num) {
			Py_DECREF(listObj);
			throw logic_error("Unable to allocate memory for Python list");
		}
		PyList_SET_ITEM(listObj, i, num);
	}
	return listObj;
}

PyObject* vectorofvector2listoflist_double(const vector<vector<double>> &data)
{
    PyObject* listObj = PyList_New( data.size() );
	if (!listObj) throw logic_error("Unable to allocate memory for Python list");
	for (unsigned int i = 0; i < data.size(); i++) {
		PyObject *sub_list = vector2list_double(data[i]);
		if (!sub_list) {
			Py_DECREF(listObj);
			throw logic_error("Unable to allocate memory for Python list");
		}
		PyList_SET_ITEM(listObj, i, sub_list);
	}
	return listObj;
}


extern "C" void* get_bleu_instance(PyObject* lines_of_tokens, PyObject* weights,
                                int max_n, int smoothing_func, bool auto_reweight)
{
    void* ptr;
    ptr = (void *) new BLEU_CPP(listoflist2vectorofvector_str(lines_of_tokens), listoflist2vectorofvector_float(weights), max_n, smoothing_func,  auto_reweight);
    return ptr;
}

extern "C" void* get_bleu_score(void* bleu_ptr, PyObject* hypotheses)
{
    
    vector<vector<double>> res = ((BLEU_CPP*) bleu_ptr)->get_score(listoflist2vectorofvector_str(hypotheses));
    return vectorofvector2listoflist_double(res);
}

extern "C" void* del_bleu_instance(void* bleu_ptr)
{
    delete ((BLEU_CPP*) bleu_ptr);
    return 0;
}

extern "C" void* get_selfbleu_instance(PyObject* lines_of_tokens, PyObject* weights,
                                int max_n, int smoothing_func, bool auto_reweight)
{
    void* ptr;
    ptr = (void *) new SELF_BLEU_CPP(listoflist2vectorofvector_str(lines_of_tokens), listoflist2vectorofvector_float(weights), max_n, smoothing_func,  auto_reweight);
    return ptr;
}

extern "C" void* get_selfbleu_score(void* selfbleu_ptr)
{
    
    vector<vector<double>> res = ((SELF_BLEU_CPP*) selfbleu_ptr)->get_score();
    return vectorofvector2listoflist_double(res);
}

extern "C" void* del_selfbleu_instance(void* selfbleu_ptr)
{
    delete ((SELF_BLEU_CPP*) selfbleu_ptr);
    return 0;
}