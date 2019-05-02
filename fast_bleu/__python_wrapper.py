import ctypes


def _encode_listoflist_str(data):
    return [[dd.encode('utf-8') for dd in d] for d in data]


def _load_cdll():
    import os
    curr_path = os.path.dirname(__file__) + '/'
    return ctypes.cdll.LoadLibrary(curr_path + '__fast_bleu_module.so')


class Bleu:
    def __init__(self, lines_of_tokens: list, weights: dict,
                 smoothing_func: int = 1, auto_reweight: bool = False):
        max_n = max(list(map(lambda x: len(x), weights.values())))
        min_n = min(list(map(lambda x: len(x), weights.values())))
        assert 2 <= min_n <= max_n
        assert smoothing_func in [0, 1]
        self.__weight_keys = list(weights.keys())
        self.__weights = [list(weights[k]) for k in self.__weight_keys]
        self.__init_cdll()
        lines_of_tokens = _encode_listoflist_str(lines_of_tokens)
        self.__instance = self.__get_instance(lines_of_tokens, self.__weights, max_n, smoothing_func, auto_reweight)

    def __init_cdll(self):
        self.__lib = _load_cdll()
        self.__get_instance = self.__lib.get_bleu_instance
        self.__get_score = self.__lib.get_bleu_score
        self.__del_instance = self.__lib.del_bleu_instance

        self.__get_instance.restype = ctypes.c_void_p
        self.__get_instance.argtypes = [ctypes.py_object, ctypes.py_object, ctypes.c_int, ctypes.c_int,
                                        ctypes.c_bool]
        self.__get_score.restype = ctypes.py_object
        self.__get_score.argtypes = [ctypes.c_void_p, ctypes.py_object]

        self.__del_instance.argtypes = [ctypes.c_void_p]

    def get_score(self, hypotheses: list):
        hypotheses = _encode_listoflist_str(hypotheses)
        result = self.__get_score(self.__instance, hypotheses)
        return {self.__weight_keys[i]: r for i, r in enumerate(result)}

    def __del__(self):
        if hasattr(self, '__instance') and hasattr(self, '__del_instance '):
            self.__del_instance(self.__instance)


class SelfBleu:
    def __init__(self, lines_of_tokens: list, weights: dict,
                 smoothing_func: int = 1, auto_reweight: bool = False):
        max_n = max(list(map(lambda x: len(x), weights.values())))
        min_n = min(list(map(lambda x: len(x), weights.values())))
        assert 2 <= min_n <= max_n
        assert smoothing_func in [0, 1]
        self.__weight_keys = list(weights.keys())
        self.__weights = [list(weights[k]) for k in self.__weight_keys]
        self.__init_cdll()
        lines_of_tokens = _encode_listoflist_str(lines_of_tokens)
        self.__instance = self.__get_instance(lines_of_tokens, self.__weights, max_n, smoothing_func, auto_reweight)

    def __init_cdll(self):
        self.__lib = _load_cdll()
        self.__get_instance = self.__lib.get_selfbleu_instance
        self.__get_score = self.__lib.get_selfbleu_score
        self.__del_instance = self.__lib.del_selfbleu_instance

        self.__get_instance.restype = ctypes.c_void_p
        self.__get_instance.argtypes = [ctypes.py_object, ctypes.py_object, ctypes.c_int, ctypes.c_int,
                                        ctypes.c_bool]
        self.__get_score.restype = ctypes.py_object
        self.__get_score.argtypes = [ctypes.c_void_p]

        self.__del_instance.argtypes = [ctypes.c_void_p]

    def get_score(self):
        result = self.__get_score(self.__instance)
        return {self.__weight_keys[i]: r for i, r in enumerate(result)}

    def __del__(self):
        if hasattr(self, '__instance') and hasattr(self, '__del_instance '):
            self.__del_instance(self.__instance)
