import ctypes
import faulthandler


def _encode_listoflist_str(data):
    return [[str(dd).encode('utf-8') for dd in d] for d in data]


def _load_cdll():
    import os
    curr_path = os.path.dirname(__file__) + '/'
    return ctypes.CDLL(curr_path + '__fast_bleu_module.so',
                                   use_errno=True)


class BLEU:
    """
    A class to compute BLEU score for a fixed reference set.
    It can return BLEU for different (max) n-grams simultaneously and efficiently (e.g. BLEU-2, BLEU-3 and etc.).

    Here is an example to compute BLEU-2 and BLEU-3:
    >>> ref1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'that',
    ...          'ensures', 'that', 'the', 'military', 'will', 'forever',
    ...          'heed', 'Party', 'commands']
    >>> ref2 = ['It', 'is', 'the', 'guiding', 'principle', 'which',
    ...          'guarantees', 'the', 'military', 'forces', 'always',
    ...          'being', 'under', 'the', 'command', 'of', 'the', 'Party']
    >>> ref3 = ['It', 'is', 'the', 'practical', 'guide', 'for', 'the',
    ...          'army', 'always', 'to', 'heed', 'the', 'directions',
    ...          'of', 'the', 'party']

    >>> hyp1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'which',
    ...         'ensures', 'that', 'the', 'military', 'always',
    ...         'obeys', 'the', 'commands', 'of', 'the', 'party']
    >>> hyp2 = ['he', 'read', 'the', 'book', 'because', 'he', 'was',
    ...         'interested', 'in', 'world', 'history']

    >>> list_of_references = [ref1, ref2, ref3]
    >>> hypotheses = [hyp1, hyp2]
    >>> weights = {'bigram': (1/2., 1/2.), 'trigram': (1/3., 1/3., 1/3.)}

    >>> bleu = BLEU(list_of_references, weights)
    >>> bleu.get_score(hypotheses)
    {'bigram': [0.7453559924999299, 0.0191380231127159], 'trigram': [0.6240726901657495, 0.013720869575946234]}

    which means: 
        BLEU-2 for hyp1 is 0.7453559924999299
        BLEU-2 for hyp2 is 0.0191380231127159

        BLEU-3 for hyp1 is 0.6240726901657495
        BLEU-3 for hyp2 is 0.013720869575946234


    Parameters
    ----------
    lines_of_tokens : list
        Reference set.
        List of list of tokens (list of references; each reference is a list of token).
        **Caution** Each token is converted to string format during computation.
    weights : dict, optional
        weights for unigrams, bigrams, trigrams and so on for each BLEU-N.
        A key must be provided for each BLEU-N weights; the BLEU-N will be indentified with this key.
        By default {4: (1./4, 1./4, 1./4, 1./4)}.
    smoothing_func : int, optional
        Smoothing function type. 0 for method_0 and 1 for method_1 (the same as NLTK), by default 1.
    auto_reweight : bool, optional
        Option to re-normalize the weights uniformly, by default False.
    """

    def __init__(self, lines_of_tokens: list, weights: dict = {4: (1./4, 1./4, 1./4, 1./4)},
                 smoothing_func: int = 1, auto_reweight: bool = False, verbose: bool = False):
        max_n = max(list(map(lambda x: len(x), weights.values())))
        min_n = min(list(map(lambda x: len(x), weights.values())))
        self.__weight_keys = list(weights.keys())
        self.__weights = [list(weights[k]) for k in self.__weight_keys]
        assert 2 <= min_n <= max_n, '2 <= min_n <= max_n; got 2 <= {} <= {}'.format(min_n, max_n)
        assert smoothing_func in [0, 1], 'Smoothing function only supports 0 or 1 type.'
        assert not (False in [abs(1. - sum(w)) < 1e-15 for w in self.__weights]
                    ), 'All weights must sum to one.'
        self.__init_cdll()
        lines_of_tokens = _encode_listoflist_str(lines_of_tokens)

        faulthandler_enabled = faulthandler.is_enabled()
        faulthandler.enable()
        self.__instance = self.__get_instance(
            lines_of_tokens, self.__weights, max_n, smoothing_func, auto_reweight, verbose)
        if not faulthandler_enabled:
            faulthandler.disable()

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
        """
        computes BLEU-N score for each hypothesis.

        Parameters
        ----------
        hypotheses : list
            Hypothesis set.
            List of list of tokens (list of hypotheses; each hypothesis is a list of token).
            **Caution** Each token is converted to string format during computation.

        Returns
        -------
        dict
            BLEU-N score of each hypothesis. 
            Each BLEU-N is identified by a key according to the keys provided by 'weights' in __init__.
        """
        hypotheses = _encode_listoflist_str(hypotheses)
        faulthandler_enabled = faulthandler.is_enabled()
        faulthandler.enable()
        result = self.__get_score(self.__instance, hypotheses)
        if not faulthandler_enabled:
            faulthandler.disable()
        return {self.__weight_keys[i]: r for i, r in enumerate(result)}

    def __del__(self):
        if hasattr(self, '_BLEU__instance') and hasattr(self, '_BLEU__del_instance'):
            self.__del_instance(self.__instance)


class SelfBLEU:
    """
    A class to compute SelfBLEU score for a fixed reference set.
    It can return SelfBLEU for different (max) n-grams simultaneously and efficiently (e.g. SelfBLEU-2, SelfBLEU-3 and etc.).

    Here is an example to compute SelfBLEU-2 and SelfBLEU-3:
    >>> ref1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'that',
    ...          'ensures', 'that', 'the', 'military', 'will', 'forever',
    ...          'heed', 'Party', 'commands']
    >>> ref2 = ['It', 'is', 'the', 'guiding', 'principle', 'which',
    ...          'guarantees', 'the', 'military', 'forces', 'always',
    ...          'being', 'under', 'the', 'command', 'of', 'the', 'Party']
    >>> ref3 = ['It', 'is', 'the', 'practical', 'guide', 'for', 'the',
    ...          'army', 'always', 'to', 'heed', 'the', 'directions',
    ...          'of', 'the', 'party']

    >>> hyp1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'which',
    ...         'ensures', 'that', 'the', 'military', 'always',
    ...         'obeys', 'the', 'commands', 'of', 'the', 'party']
    >>> hyp2 = ['he', 'read', 'the', 'book', 'because', 'he', 'was',
    ...         'interested', 'in', 'world', 'history']

    >>> list_of_references = [ref1, ref2, ref3]
    >>> hypotheses = [hyp1, hyp2]
    >>> weights = {'bigram': (1/2., 1/2.), 'trigram': (1/3., 1/3., 1/3.)}

    >>> self_bleu = SelfBLEU(list_of_references, weights)
    >>> self_bleu.get_score()
    {'bigram': [0.25819888974716115, 0.3615507630310936, 0.37080992435478316],
         'trigram': [0.07808966062765045, 0.20140620205719248, 0.21415334758254043]}

    which means: 
        SelfBLEU-2 for ref1 is 0.25819888974716115
        SelfBLEU-2 for ref2 is 0.3615507630310936
        SelfBLEU-2 for ref3 is 0.37080992435478316

        SelfBLEU-3 for ref1 is 0.07808966062765045
        SelfBLEU-3 for ref2 is 0.20140620205719248
        SelfBLEU-3 for ref3 is 0.21415334758254043


    Parameters
    ----------
    lines_of_tokens : list
        Reference set.
        List of list of tokens (list of references; each reference is a list of token).
        **Caution** Each token is converted to string format during computation.
    weights : dict, optional
        weights for unigrams, bigrams, trigrams and so on for each SelfBLEU-N.
        A key must be provided for each SelfBLEU-N weights; the SelfBLEU-N will be indentified with this key.
        By default {4: (1./4, 1./4, 1./4, 1./4)}.
    smoothing_func : int, optional
        Smoothing function type. 0 for method_0 and 1 for method_1 (the same as NLTK), by default 1.
    auto_reweight : bool, optional
        Option to re-normalize the weights uniformly, by default False.
    """

    def __init__(self, lines_of_tokens: list, weights: dict = {4: (1./4, 1./4, 1./4, 1./4)},
                 smoothing_func: int = 1, auto_reweight: bool = False, verbose: bool = False):
        max_n = max(list(map(lambda x: len(x), weights.values())))
        min_n = min(list(map(lambda x: len(x), weights.values())))
        self.__weight_keys = list(weights.keys())
        self.__weights = [list(weights[k]) for k in self.__weight_keys]
        assert 2 <= min_n <= max_n, '2 <= min_n <= max_n; got 2 <= {} <= {}'.format(min_n, max_n)
        assert smoothing_func in [0, 1], 'Smoothing function only supports 0 or 1 type.'
        assert not (False in [abs(1. - sum(w)) < 1e-15 for w in self.__weights]
                    ), 'All weights must sum to one.'
        self.__init_cdll()
        lines_of_tokens = _encode_listoflist_str(lines_of_tokens)

        faulthandler_enabled = faulthandler.is_enabled()
        faulthandler.enable()
        self.__instance = self.__get_instance(
            lines_of_tokens, self.__weights, max_n, smoothing_func, auto_reweight, verbose)
        if not faulthandler_enabled:
            faulthandler.disable()

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
        """
        computes SelfBLEU-N score for each reference.

        Returns
        -------
        dict
            SelfBLEU-N score of each hypothesis. 
            Each SelfBLEU-N is identified by a key according to the keys provided by 'weights' in __init__.
        """
        faulthandler_enabled = faulthandler.is_enabled()
        faulthandler.enable()
        result = self.__get_score(self.__instance)
        if not faulthandler_enabled:
            faulthandler.disable()
        return {self.__weight_keys[i]: r for i, r in enumerate(result)}

    def __del__(self):
        if hasattr(self, '_SelfBLEU__instance') and hasattr(self, '_SelfBLEU__del_instance'):
            self.__del_instance(self.__instance)
