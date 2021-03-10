import os

from glob import glob
from nltk import ToktokTokenizer
import numpy as np
from nltk.translate.bleu_score import SmoothingFunction
from fast_bleu import *

os.system("set -x; python setup.py build_ext --build-lib=./")

# min_n = 2
max_n = 5
weights = np.ones(max_n) / float(max_n)


def nltk_org_bleu(refs, hyps):
    from nltk.translate.bleu_score import sentence_bleu
    return [sentence_bleu(refs, hyp, weights=weights, smoothing_function=SmoothingFunction(epsilon=1. / 10).method1)
            for hyp in hyps]


def nltk_bleu(refs, hyps):
    from old_metrics.bleu import Bleu
    bleu = Bleu(refs, weights, smoothing_function=SmoothingFunction(epsilon=1. / 10).method1)
    return bleu.get_score(hyps)
    # return [1. for hyp in hyps]


def cpp_bleu(refs, hyps):
    w = {i: list(np.ones(i) / (i)) for i in range(2, 6)}
    bleu = BLEU(refs, w, verbose=True)
    return bleu.get_score(hyps)[max_n]


def nltk_self_bleu(refs, hyps):
    from old_metrics.self_bleu import SelfBleu
    bleu = SelfBleu(refs, weights, smoothing_function=SmoothingFunction(
        epsilon=1. / 10).method1, verbose=False)
    res = bleu.get_score()
    del bleu
    return res
    # return [1. for hyp in hyps]


def cpp_self_bleu(refs, hyps):
    from fast_bleu.__python_wrapper__ import SelfBLEU
    w = {i: list(np.ones(i) / (i)) for i in range(2, 6)}
    bleu = SelfBLEU(refs, w, verbose=True)
    res = bleu.get_score()
    del bleu
    return res[max_n]


def get_execution_time(func):
    import time
    start = time.time()
    result = np.array(func(ref_tokens, test_tokens))
    end = time.time()
    return result, end-start


def compare(nltk_func, cpp_func):
    cpp_result, cpp_time = get_execution_time(cpp_func)
    nltk_result, nltk_time = get_execution_time(nltk_func)

    all_in_one = np.core.records.fromarrays([nltk_result, cpp_result, np.abs(nltk_result - cpp_result)],
                                            names='nltk,cpp,diff')
    # print(all_in_one)
    print('sum diff ' + str(np.sum(all_in_one.diff)))
    print('nltk: {}, cpp: {}, cpp speedup: {}'.format(
        nltk_time, cpp_time, float(nltk_time) / cpp_time))


tokenizer = ToktokTokenizer().tokenize

ref_tokens = []
test_tokens = []

with open('data/t.txt') as file:
# with open('data/coco60-test.txt') as file:
    lines = file.readlines()
for line in lines:
    ref_tokens.append(tokenizer(line))

with open('data/g.txt') as file:
# with open('data/coco60-train.txt') as file:
    lines = file.readlines()
for line in lines:
    test_tokens.append(tokenizer(line))

print('tokenized!')


compare(nltk_org_bleu, cpp_bleu)
compare(nltk_bleu, cpp_bleu)
compare(nltk_self_bleu, cpp_self_bleu)

# res, ti = get_execution_time(cpp_bleu)
# res, ti = get_execution_time(cpp_self_bleu)
# res = np.mean(res)
# print(res, ti)