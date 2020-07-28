from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext
import setuptools
from glob import glob
from nltk import ToktokTokenizer
import numpy as np
from nltk.translate.bleu_score import SmoothingFunction
from fast_bleu import *

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
    print(BLEU)
    w = {i: list(np.ones(i) / (i)) for i in range(2, 6)}
    bleu = BLEU(refs, w, verbose=False)
    return bleu.get_score(hyps)[max_n]


def nltk_self_bleu(refs, hyps):
    from old_metrics.self_bleu import SelfBleu
    bleu = SelfBleu(refs, weights, smoothing_function=SmoothingFunction(epsilon=1. / 10).method1, verbose=False)
    res = bleu.get_score()
    del bleu
    return res
    # return [1. for hyp in hyps]


def cpp_self_bleu(refs, hyps):
    from fast_bleu.__python_wrapper import SelfBLEU
    w = {i: list(np.ones(i) / (i)) for i in range(2, 6)}
    bleu = SelfBLEU(refs, w, verbose=False)
    res = bleu.get_score()
    del bleu
    return res[max_n]


def compare(nltk_func, cpp_func):
    import time
    start = time.time()
    cpp_result = np.array(cpp_func(ref_tokens, test_tokens))
    end = time.time()
    cpp_time = end - start

    start = time.time()
    nltk_result = np.array(nltk_func(ref_tokens, test_tokens))
    end = time.time()
    nltk_time = end - start

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
for line in lines[:500]:
    ref_tokens.append(tokenizer(line))

with open('data/g.txt') as file:
# with open('data/coco60-train.txt') as file:
    lines = file.readlines()
for line in lines[:500]:
    test_tokens.append(tokenizer(line))

print('tokenized!')


class BuildExtWithoutPlatformSuffix(build_ext):
    def get_ext_filename(self, ext_name):
        super().get_ext_filename(ext_name)
        return ext_name + '.so'
    # pass


include_dirs = ['fast_bleu/cpp_sources/headers/']
setup = setuptools.setup(
    name='fast_bleu',
    ext_modules=[
        Extension(
            name="fast_bleu.__fast_bleu_module",
            sources=glob('fast_bleu/cpp_sources/sources/*.cpp'),
            extra_compile_args=['-fopenmp', '-std=c++11'],
            extra_link_args=['-fopenmp', '-std=c++11'],
            include_dirs=include_dirs,
        ), ],
    packages=['fast_bleu'],
    cmdclass={'build_ext': BuildExtWithoutPlatformSuffix},
    script_args=['build_ext', '--build-lib', './']
)

compare(nltk_org_bleu, cpp_bleu)
# compare(nltk_bleu, cpp_bleu)
# compare(nltk_self_bleu, cpp_self_bleu)

# counter_test()

# self_bleu_test()
