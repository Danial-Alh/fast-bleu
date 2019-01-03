import numpy as np
from nltk.translate.bleu_score import SmoothingFunction


def counter_test():
    from build.counter_cy import Counter

    counter = Counter(['ali2'])
    counter['hello'] = 2
    print(counter['hello'])
    print(counter['ali2'])
    print(counter['ali3'])
    # print(counter.get('123', -4))
    print(counter['123'])


n = 2
weights = np.ones(n) / float(n)


def nltk_org_bleu(refs, hyps):
    from nltk.translate.bleu_score import sentence_bleu
    return [sentence_bleu(refs, hyp, weights=weights, smoothing_function=SmoothingFunction(epsilon=1. / 10).method1)
            for hyp in hyps]
    # return [1. for hyp in hyps]


def nltk_bleu(refs, hyps):
    from old_metrics.bleu import Bleu
    bleu = Bleu(refs, weights, smoothing_function=SmoothingFunction(epsilon=1. / 10).method1,
                other_instance=Bleu(refs, np.ones(5) / 5.))
    return bleu.get_score(hyps)
    # return [1. for hyp in hyps]


def cpp_bleu(refs, hyps):
    from lib.bleu import Bleu
    bleu = Bleu(refs, weights, smoothing_function=1,
                other_instance=Bleu(refs, np.ones(5) / 5.))
    return bleu.get_score(hyps)


def nltk_self_bleu(refs, hyps):
    from old_metrics.self_bleu import SelfBleu
    b5 = SelfBleu(hyps, np.ones(5) / 5.)
    bleu = SelfBleu(hyps, weights, smoothing_function=SmoothingFunction(epsilon=1. / 10).method1,
                    other_instance=b5)
    res = bleu.get_score()
    del b5
    del bleu
    return res
    # return [1. for hyp in hyps]


def cpp_self_bleu(refs, hyps):
    from lib.self_bleu import SelfBleu
    b5 = SelfBleu(hyps, np.ones(5) / 5.)
    bleu = SelfBleu(hyps, weights, smoothing_function=1,
                    other_instance=b5)
    res = bleu.get_score()
    del bleu
    del b5
    return res


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
    print(all_in_one)
    print('sum diff ' + str(np.sum(all_in_one.diff)))
    print('nltk: {}, cpp: {}, cpp speedup: {}'.format(nltk_time, cpp_time, float(nltk_time) / cpp_time))


from nltk import word_tokenize

ref_tokens = []
test_tokens = []

with open('data/t.txt') as file:
    lines = file.readlines()
for line in lines:
    ref_tokens.append(word_tokenize(line))

with open('data/g.txt') as file:
    lines = file.readlines()
for line in lines:
    test_tokens.append(word_tokenize(line))

print('tokenized!')

# compare(nltk_org_bleu, cpp_bleu)
compare(nltk_self_bleu, cpp_self_bleu)

# counter_test()

# self_bleu_test()
