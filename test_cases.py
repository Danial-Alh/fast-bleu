import numpy as np
from nltk.translate.bleu_score import SmoothingFunction


def counter_test():
    from build.counter_cy import Counter

    counter = Counter(['اصغر', 'اصغر', 'ali2'])
    counter['hello'] = 2
    print(counter['hello'])
    print(counter['اصغر'])
    print(counter['ali2'])
    print(counter['ali3'])
    # print(counter.get('123', -4))
    print(counter['123'])


def bleu_test():
    weights = [1. / 4 for _ in range(4)]

    # def nltk_bleu(refs, hyps):
    #     from nltk.translate.bleu_score import sentence_bleu
    #     return [sentence_bleu(refs, hyp, weights=weights) for hyp in hyps]
    #     # return [1. for hyp in hyps]

    def nltk_bleu(refs, hyps):
        from old_metrics.bleu import Bleu
        bleu = Bleu(refs, weights, smoothing_function=SmoothingFunction(epsilon=1./10).method1)
        return bleu.get_score(hyps)
        # return [1. for hyp in hyps]

    def cpp_bleu(refs, hyps):
        from lib.bleu import Bleu
        bleu = Bleu(refs, weights, smoothing_function=1, other_instance=Bleu(refs, [0.5, 0.5]))
        return bleu.get_score(hyps)

    from nltk import word_tokenize
    train_tokens = []
    test_tokens = []
    with open('data/coco60-train.txt') as file:
        lines = file.readlines()
    train = lines[:1000]
    test = lines[-1000:]
    for line in train:
        train_tokens.append(word_tokenize(line))
    for line in test:
        test_tokens.append(word_tokenize(line))
    print('tokenized!')

    import time

    start = time.time()
    cpp_result = np.array(cpp_bleu(train_tokens, test_tokens))
    end = time.time()
    cpp_time = end - start

    start = time.time()
    nltk_result = np.array(nltk_bleu(train_tokens, test_tokens))
    end = time.time()
    nltk_time = end - start

    all_in_one = np.core.records.fromarrays([nltk_result, cpp_result, np.abs(nltk_result - cpp_result)],
                                            names='nltk,cpp,diff')
    print(all_in_one)
    print('sum diff ' + str(np.sum(all_in_one.diff)))
    print('nltk: {}, cpp: {}, cpp speedup: {}'.format(nltk_time, cpp_time, float(nltk_time) / cpp_time))


# counter_test()
bleu_test()
