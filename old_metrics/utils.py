import os
from multiprocessing.pool import Pool

import nltk
from nltk.translate.bleu_score import ngrams
from tqdm import tqdm


def tokenize(lines, compute_in_parallel=False):
    if isinstance(lines, str):
        return nltk.word_tokenize(lines)
    if compute_in_parallel:
        return Threader(lines, nltk.word_tokenize).run()
    return [nltk.word_tokenize(l) for l in lines]


class Ngram:
    def __init__(self, n):
        self.n = n

    def tmp_ngram(self, x):
        return (list(ngrams(x, self.n)) if len(x) >= self.n else [])


def get_ngrams(sentences, n, use_pool_thread=True):
    ng = Ngram(n)
    if use_pool_thread:
        local_ngramgs = Threader(sentences, ng.tmp_ngram).run()
    else:
        local_ngramgs = [ng.tmp_ngram(sentence) for sentence in sentences]
    return local_ngramgs


class Threader:

    def __init__(self, items, func, proc_num=None, show_tqdm=False):
        self.items = items
        self.func = func
        self.show_tqdm = show_tqdm
        self.total_size = len(items)
        self.proc_num = proc_num
        if proc_num is None:
            self.proc_num = os.cpu_count()
        self.pool = Pool(self.proc_num)
        self.batch_size = int(self.total_size / self.proc_num)
        if self.batch_size == 0:
            self.batch_size = 1

    def run(self):
        import time
        handles = list()
        for i in range(self.proc_num):
            time.sleep(.2)
            handles.append(self.pool.apply_async(self.dummy_splitter, args=(i,)))

        time.sleep(3)

        results = []
        for r in handles:
            results.extend(r.get())
        self.pool.close()
        self.pool.terminate()
        self.pool.join()
        del self.pool
        return results

    def dummy_splitter(self, n):
        if n == (self.proc_num - 1):
            curr_slice = slice(n * self.batch_size, len(self.items))
        else:
            curr_slice = slice(n * self.batch_size, (n + 1) * self.batch_size)
        if curr_slice.start >= len(self.items):
            return []
        sub_items = self.items[curr_slice]
        if self.show_tqdm:
            return [self.func(item) for item in tqdm(sub_items)]
        return [self.func(item) for item in sub_items]

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict
