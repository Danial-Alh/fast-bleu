import os
from collections import Counter
from copy import deepcopy

import numpy as np
from nltk.translate.bleu_score import SmoothingFunction

from .bleu import corpus_bleu
from .utils import get_ngrams, Threader


class SelfBleu():  # this class speedup computation when reference is same for multisample
    # Base on https://www.nltk.org/_modules/nltk/translate/bleu_score.html
    def __init__(self, references, weights=np.ones(3) / 3., smoothing_function=SmoothingFunction().method1,
                 auto_reweigh=False, process_num=None, cached_fields=None):
        self.references = references
        self.weights = weights
        self.smoothing_function = smoothing_function
        self.auto_reweigh = auto_reweigh
        self.max_n = len(weights)
        if process_num is None:
            self.process_num = os.cpu_count()
        else:
            self.process_num = process_num

        print('self-bleu{} init!'.format(self.max_n))
        if cached_fields is None:
            self.ref_lens = list(len(reference) for reference in references)
            self.references_ngrams = [get_ngrams(references, n + 1) for n in range(self.max_n)]
            self.references_counts = [[Counter(l) for l in self.references_ngrams[n]] for n in range(self.max_n)]
            tmp_counts = [self.get_reference_max_counts(n) for n in range(self.max_n)]
            self.reference_max_counts, self.reference_max2_counts = \
                [t[0] for t in tmp_counts], [t[1] for t in tmp_counts]
        else:
            ref_lens, \
            references_ngrams, \
            references_counts, \
            reference_max_counts, \
            reference_max2_counts = cached_fields
            self.ref_lens = ref_lens[:self.max_n]
            self.references_ngrams = references_ngrams[:self.max_n]
            self.references_counts = references_counts[:self.max_n]
            self.reference_max_counts = reference_max_counts[:self.max_n]
            self.reference_max2_counts = reference_max2_counts[:self.max_n]

    def get_cached_fields(self):
        return self.ref_lens, \
               self.references_ngrams, \
               self.references_counts, \
               self.reference_max_counts, \
               self.reference_max2_counts

    def get_score(self):
        print('evaluating self-bleu {}!'.format(self.max_n))
        ref_max_counts = deepcopy(self.reference_max_counts)
        return [self.tmp_get_score(ref_max_counts, i) for i in range(len(self.references))]

    def tmp_get_score(self, ref_max_counts, i):
        item = self.references[i]
        item_counts = [self.references_counts[n][i] for n in range(self.max_n)]
        refs = self.references[:i] + self.references[i + 1:]
        ref_lens = self.ref_lens[:i] + self.ref_lens[i + 1:]
        # ref_counts = [self.references_counts[n][:i] + self.references_counts[n][i + 1:] for n in range(self.max_n)]
        for n in range(self.max_n):
            for ng in item_counts[n]:
                if self.reference_max_counts[n][ng] == item_counts[n][ng]:
                    ref_max_counts[n][ng] = self.reference_max2_counts[n][ng]
        result = corpus_bleu(refs, item,
                             ref_max_counts, ref_lens, self.weights,
                             self.smoothing_function, self.auto_reweigh)
        for n in range(self.max_n):
            for ng in item_counts[n]:
                ref_max_counts[n][ng] = self.reference_max_counts[n][ng]
        return result

    def get_reference_max_counts(self, n):
        print('calculating max counts n = %d!' % ((n + 1),))
        ngram_keys = list(set([x for y in self.references_ngrams[n] for x in y]))
        thread_result = Threader(ngram_keys, self.tmp_get_reference_max_counts, show_tqdm=True).run()
        thread_result = np.array(thread_result)
        return dict(zip(ngram_keys, thread_result[:, 0])), \
               dict(zip(ngram_keys, thread_result[:, 1]))

    def tmp_get_reference_max_counts(self, ngram):
        counts = [x.get(ngram, 0) for x in self.references_counts[len(ngram) - 1]]
        max_arg = np.argmax(counts)
        ommited_arg_max_array = counts[:max_arg] + counts[max_arg + 1:]
        max2_arg = np.argmax(ommited_arg_max_array)
        return counts[max_arg], ommited_arg_max_array[max2_arg]
