from fast_bleu import BLEU

reference_corpus = [[str(i) + str(j) for i in range(100)] for j in range(70000)]
weight = {'4': (.25, .25, .25, .25)}
BLEU(reference_corpus, weight, verbose=True)