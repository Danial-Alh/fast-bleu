def test():
    from fast_bleu import BLEU, SelfBLEU

    ref1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'that',
            'ensures', 'that', 'the', 'military', 'will', 'forever',
            'heed', 'Party', 'commands']
    ref2 = ['It', 'is', 'the', 'guiding', 'principle', 'which',
            'guarantees', 'the', 'military', 'forces', 'always',
            'being', 'under', 'the', 'command', 'of', 'the', 'Party']
    ref3 = ['It', 'is', 'the', 'practical', 'guide', 'for', 'the',
            'army', 'always', 'to', 'heed', 'the', 'directions',
            'of', 'the', 'party']
    hyp1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'which',
            'ensures', 'that', 'the', 'military', 'always',
            'obeys', 'the', 'commands', 'of', 'the', 'party']
    hyp2 = ['he', 'read', 'the', 'book', 'because', 'he', 'was',
            'interested', 'in', 'world', 'history']

    list_of_references = [ref1, ref2, ref3]
    hypotheses = [hyp1, hyp2]

    weights = {'bigram': (1/2., 1/2.), 'trigram': (1/3., 1/3., 1/3.)}
    bleu = BLEU(list_of_references, weights)
    print(bleu.get_score(hypotheses))

    self_bleu = SelfBLEU(list_of_references, weights)
    print(self_bleu.get_score())

test()