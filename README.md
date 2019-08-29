# FastBLEU Package

This is a fast multithreaded C++ implementation of NLTK BLEU; computing BLEU and SelfBLEU score for a fixed reference set.
It can return (Self)BLEU for different (max) n-grams simultaneously and efficiently (e.g. BLEU-2, BLEU-3 and etc.).

## Installation
PyPI latest stable release

~~~~~~~~~~~~~~~~~~~~~~~~
pip install --user FastBLEU
~~~~~~~~~~~~~~~~~~~~~~~~

## Sample Usage
Here is an example to compute BLEU-2, BLEU-3, SelfBLEU-2 and SelfBLEU-3:

```python
>>> from fast_bleu import BLEU, SelfBLEU
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
```

which means:
* BLEU-2 for hyp1 is 0.7453559924999299
* BLEU-2 for hyp2 is 0.0191380231127159

* BLEU-3 for hyp1 is 0.6240726901657495
* BLEU-3 for hyp2 is 0.013720869575946234

```python
>>> self_bleu = SelfBLEU(list_of_references, weights)
>>> self_bleu.get_score()
{'bigram': [0.25819888974716115, 0.3615507630310936, 0.37080992435478316],
        'trigram': [0.07808966062765045, 0.20140620205719248, 0.21415334758254043]}
```

which means: 
* SelfBLEU-2 for ref1 is 0.25819888974716115
* SelfBLEU-2 for ref2 is 0.3615507630310936
* SelfBLEU-2 for ref3 is 0.37080992435478316

* SelfBLEU-3 for ref1 is 0.07808966062765045
* SelfBLEU-3 for ref2 is 0.20140620205719248
* SelfBLEU-3 for ref3 is 0.21415334758254043

**Caution** Each token of reference set is converted to string format during computation.

For further details, refer to the documentation provided in the source codes.

# Citation

Please cite our paper if it helps with your research.

* ACL Anthology: https://www.aclweb.org/anthology/W19-2311
* Arxiv link: https://arxiv.org/abs/1904.03971

```latex
@article{https://arxiv.org/abs/1904.03971,
  title={Jointly Measuring Diversity and Quality in Text Generation Models},
  author={Montahaei, Ehsan and Alihosseini, Danial and Baghshah, Mahdieh Soleymani},
  journal={NAACL HLT 2019},
  pages={90},
  year={2019}
}
```