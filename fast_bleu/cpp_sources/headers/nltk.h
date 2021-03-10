#ifndef NLTK_CPP_H
#define NLTK_CPP_H
#include <iostream>
#include <vector>
using namespace std;

vector<string> *get_ngrams(vector<string> *tokens, int n);

void smooth_0(int size, Fraction *p_n);

void smooth_1(int size, Fraction *p_n);

Fraction modified_precision(CustomMap **reference_max_counts,
                            vector<string> *hypothesis, int n);

int closest_ref_length(int num_refs, int *ref_lens, int hyp_len);

double brevity_penalty(int closest_ref_len, int hyp_len);

double corpus_bleu(int num_refs, int max_n,
                   vector<string> **references,
                   vector<string> *hypothesis,
                   CustomMap **reference_max_counts,
                   int* ref_lens,
                   vector<float> weights,
                   int smoothing_function,
                   bool auto_reweight);

#endif