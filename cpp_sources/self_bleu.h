#ifndef SELF_BLEU_CPP_H
#define SELF_BLEU_CPP_H
#include <string>
#include <vector>
#include <map>
#include "fraction.h"
#include "counter.h"

using namespace std;

class SELF_BLEU_CPP
{
  public:
    ~SELF_BLEU_CPP();
    SELF_BLEU_CPP();
    SELF_BLEU_CPP(vector<vector<string>>, float [], int, int, bool, SELF_BLEU_CPP*);
    void get_score(double*);

  private:
    vector<string> **references;
    vector<string> ***references_ngrams;
    Counter ***references_counts;
    map<string, int> **reference_max_counts;
    map<string, int> **reference_max2_counts;
    float *weights;
    int smoothing_function;
    bool auto_reweigh;
    int max_n;
    int *ref_lens;
    int number_of_refs;
    int n_cores;


    void get_max_counts(int);
};

#endif