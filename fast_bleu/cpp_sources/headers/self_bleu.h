#ifndef SELF_BLEU_CPP_H
#define SELF_BLEU_CPP_H
#include <string>
#include <vector>
#include "custmap.h"
#include "fraction.h"
#include "counter.h"

using namespace std;

class SELF_BLEU_CPP
{
public:
  ~SELF_BLEU_CPP();
  SELF_BLEU_CPP();
  SELF_BLEU_CPP(vector<vector<string>>, vector<vector<float>>, int, int, bool, bool);
  vector<vector<double>> get_score();

private:
  vector<string> **references;
  vector<string> ***references_ngrams;
  Counter ***references_counts;
  CustomMap **reference_max_counts;
  CustomMap **reference_max2_counts;
  vector<vector<float>> weights;
  int smoothing_function;
  bool auto_reweight;
  int max_n;
  int *ref_lens;
  int number_of_refs;
  int n_cores;
  bool verbose;

  void get_max_counts(int);
  void get_max_counts_old(int);
};

#endif