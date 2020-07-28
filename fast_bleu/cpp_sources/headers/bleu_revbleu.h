#ifndef BLEU_REVBLEU_CPP_H
#define BLEU_REVBLEU_CPP_H
#include <string>
#include <vector>
#include "fraction.h"
#include "counter.h"
#include "custmap.h"

using namespace std;

class BLEU_REVBLEU_CPP
{
public:
  BLEU_REVBLEU_CPP();
  ~BLEU_REVBLEU_CPP();
  BLEU_REVBLEU_CPP(vector<vector<string>>, vector<vector<float>>, int, int, bool, bool);
  vector<vector<double>> get_score(vector<vector<string>>);

private:
  vector<string> **references;
  vector<string> ***references_ngrams;
  Counter ***references_counts;
  CustomMap **reference_max_counts;
  vector<vector<float>> weights;
  int smoothing_function;
  bool auto_reweigh;
  int max_n;
  int *ref_lens;
  int number_of_refs;
  int n_cores;
  bool verbose;

  void get_max_counts(int);
};

#endif