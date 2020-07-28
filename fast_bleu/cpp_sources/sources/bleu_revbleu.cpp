#include <iostream>
#include <algorithm>
#include <set>
#include <utility>
#include <thread>
#include <stdexcept>
#include <limits>
#include "bleu_revbleu.h"
#include "counter.h"
#include "nltk.h"

using namespace std;

BLEU_REVBLEU_CPP::BLEU_REVBLEU_CPP()
{
}

BLEU_REVBLEU_CPP::~BLEU_REVBLEU_CPP()
{
    delete[] ref_lens;

    for (int i = 0; i < this->number_of_refs; i++)
        delete references[i];
    delete[] this->references;

    for (int n = 0; n < this->max_n; n++)
    {
        for (int i = 0; i < this->number_of_refs; i++)
            delete this->references_ngrams[n][i];
        delete[] this->references_ngrams[n];
    }
    delete[] this->references_ngrams;

    for (int n = 0; n < this->max_n; n++)
    {
        for (int i = 0; i < this->number_of_refs; i++)
            delete this->references_counts[n][i];
        delete[] this->references_counts[n];
    }
    delete[] this->references_counts;

    for (int n = 0; n < this->max_n; n++)
        delete this->reference_max_counts[n];
    delete[] this->reference_max_counts;
}

BLEU_REVBLEU_CPP::BLEU_REVBLEU_CPP(vector<vector<string>> lines_of_tokens, vector<vector<float>> weights,
                                   int max_n, int smoothing_func, bool auto_reweight, bool verbose)
{
    this->n_cores = thread::hardware_concurrency();
    this->references = new vector<string> *[lines_of_tokens.size()];
    this->references_ngrams = new vector<string> **[max_n];
    for (int i = 0; i < max_n; i++)
        this->references_ngrams[i] = new vector<string> *[lines_of_tokens.size()];
    this->references_counts = new Counter **[max_n];
    for (int i = 0; i < max_n; i++)
        this->references_counts[i] = new Counter *[lines_of_tokens.size()];
    this->reference_max_counts = new CustomMap *[max_n];

    this->ref_lens = new int[lines_of_tokens.size()];
    this->weights = weights;
    this->max_n = max_n;
    this->auto_reweigh = auto_reweigh;
    this->smoothing_function = smoothing_func;
    this->number_of_refs = (int)lines_of_tokens.size();
    this->verbose = verbose;

    if (this->verbose)
        cout << "bleu" << max_n << " init!" << endl;

    for (int i = 0; i < this->number_of_refs; i++)
    {
        this->references[i] = new vector<string>(lines_of_tokens[i]);
        this->ref_lens[i] = lines_of_tokens[i].size();
    }
    for (int n = 0; n < this->max_n; n++)
    {
        for (int i = 0; i < this->number_of_refs; i++)
        {
            this->references_ngrams[n][i] = get_ngrams(this->references[i], n + 1);
            this->references_counts[n][i] = new Counter(this->references_ngrams[n][i]);
        }
        this->get_max_counts(n);
    }
}

void BLEU_REVBLEU_CPP::get_max_counts(int n)
{
    this->reference_max_counts[n] = new CustomMap();
    set<string> ngrams_set = set<string>();
    for (int i = 0; i < this->number_of_refs; i++)
        for (string &ng : *this->references_ngrams[n][i])
            ngrams_set.insert(ng);
    if (this->verbose)
        cout << n + 1 << "grams: " << ngrams_set.size() << endl;

    int temp_max_counts[ngrams_set.size()];
    vector<string> ngrams_set_list = vector<string>(ngrams_set.cbegin(), ngrams_set.cend());
#pragma omp parallel
    {
#pragma omp for schedule(guided)
        for (int i = 0; i < (int)ngrams_set_list.size(); i++)
        {
            string ng = ngrams_set_list.at(i);
            int counts[this->number_of_refs];
            for (int j = 0; j < this->number_of_refs; j++)
                counts[j] = references_counts[n][j]->get(ng);
            int max_value = *max_element(counts, counts + number_of_refs);
            temp_max_counts[i] = max_value;
        }
    }
    for (int i = 0; i < (int)ngrams_set.size(); i++)
        (*reference_max_counts[n])[ngrams_set_list.at(i)] = temp_max_counts[i];
}

vector<vector<double>> BLEU_REVBLEU_CPP::get_score(vector<vector<string>> hypotheses)
{
    vector<vector<double>> results;
    for (vector<float> &w : this->weights)
    {
        double temp_results[hypotheses.size()];
        int curr_n = w.size();
        //        cout << "calculating bleu" << curr_n << " scores!" << endl;
#pragma omp parallel
        {
#pragma omp for schedule(guided)
            for (int i = 0; i < (int)hypotheses.size(); i++)
            {
                vector<string> *hypothesis = &(hypotheses[i]);
                temp_results[i] = corpus_bleu(number_of_refs, curr_n,
                                              references,
                                              hypothesis,
                                              reference_max_counts,
                                              ref_lens,
                                              w,
                                              smoothing_function,
                                              auto_reweigh);
            }
        }
        results.push_back(vector<double>());
        for (int i = 0; i < (int)hypotheses.size(); i++)
            results.back().push_back(temp_results[i]);
    }
    return results;
}