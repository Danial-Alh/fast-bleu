#include <iostream>
#include <algorithm>
#include <utility>
#include <thread>
#include <set>
#include <stdexcept>
#include <limits>
#include "bleu.h"
#include "counter.h"
#include "nltk.h"

using namespace std;

BLEU_CPP::BLEU_CPP()
{
}

BLEU_CPP::~BLEU_CPP()
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

BLEU_CPP::BLEU_CPP(vector<vector<string>> lines_of_tokens, vector<vector<float>> weights,
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
    this->auto_reweight = auto_reweight;
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

void BLEU_CPP::get_max_counts_old(int n)
{
    this->reference_max_counts[n] = new CustomMap();
    vector<string> ngram_keys = vector<string>();
    for (int i = 0; i < this->number_of_refs; i++)
        for (string &ng : *this->references_ngrams[n][i])
            if (find(ngram_keys.begin(), ngram_keys.end(), ng) == ngram_keys.end())
                ngram_keys.push_back(ng);
    if (this->verbose)
        cout << n + 1 << "grams: " << ngram_keys.size() << endl;

    int *temp_max_counts = new int[ngram_keys.size()];
#pragma omp parallel
    {
#pragma omp for schedule(guided)
        for (int i = 0; i < (int)ngram_keys.size(); i++)
        {
            string &ng = ngram_keys[i];
            int max_value = 0;
            for (int j = 0; j < number_of_refs; j++)
            {
                int temp_value = references_counts[n][j]->get(ng);
                if (temp_value > max_value)
                    max_value = temp_value;
            }
            temp_max_counts[i] = max_value;
        }
    }
    for (int i = 0; i < (int)ngram_keys.size(); i++)
        (*reference_max_counts[n])[ngram_keys[i]] = temp_max_counts[i];
    
    delete[] temp_max_counts;
}

void BLEU_CPP::get_max_counts(int n)
{
    this->reference_max_counts[n] = new CustomMap();
    set<string> ngrams_set = set<string>();

    for (int i = 0; i < this->number_of_refs; i++)
        for (auto &ngram_count : *this->references_counts[n][i])
            ngrams_set.insert(ngram_count.first);

    if (this->verbose)
        cout << n + 1 << "grams: " << ngrams_set.size() << endl;

    auto ngrams_set_list = vector<string>(ngrams_set.cbegin(), ngrams_set.cend());
    auto temp_ngram_counts = map<string, vector<int>>();
    int *temp_max_counts = new int[ngrams_set.size()];

    for (string &ng : ngrams_set_list)
        temp_ngram_counts.insert(pair<string, vector<int>>(ng, vector<int>()));

    for (int j = 0; j < number_of_refs; j++)
        for (auto &ngram_count : *references_counts[n][j])
            temp_ngram_counts[ngram_count.first].push_back(ngram_count.second);

#pragma omp parallel
    {
#pragma omp for schedule(guided)
        for (int i = 0; i < (int)ngrams_set_list.size(); i++)
        {
            string ng = ngrams_set_list.at(i);
            int max_val = *max_element(temp_ngram_counts[ng].cbegin(),
                                   temp_ngram_counts[ng].cend());
            temp_max_counts[i] = max_val;
        }
    }

    for (int i = 0; i < (int)ngrams_set_list.size(); i++)
        (*reference_max_counts[n])[ngrams_set_list.at(i)] = temp_max_counts[i];
    
    delete[] temp_max_counts;
}

vector<vector<double>> BLEU_CPP::get_score(vector<vector<string>> hypotheses)
{
    vector<vector<double>> results;
    for (vector<float> &w : this->weights)
    {
        double *temp_results = new double[hypotheses.size()];
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
                                              auto_reweight);
            }
        }
        results.push_back(vector<double>());
        for (int i = 0; i < (int)hypotheses.size(); i++)
            results.back().push_back(temp_results[i]);
        
        delete[] temp_results;
    }
    return results;
}