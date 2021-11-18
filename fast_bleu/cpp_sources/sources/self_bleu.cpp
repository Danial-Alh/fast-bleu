#include <iostream>
#include <algorithm>
#include <utility>
#include <thread>
#include <set>
#include <stdexcept>
#include <limits>
#include "self_bleu.h"
#include "counter.h"
#include "nltk.h"

using namespace std;

SELF_BLEU_CPP::SELF_BLEU_CPP()
{
}

SELF_BLEU_CPP::~SELF_BLEU_CPP()
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

    for (int n = 0; n < this->max_n; n++)
        delete this->reference_max2_counts[n];
    delete[] this->reference_max2_counts;
}

SELF_BLEU_CPP::SELF_BLEU_CPP(vector<vector<string>> lines_of_tokens, vector<vector<float>> weights,
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
    this->reference_max2_counts = new CustomMap *[max_n];

    this->ref_lens = new int[lines_of_tokens.size()];
    this->weights = weights;
    this->max_n = max_n;
    this->auto_reweight = auto_reweight;
    this->smoothing_function = smoothing_func;
    this->number_of_refs = (int)lines_of_tokens.size();
    this->verbose = verbose;

    if (this->verbose)
        cout << "self_bleu" << max_n << " init!" << endl;

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

void SELF_BLEU_CPP::get_max_counts_old(int n)
{
    this->reference_max_counts[n] = new CustomMap();
    this->reference_max2_counts[n] = new CustomMap();
    vector<string> ngram_keys = vector<string>();
    for (int i = 0; i < this->number_of_refs; i++)
        for (string &ng : *(this->references_ngrams[n][i]))
            if (find(ngram_keys.begin(), ngram_keys.end(), ng) == ngram_keys.end())
                ngram_keys.push_back(ng);
    if (this->verbose)
        cout << n + 1 << "grams: " << ngram_keys.size() << endl;

    int *temp_max_counts = new int[ngram_keys.size()];
    int *temp_max2_counts = new int[ngram_keys.size()];

#pragma omp parallel
    {
#pragma omp for schedule(guided)
        for (int i = 0; i < (int)ngram_keys.size(); i++)
        {
            string &ng = ngram_keys[i];
            int *counts = new int[number_of_refs];
            for (int j = 0; j < number_of_refs; j++)
                counts[j] = references_counts[n][j]->get(ng);
            int *max_value_ptr = max_element(counts, counts + number_of_refs);
            temp_max_counts[i] = *max_value_ptr;
            (*max_value_ptr) = -1;
            int max_value = *max_element(counts, counts + number_of_refs);
            temp_max2_counts[i] = max_value;
            delete[] counts;
        }
    }

    for (int i = 0; i < (int)ngram_keys.size(); i++)
    {
        string &ng = ngram_keys[i];
        (*reference_max_counts[n])[ng] = temp_max_counts[i];
        (*reference_max2_counts[n])[ng] = temp_max2_counts[i];
    }

    delete[] temp_max_counts;
    delete[] temp_max2_counts;
}

void SELF_BLEU_CPP::get_max_counts(int n)
{
    this->reference_max_counts[n] = new CustomMap();
    this->reference_max2_counts[n] = new CustomMap();
    set<string> ngrams_set = set<string>();

    for (int i = 0; i < this->number_of_refs; i++)
        for (auto &ngram_count : *this->references_counts[n][i])
            ngrams_set.insert(ngram_count.first);

    if (this->verbose)
        cout << n + 1 << "grams: " << ngrams_set.size() << endl;

    auto ngrams_set_list = vector<string>(ngrams_set.cbegin(), ngrams_set.cend());
    auto temp_ngram_counts = map<string, vector<int>>();

    int *temp_max_counts = new int[ngrams_set.size()];
    int *temp_max2_counts = new int[ngrams_set.size()];

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
            vector<int>::iterator max_val_ptr = max_element(temp_ngram_counts[ng].begin(),
                                      temp_ngram_counts[ng].end());
            temp_max_counts[i] = *max_val_ptr;
            if (temp_ngram_counts[ng].size() == 1)
                temp_max2_counts[i] = 0;
            else
            {
                (*max_val_ptr) = -1;
                max_val_ptr = max_element(temp_ngram_counts[ng].begin(),
                                          temp_ngram_counts[ng].end());
                temp_max2_counts[i] = *max_val_ptr;
            }
        }
    }

    for (int i = 0; i < (int)ngrams_set_list.size(); i++)
    {
        string &ng = ngrams_set_list.at(i);
        (*reference_max_counts[n])[ng] = temp_max_counts[i];
        (*reference_max2_counts[n])[ng] = temp_max2_counts[i];
    }

    delete[] temp_max_counts;
    delete[] temp_max2_counts;
}

vector<vector<double>> SELF_BLEU_CPP::get_score()
{
    vector<vector<double>> results;
    for (vector<float> &w : this->weights)
    {
        int curr_n = w.size();
        CustomMap **ref_max_counts = new CustomMap*[curr_n];
        vector<string> **refs = new vector<string>*[number_of_refs - 1];
        int *lens = new int[number_of_refs - 1];
        double *temp_results = new double[number_of_refs];
        //        cout << "calculating self_bleu" << curr_n << " scores!" << endl;

        for (int n = 0; n < curr_n; n++)
            ref_max_counts[n] = new CustomMap(*(reference_max_counts[n]));

        for (int i = 0; i < number_of_refs; i++)
        {
            for (int s = 0, t = 0; s < number_of_refs; s++)
            {
                if (s == i)
                {
                    t = -1;
                    continue;
                }
                lens[s + t] = ref_lens[s];
            }
            for (int s = 0, t = 0; s < number_of_refs; s++)
            {
                if (s == i)
                {
                    t = -1;
                    continue;
                }
                refs[s + t] = references[s];
            }
            for (int n = 0; n < curr_n; n++)
                for (auto const &p : *(references_counts[n][i]))
                {
                    string const &ng = p.first;
                    if (reference_max_counts[n]->get(ng) == references_counts[n][i]->get(ng))
                        (*ref_max_counts[n])[ng] = reference_max2_counts[n]->get(ng);
                }
            vector<string> *hypothesis = references[i];
            temp_results[i] = corpus_bleu(number_of_refs, curr_n,
                                          refs,
                                          hypothesis,
                                          ref_max_counts,
                                          lens,
                                          w,
                                          smoothing_function,
                                          auto_reweight);
            for (int n = 0; n < curr_n; n++)
                for (auto const &p : *(references_counts[n][i]))
                {
                    string const &ng = p.first;
                    (*ref_max_counts[n])[ng] = reference_max_counts[n]->get(ng);
                }
        }

        for (int n = 0; n < curr_n; n++)
            delete ref_max_counts[n];
        results.push_back(vector<double>());
        for (int i = 0; i < number_of_refs; i++)
            results.back().push_back(temp_results[i]);

        delete[] temp_results;
        delete[] lens;
        delete[] refs;
        delete[] ref_max_counts;
    }

    return results;
}