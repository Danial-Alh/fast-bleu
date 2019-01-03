#include <iostream>
#include <algorithm>
#include <utility>
#include <thread>
#include <stdexcept>
#include <limits>
#include "self_bleu.h"
#include "counter.cpp"
#include "nltk.cpp"

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
}

SELF_BLEU_CPP::SELF_BLEU_CPP(vector<vector<string>> lines_of_tokens, float weights[],
                             int max_n, int smoothing_func, bool auto_reweight, SELF_BLEU_CPP *other_instance)
{
    this->n_cores = thread::hardware_concurrency();
    this->references = new vector<string> *[lines_of_tokens.size()];
    this->references_ngrams = new vector<string> **[max_n];
    for (int i = 0; i < max_n; i++)
        this->references_ngrams[i] = new vector<string> *[lines_of_tokens.size()];
    this->references_counts = new Counter **[max_n];
    for (int i = 0; i < max_n; i++)
        this->references_counts[i] = new Counter *[lines_of_tokens.size()];
    this->reference_max_counts = new map<string, int> *[max_n];
    this->reference_max2_counts = new map<string, int> *[max_n];

    this->ref_lens = new int[lines_of_tokens.size()];
    this->weights = weights;
    this->max_n = max_n;
    this->auto_reweigh = auto_reweigh;
    this->smoothing_function = smoothing_func;
    this->number_of_refs = (int)lines_of_tokens.size();

    cout << "self_bleu" << max_n << " init!" << endl;

    for (int i = 0; i < this->number_of_refs; i++)
    {
        this->references[i] = new vector<string>(lines_of_tokens[i]);
        this->ref_lens[i] = lines_of_tokens[i].size();
    }
    for (int n = 0; n < this->max_n; n++)
    {
        if (other_instance == NULL || n >= other_instance->max_n)
            for (int i = 0; i < this->number_of_refs; i++)
                this->references_ngrams[n][i] = get_ngrams(this->references[i], n + 1);
        else
            for (int i = 0; i < this->number_of_refs; i++)
                this->references_ngrams[n][i] = new vector<string>(*(other_instance->references_ngrams[n][i]));
    }
    for (int n = 0; n < this->max_n; n++)
    {
        if (other_instance == NULL || n >= other_instance->max_n)
            for (int i = 0; i < this->number_of_refs; i++)
                this->references_counts[n][i] = new Counter(this->references_ngrams[n][i]);
        else
            for (int i = 0; i < this->number_of_refs; i++)
                this->references_counts[n][i] = new Counter(other_instance->references_counts[n][i]);
    }
    for (int n = 0; n < max_n; n++)
    {
        if (other_instance == NULL || n >= other_instance->max_n)
        {
            this->reference_max_counts[n] = new map<string, int>();
            this->reference_max2_counts[n] = new map<string, int>();
            this->get_max_counts(n);
        }
        else
        {
            this->reference_max_counts[n] = new map<string, int>(*(other_instance->reference_max_counts[n]));
            this->reference_max2_counts[n] = new map<string, int>(*(other_instance->reference_max2_counts[n]));
        }
    }
}

void SELF_BLEU_CPP::get_max_counts(int n)
{
    vector<string> ngram_keys = vector<string>();
    for (int i = 0; i < this->number_of_refs; i++)
        for (string &ng : *(this->references_ngrams[n][i]))
            if (find(ngram_keys.begin(), ngram_keys.end(), ng) == ngram_keys.end())
                ngram_keys.push_back(ng);
    cout << n + 1 << "grams: " << ngram_keys.size() << endl;

    for (string &ng : ngram_keys)
    {
        (*reference_max_counts[n])[ng] = 0;
        (*reference_max2_counts[n])[ng] = 0;
    }

#pragma omp parallel
    {
#pragma omp for schedule(guided)
        for (int i = 0; i < (int)ngram_keys.size(); i++)
        {
            string &ng = ngram_keys[i];
            int *counts = new int[number_of_refs];
            for (int j = 0; j < number_of_refs; j++)
                counts[j] = references_counts[n][j]->get(ng, 0);
            int *max_value_ptr = max_element(counts, counts + number_of_refs);
            (*reference_max_counts[n])[ng] = *max_value_ptr;
            (*max_value_ptr) = -1;
            int max_value = *max_element(counts, counts + number_of_refs);
            (*reference_max2_counts[n])[ng] = max_value;
            delete[] counts;
        }
    }
}

void SELF_BLEU_CPP::get_score(double *results)
{
    if (results == NULL)
        throw invalid_argument("results ptr points to NULL!");
    cout << "calculating self_bleu" << max_n << " scores!" << endl;

    vector<string> *refs[number_of_refs - 1];
    int lens[number_of_refs - 1];
    map<string, int> *ref_max_counts[max_n];
    for (int n = 0; n < max_n; n++)
        ref_max_counts[n] = new map<string, int>(*(reference_max_counts[n]));

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
        for (int n = 0; n < max_n; n++)
            for (pair<string, int> const &p : *(references_counts[n][i]))
            {
                string const &ng = p.first;
                if ((*(reference_max_counts[n]))[ng] == (*(references_counts[n][i]))[ng])
                    (*(ref_max_counts[n]))[ng] = (*(reference_max2_counts[n]))[ng];
            }
        vector<string> *hypothesis = references[i];
        results[i] = corpus_bleu(number_of_refs, max_n,
                                 refs,
                                 hypothesis,
                                 ref_max_counts,
                                 lens,
                                 weights,
                                 smoothing_function,
                                 auto_reweigh);
        for (int n = 0; n < max_n; n++)
            for (pair<string, int> const &p : *(references_counts[n][i]))
            {
                string const &ng = p.first;
                (*(ref_max_counts[n]))[ng] = (*(reference_max_counts[n]))[ng];
            }
    }

    for (int n = 0; n < max_n; n++)
        delete ref_max_counts[n];
}