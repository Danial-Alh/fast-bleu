#include <limits>
#include <vector>
#include <string>
#include <cmath>
#include <utility>
#include <numeric>
#include <algorithm>
#include <tuple>
#include "custmap.h"
#include "counter.h"
#include "fraction.h"
#include "custmap.h"


using namespace std;

vector<string> *get_ngrams(vector<string> *tokens, int n)
{
    vector<string> *result = new vector<string>();
    if ((int)tokens->size() < n)
        return result;
    for (int i = 0; i < (int)tokens->size() - n + 1; i++)
    {
        string ngram = "";
        for (int j = 0; j < n; j++)
            ngram += tokens->at(i + j) + " ";
        ngram = ngram.substr(0, ngram.size() - 1);
        result->push_back(ngram);
    }
    return result;
}

void smooth_0(int size, Fraction *p_n)
{
    for (int i = 0; i < size; i++)
        if (p_n[i].numerator == 0)
        {
            p_n[i].numerator = 1;
            p_n[i].denominator = numeric_limits<long long>::max();
        }
}

void smooth_1(int size, Fraction *p_n)
{
    Fraction epsilon = Fraction(1, 10);
    for (int i = 0; i < size; i++)
        if (p_n[i].numerator == 0)
        {
            p_n[i].numerator = epsilon.numerator;
            p_n[i].denominator = p_n[i].denominator * epsilon.denominator;
        }
}

Fraction modified_precision(CustomMap **reference_max_counts,
                            vector<string> *hypothesis, int n)
{
    vector<string> *hyp_ngrams = get_ngrams(hypothesis, n);
    Counter counts = Counter(hyp_ngrams);
    CustomMap &max_counts = *(reference_max_counts[n - 1]);

    int numerator = 0;
    int denominator = 0;

    for (auto const &p : counts)
        denominator += counts.get(p.first);
    denominator = max(1, denominator);

    for (auto const &p : counts)
        numerator += min(counts.get(p.first), max_counts.get(p.first));

    delete hyp_ngrams;
    return Fraction(numerator, denominator);
}

int closest_ref_length(int num_refs, int *ref_lens, int hyp_len)
{
    tuple<int, int> *tmp = new tuple<int, int>[num_refs];
    for (int i = 0; i < num_refs; i++)
        tmp[i] = make_tuple(abs(ref_lens[i] - hyp_len), ref_lens[i]);
    tuple<int, int> closest_ref_len = *min_element(tmp, tmp + num_refs);
    int result = get<1>(closest_ref_len);
    delete[] tmp;
    return result;
}

double brevity_penalty(int closest_ref_len, int hyp_len)
{
    if (hyp_len > closest_ref_len)
        return 1;
    else if (hyp_len == 0)
        return 0;
    else
        return exp(1 - ((double)closest_ref_len) / hyp_len);
}

double corpus_bleu(int num_refs, int max_n,
                   __attribute__((unused)) vector<string> **references,
                   vector<string> *hypothesis,
                   CustomMap **reference_max_counts,
                   int *ref_lens,
                   vector<float> weights,
                   int smoothing_function,
                   bool auto_reweight)
{
    Fraction *p_n = new Fraction[max_n];
    // Key = ngram order
    // numerator value = no. of ngram matches.
    // denominator  value = no. of ngram in ref.

    int hyp_lengths = 0, ref_lengths = 0;
    void (*smoothing_functions[])(int, Fraction *) = {&smooth_0, &smooth_1};

    for (int i = 0; i < max_n; i++)
        p_n[i] = modified_precision(reference_max_counts, hypothesis, i + 1);

    int hyp_len = hypothesis->size();
    hyp_lengths += hyp_len;
    ref_lengths += closest_ref_length(num_refs, ref_lens, hyp_len);

    double bp = brevity_penalty(ref_lengths, hyp_lengths);

    float temp_weights[4] = {0.25};
    if (auto_reweight)
        if (hyp_lengths < 4 && equal(weights.begin(), weights.end(), temp_weights))
            for (int i = 0; i < max_n; i++)
                weights[i] = 1. / hyp_lengths;

    if (p_n[0].numerator == 0)
        return 0;

    if (smoothing_function < 0 || smoothing_function > 1)
        throw invalid_argument("smoothing 0 to 1 is only supported :)");

    smoothing_functions[smoothing_function](max_n, p_n);
    double s = 0.0;
    for (int i = 0; i < max_n; i++)
    {
        if (smoothing_function == 0 && p_n[i].numerator == 1 && p_n[i].denominator == numeric_limits<long long>::max())
            s += ((double)weights[i]) * log(numeric_limits<long double>::min());
        else
            s += ((double)weights[i]) * log(((long double)p_n[i].numerator) / ((long double)p_n[i].denominator));
    }

    // precise calculation???

    delete[] p_n;

    return bp * exp(s);
}