#include "counter.h"

using namespace std;

Counter::Counter() : CustomMap()
{
}

Counter::Counter(vector<string> *all_items) : CustomMap()
{
    for(vector<string>::iterator it = all_items->begin(); it != all_items->end(); ++it)
    {
        (*this)[*it] += 1;
    }
}