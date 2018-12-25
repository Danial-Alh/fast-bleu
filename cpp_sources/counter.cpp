#include "counter.h"

using namespace std;

Counter::Counter()
{
}

Counter::Counter(vector<string> *all_items)
{
    for(vector<string>::iterator it = all_items->begin(); it != all_items->end(); ++it)
    {
        (*this)[*it] = this->get(*it, 0) + 1;
    }
}

int Counter::get(string item, int default_value)
{
    Counter::iterator key_iterator = this->find(item);
    if (key_iterator == this->end())
        return default_value;
    return key_iterator->second;
    
}