#ifndef COUNTER_H
#define COUNTER_H
#include <string>
#include <vector>
#include "custmap.h"

using namespace std;

class Counter : public CustomMap
{
  public:
    Counter();
    Counter(vector<string> *all_items);
};

#endif