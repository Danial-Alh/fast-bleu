#ifndef COUNTER_H
#define COUNTER_H
#include <string>
#include <vector>
#include <map>

using namespace std;

class Counter : public map<string, int> {
    public:
        Counter();
        Counter(vector<string> *all_items);
        Counter(Counter *counter);
        int get(string item, int default_value);
};

#endif