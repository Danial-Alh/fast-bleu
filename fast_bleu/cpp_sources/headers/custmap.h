#ifndef CUSTMAP_H
#define CUSTMAP_H
#include <string>
#include <vector>
#include <map>
#include <stdexcept>

using namespace std;

class CustomMap : public map<string, int>
{
  public:
    CustomMap();
    CustomMap(map<string, int> &);
    // const int& at (const string& k) const;
    // int& operator[](const string& k) ;
    // const int& operator[](const string& k) const;
    int get(const string& k);
};

#endif