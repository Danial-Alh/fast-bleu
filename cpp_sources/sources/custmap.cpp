#include "custmap.h"
#include <iostream>

CustomMap::CustomMap() : map<string, int>()
{
}

CustomMap::CustomMap(map<string, int> &obj) : map<string, int>(obj)
{
}

// const int& CustomMap::at (const string& key) const
// {
//     cout << "hereee" << endl;
//     try
//     {
//         return this->map<string, int>::at(key);
//     }
//     catch (const out_of_range &e)
//     {
//         return 0;
//     }
// }

// int &CustomMap::operator[](const string &k)
// {
//     return this->map<string, int>::operator[](k);
// }

// const int &CustomMap::operator[](const string &k) const
// {
//     cout << "hereee" << endl;
//     CustomMap::const_iterator res = this->find(k);
//     if (res != this->end())
//         return res->second;
//     return 0;
// }

int CustomMap::get(const string &k)
{
    CustomMap::const_iterator res = this->find(k);
    if (res != this->end())
        return res->second;
    return 0;
}

// int main(int argc, char const *argv[])
// {
//     map<string, int> b = map<string, int>();
//     b["b"] = 2;
//     b["b"] += 7;
//     CustomMap a = CustomMap(b);
//     a["a"] = 3;
//     a["a"] += 8;
//     cout << a["a"] << endl;
//     cout << a["b"] << endl;

//     return 0;
// }