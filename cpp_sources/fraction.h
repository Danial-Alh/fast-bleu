#ifndef FRACTION_CPP_H
#define FRACTION_CPP_H
#include <iostream>

using namespace std;

class Fraction {
    private:
        long long gcd(long long, long long);

    public:
        long long numerator, denominator;

        Fraction();
        Fraction(long long, long long);

        operator int();
        operator float();
        operator double();
};

#endif