#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <omp.h>
#include "gbm.hpp"

class LSMC {
public:
    static double priceAmericanPut(double S0, double K, double r, double sigma,
        double T, int N_steps, int N_paths, int seed = 1234);
};
