#pragma once
#include <cstddef>
#include <vector>
#include "kmeans.h"

// Hybrid MPI + OpenMP K-Means.
KMeansResult runKMeansHybrid(const std::vector<double> &full_data,
                             size_t n, size_t d, size_t k,
                             int max_iterations, double tolerance,
                             unsigned int seed);