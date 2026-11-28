#pragma once
#include <cstddef>
#include <vector>
#include "kmeans.h"

// full_data is only read on rank 0 (other ranks may pass an empty vector).
// result.assignments is only filled on rank 0; centroids, iterations and
// inertia are valid on every rank.
KMeansResult runKMeansMPI(const std::vector<double> &full_data,
                          size_t n, size_t d, size_t k,
                          int max_iterations, double tolerance,
                          unsigned int seed);