
#ifndef KMEANS_H
#define KMEANS_H

#include "dataset.h"

#include <vector>
#include <cstddef>

struct KMeansResult
{
    // Cluster assigned to each point
    std::vector<int> assignments;

    // Final centroid coordinates
    std::vector<double> centroids;

    // Number of iterations performed
    int iterations;

    // Inside the cluster sum of squared distances
    double inertia;
};

KMeansResult runKMeans(
    const Dataset &dataset,
    size_t k,
    int max_iterations,
    double tolerance,
    unsigned int seed);

#endif
