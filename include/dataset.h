
#ifndef DATASET_H
#define DATASET_H

#include <vector>
#include <cstddef>

struct Dataset
{
    size_t n; // Number of data points
    size_t d; // Number of dimensions

    std::vector<double> data;

    Dataset(size_t points, size_t dimensions)
        : n(points), d(dimensions), data(points * dimensions)
    {
    }
};

// Generate synthetic data using Gaussian clusters
Dataset generateDataset(
    size_t n,
    size_t d,
    size_t k,
    unsigned int seed);

#endif
