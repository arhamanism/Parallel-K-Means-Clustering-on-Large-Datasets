
#include "dataset.h"

#include <random>
#include <stdexcept>
#include <vector>

Dataset generateDataset(
    size_t n,
    size_t d,
    size_t k,
    unsigned int seed)
{
    if (n == 0 || d == 0 || k == 0 || k > n)
    {
        throw std::invalid_argument("Invalid dataset parameters");
    }

    Dataset dataset(n, d);

    std::mt19937 generator(seed);

    // Generate random cluster centers
    std::uniform_real_distribution<double> center_dist(-10.0, 10.0);

    // Generate noise around each center
    std::normal_distribution<double> noise_dist(0.0, 1.0);

    std::vector<double> centers(k * d);

    for (size_t i = 0; i < k * d; i++)
    {
        centers[i] = center_dist(generator);
    }

    // Assign each point to a randomly selected cluster
    std::uniform_int_distribution<size_t> cluster_dist(0, k - 1);

    for (size_t i = 0; i < n; i++)
    {
        size_t cluster = cluster_dist(generator);

        for (size_t j = 0; j < d; j++)
        {
            double center = centers[cluster * d + j];

            dataset.data[i * d + j] =
                center + noise_dist(generator);
        }
    }

    return dataset;
}
