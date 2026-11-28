
#include "kmeans.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include <omp.h>

KMeansResult runKMeans(
    const Dataset &dataset,
    size_t k,
    int max_iterations,
    double tolerance,
    unsigned int seed)
{
    size_t n = dataset.n;
    size_t d = dataset.d;

    if (n == 0 || d == 0 || k == 0 || k > n ||
        max_iterations <= 0 || tolerance < 0)
    {
        throw std::invalid_argument("Invalid K-Means parameters");
    }

    // Initializing centroids

    std::mt19937 generator(seed);

    std::vector<size_t> indices(n);

    std::iota(indices.begin(), indices.end(), 0);

    std::shuffle(indices.begin(), indices.end(), generator);

    std::vector<double> centroids(k * d);

    for (size_t cluster = 0; cluster < k; cluster++)
    {
        size_t point = indices[cluster];

        for (size_t j = 0; j < d; j++)
        {
            centroids[cluster * d + j] =
                dataset.data[point * d + j];
        }
    }

    std::vector<int> assignments(n, -1);

    std::vector<double> sums(k * d, 0.0);

    std::vector<size_t> counts(k, 0);

    // Preparing thread-private arrays

    int num_threads = omp_get_max_threads();

    // Each thread gets its own K * D sum array and K cluster counts.

    std::vector<double> thread_sums(
        static_cast<size_t>(num_threads) * k * d,
        0.0);

    std::vector<size_t> thread_counts(
        static_cast<size_t>(num_threads) * k,
        0);

    int iterations = 0;

    // Main K-Means loop

    for (int iter = 0; iter < max_iterations; iter++)
    {
        iterations++;

        // Resetting global accumulators
        std::fill(sums.begin(), sums.end(), 0.0);

        std::fill(counts.begin(), counts.end(), 0);

        // Resetting thread-private accumulators
        std::fill(thread_sums.begin(), thread_sums.end(), 0.0);

        std::fill(thread_counts.begin(), thread_counts.end(), 0);

        // Doing Parallel point assignment

#pragma omp parallel
        {
            int tid = omp_get_thread_num();

            size_t sum_offset =
                static_cast<size_t>(tid) * k * d;

            size_t count_offset =
                static_cast<size_t>(tid) * k;

#pragma omp for schedule(static)
            for (long long i = 0; i < static_cast<long long>(n); i++)
            {
                double min_distance =
                    std::numeric_limits<double>::max();

                int best_cluster = 0;

                // Finding nearest centroid
                for (size_t cluster = 0; cluster < k; cluster++)
                {
                    double distance = 0.0;

                    for (size_t j = 0; j < d; j++)
                    {
                        double diff =
                            dataset.data[static_cast<size_t>(i) * d + j] - centroids[cluster * d + j];

                        distance += diff * diff;
                    }

                    if (distance < min_distance)
                    {
                        min_distance = distance;

                        best_cluster =
                            static_cast<int>(cluster);
                    }
                }

                // Every iteration is writing to a unique assignment
                assignments[static_cast<size_t>(i)] = best_cluster;

                // Updating this thread's private count
                thread_counts[count_offset + best_cluster]++;

                // Updating this thread's private cluster sum
                for (size_t j = 0; j < d; j++)
                {
                    thread_sums[sum_offset +
                                static_cast<size_t>(best_cluster) * d + j] += dataset.data[static_cast<size_t>(i) * d + j];
                }
            }
        }

        // Combining thread accumulators

        for (int t = 0; t < num_threads; t++)
        {
            size_t sum_offset =
                static_cast<size_t>(t) * k * d;

            size_t count_offset =
                static_cast<size_t>(t) * k;

            for (size_t cluster = 0; cluster < k; cluster++)
            {
                counts[cluster] +=
                    thread_counts[count_offset + cluster];

                for (size_t j = 0; j < d; j++)
                {
                    sums[cluster * d + j] +=
                        thread_sums[sum_offset + cluster * d + j];
                }
            }
        }

        // Calculating new centroids

        std::vector<double> new_centroids = centroids;

        for (size_t cluster = 0; cluster < k; cluster++)
        {
            // Keep previous centroid if cluster is empty
            if (counts[cluster] == 0)
            {
                continue;
            }

            for (size_t j = 0; j < d; j++)
            {
                new_centroids[cluster * d + j] =
                    sums[cluster * d + j] / counts[cluster];
            }
        }

        // Checking convergence

        double movement_squared = 0.0;

        for (size_t i = 0; i < k * d; i++)
        {
            double diff =
                new_centroids[i] - centroids[i];

            movement_squared += diff * diff;
        }

        centroids = new_centroids;

        if (movement_squared <= tolerance * tolerance)
        {
            break;
        }
    }

    // Final assignments and inertia

    double inertia = 0.0;

#pragma omp parallel for reduction(+ : inertia) schedule(static)
    for (long long i = 0; i < static_cast<long long>(n); i++)
    {
        double min_distance =
            std::numeric_limits<double>::max();

        int best_cluster = 0;

        for (size_t cluster = 0; cluster < k; cluster++)
        {
            double distance = 0.0;

            for (size_t j = 0; j < d; j++)
            {
                double diff =
                    dataset.data[static_cast<size_t>(i) * d + j] - centroids[cluster * d + j];

                distance += diff * diff;
            }

            if (distance < min_distance)
            {
                min_distance = distance;

                best_cluster = static_cast<int>(cluster);
            }
        }

        assignments[static_cast<size_t>(i)] = best_cluster;

        // OpenMP combines each thread's local inertia
        inertia += min_distance;
    }

    // Return results

    KMeansResult result;

    result.assignments = assignments;
    result.centroids = centroids;
    result.iterations = iterations;
    result.inertia = inertia;

    return result;
}