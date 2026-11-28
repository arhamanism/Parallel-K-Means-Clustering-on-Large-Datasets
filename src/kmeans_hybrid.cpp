#include "kmeans_hybrid.h"
#include <mpi.h>
#include <omp.h>
#include <algorithm>
#include <climits>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>

// Index of the nearest centroid to point p (squared Euclidean distance).
// Read only so it is safe to call from many threads.
static int nearestCentroid(const double *p, const std::vector<double> &centroids,
                           size_t k, size_t d)
{
    int best = 0;
    double best_dist = std::numeric_limits<double>::max();
    for (size_t c = 0; c < k; ++c)
    {
        double dist = 0.0;
        for (size_t j = 0; j < d; ++j)
        {
            double diff = p[j] - centroids[c * d + j];
            dist += diff * diff;
        }
        if (dist < best_dist)
        {
            best_dist = dist;
            best = static_cast<int>(c);
        }
    }
    return best;
}

KMeansResult runKMeansHybrid(const std::vector<double> &full_data,
                             size_t n, size_t d, size_t k,
                             int max_iterations, double tolerance,
                             unsigned int seed)
{
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (n * d > static_cast<size_t>(INT_MAX))
    {
        if (rank == 0)
            std::cerr << "N*D exceeds INT_MAX; MPI counts are int.\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // partitioning across MPI processes (uneven split)
    std::vector<int> point_counts(size), point_displs(size);
    std::vector<int> elem_counts(size), elem_displs(size);
    size_t base = n / size;
    size_t rem = n % size;
    size_t offset = 0;
    for (int r = 0; r < size; ++r)
    {
        size_t pts = base + (static_cast<size_t>(r) < rem ? 1 : 0);
        point_counts[r] = static_cast<int>(pts);
        point_displs[r] = static_cast<int>(offset);
        elem_counts[r] = static_cast<int>(pts * d);
        elem_displs[r] = static_cast<int>(offset * d);
        offset += pts;
    }

    const size_t local_n = static_cast<size_t>(point_counts[rank]);
    std::vector<double> local_data(local_n * d);

    MPI_Scatterv(rank == 0 ? full_data.data() : nullptr,
                 elem_counts.data(), elem_displs.data(), MPI_DOUBLE,
                 local_data.data(), static_cast<int>(local_n * d), MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    // initialization on root (same as sequential) then bcast
    std::vector<double> centroids(k * d);
    if (rank == 0)
    {
        std::mt19937 generator(seed);
        std::vector<size_t> indices(n);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), generator);

        for (size_t cluster = 0; cluster < k; cluster++)
        {
            size_t point = indices[cluster];
            for (size_t j = 0; j < d; j++)
                centroids[cluster * d + j] = full_data[point * d + j];
        }
    }
    MPI_Bcast(centroids.data(), static_cast<int>(k * d), MPI_DOUBLE, 0,
              MPI_COMM_WORLD);

    // Thread private accumulators (one block per OpenMP thread)
    const int T = omp_get_max_threads();
    // Pad each thread's counts to a multiple of 8 long longs (64 bytes) so
    // different threads never share a cache line.
    const size_t count_stride = ((k + 7) / 8) * 8;
    std::vector<double> thread_sums(static_cast<size_t>(T) * k * d);
    std::vector<long long> thread_counts(static_cast<size_t>(T) * count_stride);

    std::vector<int> local_assign(local_n);
    std::vector<double> local_sums(k * d), global_sums(k * d);
    std::vector<long long> local_counts(k), global_counts(k);
    std::vector<double> new_centroids(k * d);
    const double tol2 = tolerance * tolerance;
    int iterations = 0;

    // main loop
    for (int it = 0; it < max_iterations; ++it)
    {
        std::fill(thread_sums.begin(), thread_sums.end(), 0.0);
        std::fill(thread_counts.begin(), thread_counts.end(), 0LL);

// OpenMP threads assign points and accumulate privately
#pragma omp parallel num_threads(T)
        {
            const int t = omp_get_thread_num();
            double *my_sums = &thread_sums[static_cast<size_t>(t) * k * d];
            long long *my_counts = &thread_counts[static_cast<size_t>(t) * count_stride];

#pragma omp for schedule(static)
            for (long long i = 0; i < static_cast<long long>(local_n); ++i)
            {
                const double *p = &local_data[static_cast<size_t>(i) * d];
                int best = nearestCentroid(p, centroids, k, d);
                local_assign[i] = best;
                my_counts[best] += 1;
                for (size_t j = 0; j < d; ++j)
                    my_sums[static_cast<size_t>(best) * d + j] += p[j];
            }
        }

        // Combining thread private results into this process's totals
        std::fill(local_sums.begin(), local_sums.end(), 0.0);
        std::fill(local_counts.begin(), local_counts.end(), 0LL);
        for (int t = 0; t < T; ++t)
        {
            for (size_t idx = 0; idx < k * d; ++idx)
                local_sums[idx] += thread_sums[static_cast<size_t>(t) * k * d + idx];
            for (size_t c = 0; c < k; ++c)
                local_counts[c] += thread_counts[static_cast<size_t>(t) * count_stride + c];
        }

        // Combining across MPI processes; every rank gets same totals
        MPI_Allreduce(local_sums.data(), global_sums.data(),
                      static_cast<int>(k * d), MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD);
        MPI_Allreduce(local_counts.data(), global_counts.data(),
                      static_cast<int>(k), MPI_LONG_LONG, MPI_SUM,
                      MPI_COMM_WORLD);

        // every rank updates centroids and decides identically
        double movement = 0.0;
        for (size_t c = 0; c < k; ++c)
        {
            for (size_t j = 0; j < d; ++j)
            {
                double value = centroids[c * d + j]; // keep old if empty
                if (global_counts[c] > 0)
                    value = global_sums[c * d + j] /
                            static_cast<double>(global_counts[c]);
                double diff = value - centroids[c * d + j];
                movement += diff * diff;
                new_centroids[c * d + j] = value;
            }
        }
        centroids = new_centroids;
        iterations = it + 1;
        if (movement <= tol2)
            break;
    }

    // final assignment, inertia, gather
    double local_inertia = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : local_inertia) num_threads(T)
    for (long long i = 0; i < static_cast<long long>(local_n); ++i)
    {
        const double *p = &local_data[static_cast<size_t>(i) * d];
        int best = nearestCentroid(p, centroids, k, d);
        local_assign[i] = best;
        for (size_t j = 0; j < d; ++j)
        {
            double diff = p[j] - centroids[static_cast<size_t>(best) * d + j];
            local_inertia += diff * diff;
        }
    }
    double inertia = 0.0;
    MPI_Allreduce(&local_inertia, &inertia, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);

    KMeansResult result;
    if (rank == 0)
        result.assignments.resize(n);
    MPI_Gatherv(local_assign.data(), static_cast<int>(local_n), MPI_INT,
                rank == 0 ? result.assignments.data() : nullptr,
                point_counts.data(), point_displs.data(), MPI_INT, 0,
                MPI_COMM_WORLD);

    result.centroids = centroids;
    result.iterations = iterations;
    result.inertia = inertia;
    return result;
}