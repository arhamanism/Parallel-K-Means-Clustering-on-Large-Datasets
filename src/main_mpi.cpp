#include <mpi.h>

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <utility>
#include <vector>

#include "dataset.h"
#include "kmeans_mpi.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 7)
    {
        if (rank == 0)
            std::cerr << "Usage: mpirun -np P " << argv[0]
                      << " N D K MAX_ITERATIONS TOLERANCE SEED\n";
        MPI_Finalize();
        return 1;
    }

    const size_t N = std::strtoull(argv[1], nullptr, 10);
    const size_t D = std::strtoull(argv[2], nullptr, 10);
    const size_t K = std::strtoull(argv[3], nullptr, 10);
    const int max_iter = std::atoi(argv[4]);
    const double tol = std::atof(argv[5]);
    const unsigned int seed = static_cast<unsigned int>(std::atoi(argv[6]));

    // Only the root generates the dataset.
    std::vector<double> full;
    double gen_time = 0.0;
    if (rank == 0)
    {
        double g0 = MPI_Wtime();
        Dataset ds = generateDataset(N, D, K, seed);
        full = std::move(ds.data);
        gen_time = MPI_Wtime() - g0;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    KMeansResult res = runKMeansMPI(full, N, D, K, max_iter, tol, seed + 1);
    MPI_Barrier(MPI_COMM_WORLD);
    double elapsed = MPI_Wtime() - t0;

    if (rank == 0)
    {
        std::vector<size_t> sizes(K, 0);
        for (int a : res.assignments)
            sizes[a]++;

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "===== K-MEANS CLUSTERING (MPI) =====\n"
                  << "MPI processes: " << size << "\n"
                  << "Data points: " << N << "\nDimensions: " << D
                  << "\nClusters: " << K << "\nMaximum iterations: " << max_iter
                  << "\nTolerance: " << tol << "\nRandom seed: " << seed << "\n\n"
                  << "===== RESULTS =====\n"
                  << "Iterations: " << res.iterations << "\n"
                  << "Inertia (WCSS): " << res.inertia << "\n"
                  << "Dataset generation time: " << gen_time << " seconds\n"
                  << "K-Means execution time: " << elapsed << " seconds\n\n"
                  << "Cluster sizes:\n";
        for (size_t c = 0; c < K; ++c)
            std::cout << "Cluster " << c << ": " << sizes[c] << " points\n";
        std::cout << "\n===== CENTROIDS =====\n";
        for (size_t c = 0; c < K; ++c)
        {
            std::cout << "Cluster " << c << ": ";
            for (size_t j = 0; j < D; ++j)
                std::cout << res.centroids[c * D + j] << " ";
            std::cout << "\n";
        }
    }

    MPI_Finalize();
    return 0;
}