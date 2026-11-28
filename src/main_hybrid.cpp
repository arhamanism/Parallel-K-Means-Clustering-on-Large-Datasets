#include <mpi.h>
#include <omp.h>

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <utility>
#include <vector>

#include "dataset.h"
#include "kmeans_hybrid.h"

int main(int argc, char **argv)
{
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (provided < MPI_THREAD_FUNNELED)
    {
        if (rank == 0)
            std::cerr << "MPI does not support MPI_THREAD_FUNNELED.\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (argc != 7)
    {
        if (rank == 0)
            std::cerr << "Usage: OMP_NUM_THREADS=T mpirun -np P --bind-to none "
                      << "-x OMP_NUM_THREADS " << argv[0]
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

    // Only the root generates the dataset (same seed as the other versions).
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
    // seed + 1 for clustering, exactly like main.cpp
    KMeansResult res = runKMeansHybrid(full, N, D, K, max_iter, tol, seed + 1);
    MPI_Barrier(MPI_COMM_WORLD);
    double elapsed = MPI_Wtime() - t0;

    if (rank == 0)
    {
        std::vector<size_t> sizes(K, 0);
        for (int a : res.assignments)
            sizes[a]++;

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "===== K-MEANS CLUSTERING (HYBRID MPI + OpenMP) =====\n"
                  << "MPI processes: " << size << "\n"
                  << "OpenMP threads per process: " << omp_get_max_threads() << "\n"
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