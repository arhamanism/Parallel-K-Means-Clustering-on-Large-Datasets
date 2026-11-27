
#include "dataset.h"
#include "kmeans.h"

#include <iostream>
#include <chrono>
#include <iomanip>
#include <stdexcept>

using namespace std;

int main(int argc, char *argv[])
{
    try
    {
        // Reading commandline arguments

        if (argc != 7)
        {
            cout << "Usage: ./kmeans N D K MAX_ITERATIONS "
                 << "TOLERANCE SEED\n";

            return 1;
        }

        size_t N = stoull(argv[1]);
        size_t D = stoull(argv[2]);
        size_t K = stoull(argv[3]);

        int max_iterations = stoi(argv[4]);

        double tolerance = stod(argv[5]);

        unsigned int seed =
            static_cast<unsigned int>(stoul(argv[6]));

        // Displaying configuration

        cout << "\n===== K-MEANS CLUSTERING =====\n";

        cout << "Data points: " << N << endl;
        cout << "Dimensions: " << D << endl;
        cout << "Clusters: " << K << endl;
        cout << "Maximum iterations: " << max_iterations << endl;
        cout << "Tolerance: " << tolerance << endl;
        cout << "Random seed: " << seed << endl;

        // Generating dataset

        cout << "\nGenerating dataset...\n";

        auto start = chrono::high_resolution_clock::now();

        Dataset dataset =
            generateDataset(N, D, K, seed);

        auto end = chrono::high_resolution_clock::now();

        double generation_time =
            chrono::duration<double>(end - start).count();

        cout << "Dataset generated successfully.\n";

        // Executing K-Means

        cout << "\nRunning K-Means...\n";

        start = chrono::high_resolution_clock::now();

        KMeansResult result =
            runKMeans(
                dataset,
                K,
                max_iterations,
                tolerance,
                seed + 1);

        end = chrono::high_resolution_clock::now();

        double execution_time =
            chrono::duration<double>(end - start).count();

        // Displaying results

        cout << "\n===== RESULTS =====\n";

        cout << "Iterations: "
             << result.iterations << endl;

        cout << "Inertia (WCSS): "
             << fixed << setprecision(4)
             << result.inertia << endl;

        cout << "Dataset generation time: "
             << generation_time << " seconds\n";

        cout << "K-Means execution time: "
             << execution_time << " seconds\n";

        cout << "\nCluster sizes:\n";

        vector<size_t> cluster_sizes(K, 0);

        for (int assignment : result.assignments)
        {
            cluster_sizes[assignment]++;
        }

        for (size_t i = 0; i < K; i++)
        {
            cout << "Cluster " << i
                 << ": " << cluster_sizes[i]
                 << " points\n";
        }

        cout << "\n===== CENTROIDS =====\n";

        for (size_t cluster = 0; cluster < K; cluster++)
        {
            cout << "Cluster " << cluster << ": ";

            for (size_t j = 0; j < D; j++)
            {
                cout << result.centroids[cluster * D + j]
                     << " ";
            }

            cout << endl;
        }
    }
    catch (const exception &e)
    {
        cerr << "Error: " << e.what() << endl;

        return 1;
    }

    return 0;
}
