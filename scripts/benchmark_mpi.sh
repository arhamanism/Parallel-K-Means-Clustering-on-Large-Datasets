#!/usr/bin/env bash
# scripts/benchmark_mpi.sh
set -e
ARGS="500000 64 10 100 0.0001 42"
RUNS=5
mkdir -p results

for P in 1 2 4; do
  for i in $(seq 1 $RUNS); do
    mpirun -np $P ./build/kmeans_mpi $ARGS > results/mpi_p${P}_run$i.txt
  done
done

echo "Done. Outputs are in results/"
