#!/usr/bin/env bash
# scripts/benchmark_openmp.sh
set -e
ARGS="500000 64 10 100 0.0001 42"
RUNS=5
mkdir -p results

for i in $(seq 1 $RUNS); do
  ./build/kmeans $ARGS > results/seq_run$i.txt
done

for T in 1 2 4; do
  for i in $(seq 1 $RUNS); do
    OMP_NUM_THREADS=$T ./build/kmeans_openmp $ARGS > results/omp_t${T}_run$i.txt
  done
done

echo "Done. Outputs are in results/"
