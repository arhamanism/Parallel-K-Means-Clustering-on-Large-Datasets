#!/usr/bin/env bash
# scripts/benchmark_hybrid.sh
set -e
ARGS="500000 64 10 100 0.0001 42"
RUNS=5
mkdir -p results

# "P:T" = P MPI processes x T OpenMP threads each (all use 4 workers)
for cfg in 1:4 2:2 4:1; do
  P=${cfg%%:*}
  T=${cfg##*:}
  for i in $(seq 1 $RUNS); do
    OMP_NUM_THREADS=$T mpirun -np $P --bind-to none -x OMP_NUM_THREADS \
      ./build/kmeans_hybrid $ARGS > results/hybrid_p${P}_t${T}_run$i.txt
  done
done

echo "Done. Outputs are in results/"