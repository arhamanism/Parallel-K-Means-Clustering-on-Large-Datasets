#!/usr/bin/env python3
# scripts/analyze_results.py
import glob, re, statistics, os

TIME_RE    = re.compile(r"K-Means execution time:\s*([\d.]+)")
INERTIA_RE = re.compile(r"Inertia \(WCSS\):\s*([\d.]+)")
ITER_RE    = re.compile(r"Iterations:\s*(\d+)")

def read_group(pattern):
    times, inertias, iters = [], set(), set()
    for path in sorted(glob.glob(pattern)):
        text = open(path).read()
        t = TIME_RE.search(text)
        if t:
            times.append(float(t.group(1)))
            inertias.add(INERTIA_RE.search(text).group(1))
            iters.add(ITER_RE.search(text).group(1))
    return times, inertias, iters

# (name, parallel units, file pattern)
configs = [("Sequential", 1, "results/seq_run*.txt")]
for T in (1, 2, 4, 6):
    configs.append(("OpenMP", T, f"results/omp_t{T}_run*.txt"))
for P in (1, 2, 4):
    configs.append(("MPI", P, f"results/mpi_p{P}_run*.txt"))
for P in (1, 2, 4):          # used in Phase 4; skipped if no files exist
    for T in (1, 2, 4):
        configs.append((f"Hybrid {P}x{T}", P * T,
                        f"results/hybrid_p{P}_t{T}_run*.txt"))

rows, t_seq = [], None
for name, units, pattern in configs:
    times, inertias, iters = read_group(pattern)
    if not times:
        continue
    med = statistics.median(times)
    if name == "Sequential":
        t_seq = med
    rows.append((name, units, len(times), min(times), med, max(times),
                 inertias, iters))

print(f"{'Implementation':<14}{'Units':>6}{'Runs':>6}{'Min(s)':>9}"
      f"{'Median(s)':>11}{'Max(s)':>9}{'Speedup':>9}{'Eff(%)':>8}")
print("-" * 72)
csv = ["implementation,units,runs,min,median,max,speedup,efficiency"]
for name, units, n, lo, med, hi, inertias, iters in rows:
    sp = t_seq / med if t_seq else float("nan")
    eff = sp / units * 100
    print(f"{name:<14}{units:>6}{n:>6}{lo:>9.2f}{med:>11.2f}{hi:>9.2f}"
          f"{sp:>9.2f}{eff:>8.1f}")
    csv.append(f"{name},{units},{n},{lo:.4f},{med:.4f},{hi:.4f},{sp:.4f},{eff:.2f}")

print("\nCorrectness check (inertia and iterations, all runs in each group):")
all_inertias = set()
for name, units, n, lo, med, hi, inertias, iters in rows:
    ok = "OK" if len(inertias) == 1 and len(iters) == 1 else "DIFFERS"
    all_inertias |= inertias
    print(f"  {name} ({units}): {ok}  inertia={sorted(inertias)} iters={sorted(iters)}")
vals = [float(x) for x in all_inertias]
agree = (max(vals) - min(vals)) / max(vals) < 1e-9
print("\nAll implementations agree (relative tolerance 1e-9):", "YES" if agree else "NO")

os.makedirs("results", exist_ok=True)
open("results/summary.csv", "w").write("\n".join(csv) + "\n")
print("Saved results/summary.csv")