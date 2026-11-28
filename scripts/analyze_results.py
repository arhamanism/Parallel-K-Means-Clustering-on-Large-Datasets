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

configs = [("Sequential", 1, "results/seq_run*.txt")]
for T in (1, 2, 4, 6):
    configs.append((f"OpenMP", T, f"results/omp_t{T}_run*.txt"))

rows, t_seq = [], None
for name, threads, pattern in configs:
    times, inertias, iters = read_group(pattern)
    if not times:
        continue
    med = statistics.median(times)
    if name == "Sequential":
        t_seq = med
    rows.append((name, threads, len(times), min(times), med, max(times),
                 inertias, iters))

print(f"{'Implementation':<12}{'Threads':>8}{'Runs':>6}{'Min(s)':>9}"
      f"{'Median(s)':>11}{'Max(s)':>9}{'Speedup':>9}{'Eff(%)':>8}")
print("-" * 72)
csv = ["implementation,threads,runs,min,median,max,speedup,efficiency"]
for name, th, n, lo, med, hi, inertias, iters in rows:
    sp = t_seq / med if t_seq else float("nan")
    eff = sp / th * 100
    print(f"{name:<12}{th:>8}{n:>6}{lo:>9.2f}{med:>11.2f}{hi:>9.2f}"
          f"{sp:>9.2f}{eff:>8.1f}")
    csv.append(f"{name},{th},{n},{lo:.4f},{med:.4f},{hi:.4f},{sp:.4f},{eff:.2f}")

print("\nCorrectness check (inertia and iterations across all runs):")
for name, th, n, lo, med, hi, inertias, iters in rows:
    ok = "OK" if len(inertias) == 1 and len(iters) == 1 else "DIFFERS"
    print(f"  {name} {th} thread(s): {ok}  inertia={sorted(inertias)} iters={sorted(iters)}")

os.makedirs("results", exist_ok=True)
open("results/summary.csv", "w").write("\n".join(csv) + "\n")
print("\nSaved results/summary.csv")
