"""Benchmark keyword clustering pipeline across scaling tiers (N = 100, 1000, 5000).

Measures runtime, peak memory (MB), and unique keyword counts on deterministic
subsets (random_state=42) of the real dataset, and evaluates against targets:
  - N = 1,000: runtime <= 10 min (600s), peak memory <= 4 GB (4096 MB)
  - N = 5,000: runtime <= 30 min (1800s), peak memory <= 6 GB (6144 MB)

Reports system environment (OS, Python version, CPU, RAM) and logs measured
results to reports/performance_results.csv.
"""
import argparse
import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import platform
import sys
import tempfile
import time

import numpy as np
import pandas as pd

# Add repo root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import config
from run_pipeline import run
from src.data_loader import KeywordInputError, load_keywords
from src.preprocessing import prepare


TARGETS = {
    100: {"max_time_sec": 120.0, "max_mem_mb": 2048.0, "desc": "100 keywords sanity tier"},
    1000: {"max_time_sec": 600.0, "max_mem_mb": 4096.0, "desc": "1,000 keywords <= 10 min and 4 GB"},
    5000: {"max_time_sec": 1800.0, "max_mem_mb": 6144.0, "desc": "5,000 keywords <= 30 min and 6 GB"},
}


def get_system_hardware():
    """Retrieve OS, Python, CPU, and RAM metrics."""
    os_name = f"{platform.system()} {platform.release()} ({platform.version()})"
    py_ver = platform.python_version()
    cpu_cores = os.cpu_count() or 1
    cpu_info = f"{platform.processor() or 'CPU'} ({cpu_cores} logical cores)"

    total_ram_gb = None
    avail_ram_gb = None

    if platform.system() == "Windows":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        try:
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                total_ram_gb = round(stat.ullTotalPhys / (1024 ** 3), 2)
                avail_ram_gb = round(stat.ullAvailPhys / (1024 ** 3), 2)
        except Exception:
            pass
    else:
        try:
            total_ram_gb = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024 ** 3), 2)
        except Exception:
            pass

    return {
        "os": os_name,
        "python_version": py_ver,
        "cpu": cpu_info,
        "total_ram_gb": total_ram_gb,
        "avail_ram_gb": avail_ram_gb,
    }


def get_peak_memory_mb():
    """Read process peak working set / max RSS in MB."""
    if platform.system() == "Windows":
        class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]
        try:
            k32 = ctypes.windll.kernel32
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            h = k32.GetCurrentProcess()
            fn = getattr(k32, "K32GetProcessMemoryInfo", None) or ctypes.windll.psapi.GetProcessMemoryInfo
            fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX), wintypes.DWORD]
            fn.restype = wintypes.BOOL
            counters = PROCESS_MEMORY_COUNTERS_EX()
            counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
            if fn(h, ctypes.byref(counters), counters.cb):
                return round(counters.PeakWorkingSetSize / (1024 * 1024), 2)
        except Exception:
            pass

    try:
        import resource
        scale = 1024 * 1024 if sys.platform == "darwin" else 1024
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale, 2)
    except Exception:
        return None


def run_benchmark(input_path: Path, output_csv: Path, seed: int = 42):
    hardware = get_system_hardware()
    print("=" * 80)
    print("VYORAI KEYWORD CLUSTERING ENGINE - PERFORMANCE BENCHMARK")
    print("=" * 80)
    print(f"OS:               {hardware['os']}")
    print(f"Python Version:   {hardware['python_version']}")
    print(f"CPU:              {hardware['cpu']}")
    print(f"System RAM:       {hardware['total_ram_gb']} GB total ({hardware['avail_ram_gb']} GB available)")
    print(f"Input File:       {input_path}")
    print("=" * 80)

    if not input_path.exists():
        print(f"\n[ERROR] Input file does not exist: {input_path}")
        print("Real keywords dataset must be placed and prepared at data/input/real_keywords.csv.")
        print("Skipping all benchmark tiers.")
        return

    try:
        raw_df = load_keywords(input_path, config.KEYWORD_COLUMN)
        prep = prepare(raw_df)
    except KeywordInputError as e:
        print(f"\n[ERROR] Failed to load keywords from {input_path}: {e}")
        return

    unique_keywords = prep.unique["keyword_normalized"].tolist()
    total_unique = len(unique_keywords)
    print(f"Loaded {len(raw_df)} input rows -> {len(prep.records)} valid -> {total_unique} unique normalized keywords.")

    tiers = [100, 1000, 5000]
    results = []

    for n in tiers:
        print(f"\n--- Benchmark Tier: N = {n} unique keywords ---")
        tgt = TARGETS[n]

        # 1. Check if dataset has enough unique keywords
        if total_unique < n:
            msg = f"Real file has {total_unique} unique keywords, fewer than requested tier N={n}. Skipping tier (no synthetic padding)."
            print(f"[SKIP] {msg}")
            results.append({
                "tier_target_n": n,
                "actual_unique_keywords": total_unique,
                "status": "SKIPPED",
                "skip_reason": msg,
                "runtime_seconds": "NA",
                "peak_memory_mb": "NA",
                "target_max_seconds": tgt["max_time_sec"],
                "target_max_mem_mb": tgt["max_mem_mb"],
                "passed_targets": "NA",
                "os": hardware["os"],
                "python_version": hardware["python_version"],
                "cpu": hardware["cpu"],
                "total_ram_gb": hardware["total_ram_gb"],
                "avail_ram_gb": hardware["avail_ram_gb"],
            })
            continue

        # 2. For N = 5000: check if RAM allows
        if n == 5000:
            avail = hardware["avail_ram_gb"]
            total = hardware["total_ram_gb"]
            # 5,000 keywords requires at least ~2.5 GB available RAM for dense distance matrices & embeddings
            if avail is not None and avail < 2.5:
                msg = f"Available system RAM ({avail} GB) is below safe headroom threshold (2.5 GB) for 5,000 keywords. Skipping tier."
                print(f"[SKIP] {msg}")
                results.append({
                    "tier_target_n": n,
                    "actual_unique_keywords": n,
                    "status": "SKIPPED",
                    "skip_reason": msg,
                    "runtime_seconds": "NA",
                    "peak_memory_mb": "NA",
                    "target_max_seconds": tgt["max_time_sec"],
                    "target_max_mem_mb": tgt["max_mem_mb"],
                    "passed_targets": "NA",
                    "os": hardware["os"],
                    "python_version": hardware["python_version"],
                    "cpu": hardware["cpu"],
                    "total_ram_gb": hardware["total_ram_gb"],
                    "avail_ram_gb": hardware["avail_ram_gb"],
                })
                continue

        # 3. Deterministic sampling (random_state=42)
        rng = np.random.RandomState(seed)
        chosen_indices = np.sort(rng.choice(total_unique, size=n, replace=False))
        sample_subset = [unique_keywords[i] for i in chosen_indices]

        # 4. Run pipeline in temporary output directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            tmp_input = tmp_path / f"bench_{n}.csv"
            pd.DataFrame({"keyword": sample_subset}).to_csv(tmp_input, index=False)
            tmp_output = tmp_path / "output"
            tmp_reports = tmp_path / "reports"

            print(f"Executing full pipeline (baseline + semantic) on {n} unique keywords...")
            start_time = time.perf_counter()
            run(tmp_input, output_dir=tmp_output, reports_dir=tmp_reports, methods="both", verbose=False)
            elapsed_sec = round(time.perf_counter() - start_time, 3)
            peak_mb = get_peak_memory_mb()

            # Target checks
            passed_time = elapsed_sec <= tgt["max_time_sec"]
            passed_mem = (peak_mb <= tgt["max_mem_mb"]) if peak_mb is not None else True
            tier_passed = passed_time and passed_mem
            status_str = "PASS" if tier_passed else "FAIL"

            print(f"  Runtime:         {elapsed_sec:.2f} s (Target <= {tgt['max_time_sec']:.0f} s: {'PASS' if passed_time else 'FAIL'})")
            print(f"  Peak Memory:     {peak_mb} MB (Target <= {tgt['max_mem_mb']:.0f} MB: {'PASS' if passed_mem else 'FAIL'})")
            print(f"  Tier Result:     {status_str}")

            results.append({
                "tier_target_n": n,
                "actual_unique_keywords": n,
                "status": "COMPLETED",
                "skip_reason": "",
                "runtime_seconds": elapsed_sec,
                "peak_memory_mb": peak_mb if peak_mb is not None else "NA",
                "target_max_seconds": tgt["max_time_sec"],
                "target_max_mem_mb": tgt["max_mem_mb"],
                "passed_targets": status_str,
                "os": hardware["os"],
                "python_version": hardware["python_version"],
                "cpu": hardware["cpu"],
                "total_ram_gb": hardware["total_ram_gb"],
                "avail_ram_gb": hardware["avail_ram_gb"],
            })

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    res_df = pd.DataFrame(results)
    res_df.to_csv(output_csv, index=False)
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(res_df.to_string(index=False))
    print(f"\nResults saved to: {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Performance benchmarking across N=100, 1000, 5000 keywords.")
    parser.add_argument("--input", default=str(PROJECT_ROOT / "data" / "input" / "real_keywords.csv"),
                        help="Path to real keywords CSV (default: data/input/real_keywords.csv)")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "reports" / "performance_results.csv"),
                        help="Path to save performance_results.csv")
    args = parser.parse_args()

    run_benchmark(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()

