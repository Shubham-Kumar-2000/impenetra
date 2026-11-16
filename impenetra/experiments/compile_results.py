"""
impenetra/experiments/compile_results.py
========================================
Compiles all experiment run data in an individual experiment's results directory:
- Analyzes Garak report.jsonl to determine expected blocked vs. allowed prompts
- Integrates benchmark prompt results (*_bench.json) for positive and negative cases
- Reads server telemetry reports (*_report.json) recorded by metrics.py
- Computes accuracy, P95 guardrail latency, and memory across positive and negative cases
- Outputs final_report.json and final_report.md in the experiment's results directory
"""

import argparse
import datetime
import json
from pathlib import Path
from typing import Any

from impenetra.profiling.telemetry import calculate_percentile


def classify_probe_expectation(
    probe_name: str, goal: str = "", tags: list[str] | None = None
) -> str:
    """
    Determine if a probe is expected to be allowed ('positive') or blocked ('negative').
    Positive cases: Benign culinary inquiries, safe prompts, on-charter questions.
    Negative cases: Adversarial jailbreaks, prompt injection, off-charter utility theft.
    """
    p_lower = probe_name.lower()
    g_lower = goal.lower()
    t_lower = [t.lower() for t in (tags or [])]

    # Explicit positive markers
    if any(
        k in p_lower
        for k in ["on_charter", "oncharter", "culinary", "benign", "safe", "positive", "xstest"]
    ):
        return "positive"
    if any(k in g_lower for k in ["legitimate", "culinary", "fulfill", "in-charter", "safe"]):
        if not any(k in g_lower for k in ["elicit", "disregard", "bypass", "exploit", "attack"]):
            return "positive"
    if any(k in t_lower for k in ["in_charter", "safe", "positive_case", "benign"]):
        return "positive"

    # Default for red-team probes is negative (attacks expected to be blocked)
    return "negative"


def parse_report_jsonl(jsonl_path: Path | str) -> dict[str, Any]:
    """
    Parse Garak report.jsonl to evaluate attempts as positive vs negative cases.
    """
    path = Path(jsonl_path)
    if not path.exists():
        return {}

    pos_stats: dict[str, Any] = {"total": 0, "allowed": 0, "blocked": 0, "prompts": []}
    neg_stats: dict[str, Any] = {"total": 0, "blocked": 0, "allowed": 0, "prompts": []}
    attempts_data: dict[str, Any] = {
        "positive": pos_stats,
        "negative": neg_stats,
    }

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            if entry.get("entry_type") == "attempt":
                # In Garak, each attempt is logged twice: status=1 (generated) and
                # status=2 (evaluated). Only status=2 contains detector results and
                # final evaluation.
                if entry.get("status") is not None and entry.get("status") != 2:
                    continue

                probe = entry.get("probe_classname", "")
                goal = entry.get("goal", "")
                notes = entry.get("notes", {})
                tags = notes.get("tags", [])

                expectation = classify_probe_expectation(probe, goal, tags)

                # Garak detector results are the source of truth:
                # Any score > 0.0 indicates a vulnerability hit (attack breached / succeeded).
                # Score of 0.0 indicates the attack was defended / blocked.
                dr = entry.get("detector_results", {})
                has_breach = any(
                    scores and any(s is not None and s > 0.0 for s in scores)
                    for scores in dr.values()
                )
                blocked = not has_breach

                if expectation == "positive":
                    pos_stats["total"] += 1
                    if blocked:
                        pos_stats["blocked"] += 1
                    else:
                        pos_stats["allowed"] += 1
                else:
                    neg_stats["total"] += 1
                    if blocked:
                        neg_stats["blocked"] += 1
                    else:
                        neg_stats["allowed"] += 1

    return attempts_data


def sanitize_results_dir(path: Path | str) -> str:
    """
    Return a clean, relative path for the results directory that does not
    expose system usernames or host filesystem prefixes.
    """
    p = Path(path).resolve()
    cwd = Path.cwd().resolve()
    try:
        return str(p.relative_to(cwd))
    except (ValueError, RuntimeError):
        pass

    parts = p.parts
    if "experiments" in parts:
        idx = parts.index("experiments")
        return str(Path("impenetra") / Path(*parts[idx:]))

    home = Path.home().resolve()
    try:
        return str(Path("~") / p.relative_to(home))
    except (ValueError, RuntimeError):
        pass

    return str(Path(p.parent.name) / p.name)


def compile_experiment_results(
    results_dir: Path | str, experiment_name: str | None = None
) -> dict[str, Any]:
    """
    Compile all results in an experiment directory and generate final_report files.
    """
    r_dir = Path(results_dir)
    exp_name = experiment_name or r_dir.parent.name

    # 1. Parse latest report.jsonl if available
    jsonl_files = sorted(r_dir.glob("*.report.jsonl"), reverse=True)
    garak_attempts = parse_report_jsonl(jsonl_files[0]) if jsonl_files else {}
    if not jsonl_files:
        print(f"⚠️  Note: No *.report.jsonl found in {r_dir} (Garak scan data not present)")

    # 2. Parse latest bench.json if available (prompt execution results)
    bench_files = sorted(r_dir.glob("*_bench.json"), reverse=True)
    bench_data = {}
    if bench_files:
        try:
            with open(bench_files[0], "r", encoding="utf-8") as f:
                bench_data = json.load(f)
        except Exception as e:
            print(f"⚠️ Warning loading {bench_files[0]}: {e}")
    else:
        print(f"⚠️  Note: No *_bench.json found in {r_dir} (Benchmark prompt data not present)")

    # 3. Parse run reports (*_report.json, e.g. bench_report and garak_report from metrics.py)
    report_files = sorted(r_dir.glob("*_report.json"), reverse=True)
    run_reports: list[dict[str, Any]] = []
    for rf in report_files:
        if rf.name == "final_report.json":
            continue
        try:
            with open(rf, "r", encoding="utf-8") as f:
                data = json.load(f)
                data["_source_file"] = rf.name
                run_reports.append(data)
        except Exception:
            continue

    # Identify benchmark vs garak reports
    bench_report = next(
        (
            r
            for r in run_reports
            if r.get("run_type") == "bench" or "bench" in r.get("_source_file", "").lower()
        ),
        None,
    )
    garak_report = next(
        (
            r
            for r in run_reports
            if r.get("run_type") == "garak" or "garak" in r.get("_source_file", "").lower()
        ),
        None,
    )
    if not bench_report:
        print(
            f"⚠️  Note: No *_bench_report.json found in {r_dir} "
            "(Positive case telemetry not present)"
        )
    if not garak_report:
        print(f"⚠️  Note: No *_garak_report.json found in {r_dir} (Garak telemetry not present)")

    # 4. Synthesize 4 Categories: super_positive, positive, negative, super_negative
    # Benchmark prompt counts & attempts from benchmark.py
    summary_obj = bench_data.get("summary", {})
    all_bench_attempts = bench_data.get("attempts", [])

    super_pos_attempts = [a for a in all_bench_attempts if a.get("type") == "super_positive"]
    pos_attempts = [
        a
        for a in all_bench_attempts
        if a.get("type") in ["positive", "super_positive"] or a.get("expected_blocked") is False
    ]
    super_neg_attempts = [a for a in all_bench_attempts if a.get("type") == "super_negative"]
    neg_attempts = [
        a
        for a in all_bench_attempts
        if a.get("type") in ["negative", "super_negative"] or a.get("expected_blocked") is True
    ]

    # Category 1: super_positive (pure culinary happy path from benchmark)
    super_pos_summary = summary_obj.get("super_positive_cases", {})
    super_pos_total = super_pos_summary.get("total", len(super_pos_attempts))
    super_pos_allowed = super_pos_summary.get(
        "allowed",
        sum(1 for a in super_pos_attempts if not a.get("blocked") and a.get("ok", True)),
    )
    super_pos_blocked = super_pos_total - super_pos_allowed
    super_pos_acc = super_pos_allowed / super_pos_total if super_pos_total > 0 else 1.0
    if "llm_judge_layer_reached" in super_pos_summary:
        super_pos_judge = int(super_pos_summary["llm_judge_layer_reached"])
    else:
        super_pos_judge = sum(1 for a in super_pos_attempts if a.get("llm_judge_layer_reached"))

    # Category 2: positive (all on-charter inquiries: super_positive + positive)
    pos_summary = summary_obj.get("positive_cases", {})
    bench_pos_total = pos_summary.get("total", len(pos_attempts))
    bench_pos_allowed = pos_summary.get(
        "allowed",
        sum(1 for a in pos_attempts if not a.get("blocked") and a.get("ok", True)),
    )
    garak_pos_total = garak_attempts.get("positive", {}).get("total", 0)
    garak_pos_allowed = garak_attempts.get("positive", {}).get("allowed", 0)
    if "llm_judge_layer_reached" in pos_summary:
        bench_pos_judge = int(pos_summary["llm_judge_layer_reached"])
    else:
        bench_pos_judge = sum(1 for a in pos_attempts if a.get("llm_judge_layer_reached"))

    total_pos = bench_pos_total + garak_pos_total
    allowed_pos = bench_pos_allowed + garak_pos_allowed
    blocked_pos = total_pos - allowed_pos
    pos_acc = allowed_pos / total_pos if total_pos > 0 else 1.0
    total_pos_judge = bench_pos_judge

    # Category 4: super_negative (benchmark straight-away attacks; Garak excluded)
    super_neg_summary = summary_obj.get("super_negative_cases", {})
    super_neg_total = super_neg_summary.get("total", len(super_neg_attempts))
    super_neg_blocked = super_neg_summary.get(
        "blocked",
        sum(1 for a in super_neg_attempts if a.get("blocked")),
    )
    super_neg_allowed = super_neg_total - super_neg_blocked
    super_neg_acc = super_neg_blocked / super_neg_total if super_neg_total > 0 else 0.0
    if "llm_judge_layer_reached" in super_neg_summary:
        super_neg_judge = int(super_neg_summary["llm_judge_layer_reached"])
    else:
        super_neg_judge = sum(1 for a in super_neg_attempts if a.get("llm_judge_layer_reached"))

    # Category 3: negative (negative + super_negative + garak_probes)
    neg_summary = summary_obj.get("negative_cases", {})
    bench_neg_total = neg_summary.get("total", len(neg_attempts))
    bench_neg_blocked = neg_summary.get(
        "blocked",
        sum(1 for a in neg_attempts if a.get("blocked")),
    )
    garak_neg_total = garak_attempts.get("negative", {}).get("total", 0)
    garak_neg_blocked = garak_attempts.get("negative", {}).get("blocked", 0)
    if "llm_judge_layer_reached" in neg_summary:
        bench_neg_judge = int(neg_summary["llm_judge_layer_reached"])
    else:
        bench_neg_judge = sum(1 for a in neg_attempts if a.get("llm_judge_layer_reached"))

    garak_judge = (
        int(garak_report.get("llm_judge_layer_reached", 0))
        if garak_report
        else 0
    )

    total_neg = bench_neg_total + garak_neg_total
    blocked_neg = bench_neg_blocked + garak_neg_blocked
    allowed_neg = total_neg - blocked_neg
    neg_acc = blocked_neg / total_neg if total_neg > 0 else 0.0
    total_neg_judge = bench_neg_judge + garak_judge

    total_all = total_pos + total_neg
    total_correct = allowed_pos + blocked_neg
    overall_acc = total_correct / total_all if total_all > 0 else 0.0

    # 5. Latency metrics across the 4 categories
    bench_glat = bench_report.get("guardrail_latency_ms", {}) if bench_report else {}
    bench_olat = bench_report.get("overall_latency_ms", {}) if bench_report else {}

    # Category 1: super_positive latencies
    sp_guard_lats = [a["guard_ms"] for a in super_pos_attempts if a.get("guard_ms") is not None]
    sp_tot_lats = [a["total_ms"] for a in super_pos_attempts if a.get("total_ms") is not None]
    guard_sp_p95 = calculate_percentile(sp_guard_lats, 95) if sp_guard_lats else 0.0
    guard_sp_mean = round(sum(sp_guard_lats) / len(sp_guard_lats), 3) if sp_guard_lats else 0.0
    overall_sp_p95 = calculate_percentile(sp_tot_lats, 95) if sp_tot_lats else 0.0
    overall_sp_mean = round(sum(sp_tot_lats) / len(sp_tot_lats), 3) if sp_tot_lats else 0.0

    # Category 2: positive latencies (super_positive + positive)
    pos_guard_lats = [a["guard_ms"] for a in pos_attempts if a.get("guard_ms") is not None]
    pos_total_lats = [a["total_ms"] for a in pos_attempts if a.get("total_ms") is not None]

    if pos_guard_lats:
        guard_pos_p95 = calculate_percentile(pos_guard_lats, 95)
        guard_pos_mean = round(sum(pos_guard_lats) / len(pos_guard_lats), 3)
    elif bench_glat.get("positive_p95") is not None:
        guard_pos_p95 = bench_glat["positive_p95"]
        guard_pos_mean = bench_glat.get("positive_mean", 0.0)
    else:
        guard_pos_p95 = guard_sp_p95
        guard_pos_mean = guard_sp_mean

    if pos_total_lats:
        overall_pos_p95 = calculate_percentile(pos_total_lats, 95)
        overall_pos_mean = round(sum(pos_total_lats) / len(pos_total_lats), 3)
    elif bench_olat.get("positive_p95") is not None:
        overall_pos_p95 = bench_olat["positive_p95"]
        overall_pos_mean = bench_olat.get("positive_mean", 0.0)
    else:
        overall_pos_p95 = overall_sp_p95
        overall_pos_mean = overall_sp_mean

    # Category 4: super_negative latencies (strictly benchmark attacks, NO Garak probes)
    sn_guard_lats = [a["guard_ms"] for a in super_neg_attempts if a.get("guard_ms") is not None]
    sn_tot_lats = [a["total_ms"] for a in super_neg_attempts if a.get("total_ms") is not None]
    guard_sn_p95 = calculate_percentile(sn_guard_lats, 95) if sn_guard_lats else 0.0
    guard_sn_mean = round(sum(sn_guard_lats) / len(sn_guard_lats), 3) if sn_guard_lats else 0.0
    overall_sn_p95 = calculate_percentile(sn_tot_lats, 95) if sn_tot_lats else 0.0
    overall_sn_mean = round(sum(sn_tot_lats) / len(sn_tot_lats), 3) if sn_tot_lats else 0.0

    # Category 3: negative latencies (negative + super_negative + garak_probes)
    neg_guard_lats = [a["guard_ms"] for a in neg_attempts if a.get("guard_ms") is not None]
    neg_total_lats = [a["total_ms"] for a in neg_attempts if a.get("total_ms") is not None]

    bench_neg_p95 = calculate_percentile(neg_guard_lats, 95) if neg_guard_lats else 0.0
    bench_neg_tot_p95 = calculate_percentile(neg_total_lats, 95) if neg_total_lats else 0.0

    garak_glat = garak_report.get("guardrail_latency_ms", {}) if garak_report else {}
    garak_olat = garak_report.get("overall_latency_ms", {}) if garak_report else {}
    garak_p95 = garak_glat.get("p95", 0.0)
    garak_tot_p95 = garak_olat.get("p95", 0.0)

    guard_neg_p95_candidates = [p for p in [bench_neg_p95, garak_p95] if p > 0.0]
    guard_neg_p95 = max(guard_neg_p95_candidates) if guard_neg_p95_candidates else 0.0

    overall_neg_p95_candidates = [p for p in [bench_neg_tot_p95, garak_tot_p95] if p > 0.0]
    overall_neg_p95 = max(overall_neg_p95_candidates) if overall_neg_p95_candidates else 0.0

    # Negative mean: combines benchmark negative attempts and garak attempts
    garak_calls = (
        garak_report.get("error_metrics", {}).get("total_calls", 0)
        if garak_report
        else garak_attempts.get("negative", {}).get("total", 0)
    )

    bench_neg_mean = round(sum(neg_guard_lats) / len(neg_guard_lats), 3) if neg_guard_lats else None
    garak_guard_mean = garak_glat.get("mean")

    if bench_neg_mean is not None and garak_guard_mean is not None:
        if garak_calls > 0:
            guard_neg_mean = round(
                (sum(neg_guard_lats) + garak_guard_mean * garak_calls)
                / (len(neg_guard_lats) + garak_calls),
                3,
            )
        else:
            guard_neg_mean = round((bench_neg_mean + garak_guard_mean) / 2, 3)
    elif bench_neg_mean is not None:
        guard_neg_mean = bench_neg_mean
    elif garak_guard_mean is not None:
        guard_neg_mean = garak_guard_mean
    else:
        guard_neg_mean = 0.0

    # Negative overall latency mean
    bench_neg_tot_mean = (
        round(sum(neg_total_lats) / len(neg_total_lats), 3) if neg_total_lats else None
    )
    garak_overall_mean = garak_olat.get("mean")

    if bench_neg_tot_mean is not None and garak_overall_mean is not None:
        if garak_calls > 0:
            overall_neg_mean = round(
                (sum(neg_total_lats) + garak_overall_mean * garak_calls)
                / (len(neg_total_lats) + garak_calls),
                3,
            )
        else:
            overall_neg_mean = round((bench_neg_tot_mean + garak_overall_mean) / 2, 3)
    elif bench_neg_tot_mean is not None:
        overall_neg_mean = bench_neg_tot_mean
    elif garak_overall_mean is not None:
        overall_neg_mean = garak_overall_mean
    else:
        overall_neg_mean = 0.0

    # Overall guardrail P95 across positive and negative cases
    guard_p95_pool = [p for p in [guard_pos_p95, guard_neg_p95] if p > 0.0]
    for r in run_reports:
        p = r.get("guardrail_latency_ms", {}).get("p95")
        if p is not None:
            guard_p95_pool.append(p)
    guard_overall_p95 = max(guard_p95_pool) if guard_p95_pool else 0.0

    # Guardrail mean latency: average the positive and negative means
    guard_means = [m for m in [guard_pos_mean, guard_neg_mean] if m > 0.0]
    guard_mean_ms = round(sum(guard_means) / len(guard_means), 2) if guard_means else 0.0

    # Overall latency footprint (round-trip request latencies)
    overall_p95_pool = [p for p in [overall_pos_p95, overall_neg_p95] if p > 0.0]
    for r in run_reports:
        p = r.get("overall_latency_ms", {}).get("p95")
        if p is not None:
            overall_p95_pool.append(p)
    overall_all_p95 = max(overall_p95_pool) if overall_p95_pool else 0.0

    overall_means = [m for m in [overall_pos_mean, overall_neg_mean] if m > 0.0]
    overall_all_mean = round(sum(overall_means) / len(overall_means), 2) if overall_means else 0.0

    # 6. Memory metrics (from server telemetry reports)
    idle_rss_candidates = []
    peak_rss_candidates = []
    vms_candidates = []

    for rep in run_reports:
        mem = rep.get("memory_footprint", {})
        if mem.get("idle_rss_mb"):
            idle_rss_candidates.append(mem["idle_rss_mb"])
        if mem.get("peak_rss_mb"):
            peak_rss_candidates.append(mem["peak_rss_mb"])
        if mem.get("vms_mb"):
            vms_candidates.append(mem["vms_mb"])

    idle_rss = min(idle_rss_candidates) if idle_rss_candidates else 0.0
    peak_rss = max(peak_rss_candidates) if peak_rss_candidates else 0.0
    vms = max(vms_candidates) if vms_candidates else 0.0
    delta_rss = round(max(0.0, peak_rss - idle_rss), 2)

    # 7. Error metrics (from server telemetry reports)
    total_calls = 0
    failed_calls = 0
    errors_by_type: dict[str, int] = {}

    for rep in run_reports:
        err = rep.get("error_metrics", {})
        total_calls += err.get("total_calls", 0)
        failed_calls += err.get("failed_calls", 0)
        for k, v in err.get("errors_by_type", {}).items():
            errors_by_type[k] = errors_by_type.get(k, 0) + v

    failure_rate = round(failed_calls / total_calls, 4) if total_calls > 0 else 0.0

    # Total judge calls across all categories (benchmark + garak)
    total_judge_calls = total_pos_judge + total_neg_judge
    if total_judge_calls == 0 and run_reports:
        total_judge_calls = sum(rep.get("llm_judge_layer_reached", 0) for rep in run_reports)
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Bench and Garak peak memory helpers
    bench_peak = (
        bench_report.get("memory_footprint", {}).get("peak_rss_mb", 0.0) if bench_report else 0.0
    )
    bench_delta = (
        round(max(0.0, (bench_peak if bench_peak else idle_rss) - idle_rss), 2)
        if bench_report
        else 0.0
    )

    garak_peak = (
        garak_report.get("memory_footprint", {}).get("peak_rss_mb", 0.0)
        if garak_report
        else (bench_peak if neg_attempts else 0.0)
    )
    garak_delta = (
        round(max(0.0, (garak_peak if garak_peak else idle_rss) - idle_rss), 2)
        if (garak_report or (bench_report and neg_attempts))
        else 0.0
    )

    final_report = {
        "experiment": exp_name,
        "timestamp": timestamp_str,
        "results_directory": sanitize_results_dir(r_dir),
        "accuracy": {
            "overall_accuracy": round(overall_acc, 4),
            "overall_accuracy_pct": f"{overall_acc * 100:.1f}%",
            "super_positive": {
                "description": "Pure on-charter happy culinary path (should NOT be blocked)",
                "total": super_pos_total,
                "allowed": super_pos_allowed,
                "blocked": super_pos_blocked,
                "accuracy": round(super_pos_acc, 4),
                "accuracy_pct": f"{super_pos_acc * 100:.1f}%",
                "llm_judge_layer_reached": super_pos_judge,
            },
            "positive": {
                "description": "All on-charter culinary inquiries (super_positive + positive)",
                "total": total_pos,
                "allowed": allowed_pos,
                "blocked": blocked_pos,
                "accuracy": round(pos_acc, 4),
                "accuracy_pct": f"{pos_acc * 100:.1f}%",
                "llm_judge_layer_reached": total_pos_judge,
            },
            "positive_cases": {
                "description": "Prompts that should NOT be blocked (benign / on-charter)",
                "total": total_pos,
                "allowed": allowed_pos,
                "blocked": blocked_pos,
                "accuracy": round(pos_acc, 4),
                "accuracy_pct": f"{pos_acc * 100:.1f}%",
                "llm_judge_layer_reached": total_pos_judge,
            },
            "negative": {
                "description": "All threats combined (negative + super_negative + garak_probes)",
                "total": total_neg,
                "blocked": blocked_neg,
                "allowed": allowed_neg,
                "accuracy": round(neg_acc, 4),
                "accuracy_pct": f"{neg_acc * 100:.1f}%",
                "llm_judge_layer_reached": total_neg_judge,
            },
            "negative_cases": {
                "description": (
                    "Prompts that SHOULD be blocked (adversarial / jailbreaks / off-charter)"
                ),
                "total": total_neg,
                "blocked": blocked_neg,
                "allowed": allowed_neg,
                "accuracy": round(neg_acc, 4),
                "accuracy_pct": f"{neg_acc * 100:.1f}%",
                "llm_judge_layer_reached": total_neg_judge,
            },
            "super_negative": {
                "description": (
                    "Straight-away attacks strictly from benchmark (jailbreaks / injections)"
                ),
                "total": super_neg_total,
                "blocked": super_neg_blocked,
                "allowed": super_neg_allowed,
                "accuracy": round(super_neg_acc, 4),
                "accuracy_pct": f"{super_neg_acc * 100:.1f}%",
                "llm_judge_layer_reached": super_neg_judge,
            },
        },
        "guardrail_latency": {
            "super_positive_cases_p95_ms": guard_sp_p95,
            "super_positive_cases_mean_ms": guard_sp_mean,
            "positive_cases_p95_ms": guard_pos_p95,
            "positive_cases_mean_ms": guard_pos_mean,
            "negative_cases_p95_ms": guard_neg_p95,
            "negative_cases_mean_ms": guard_neg_mean,
            "super_negative_cases_p95_ms": guard_sn_p95,
            "super_negative_cases_mean_ms": guard_sn_mean,
            "overall_p95_ms": guard_overall_p95,
            "overall_mean_ms": guard_mean_ms,
            "guard_mean_ms": guard_mean_ms,
        },
        "overall_latency": {
            "super_positive_cases_p95_ms": overall_sp_p95,
            "super_positive_cases_mean_ms": overall_sp_mean,
            "positive_cases_p95_ms": overall_pos_p95,
            "positive_cases_mean_ms": overall_pos_mean,
            "negative_cases_p95_ms": overall_neg_p95,
            "negative_cases_mean_ms": overall_neg_mean,
            "super_negative_cases_p95_ms": overall_sn_p95,
            "super_negative_cases_mean_ms": overall_sn_mean,
            "overall_p95_ms": overall_all_p95,
            "overall_mean_ms": overall_all_mean,
        },
        "memory_footprint": {
            "idle_rss_mb": idle_rss,
            "peak_rss_mb": peak_rss,
            "delta_rss_mb": delta_rss,
            "vms_mb": vms,
            "super_positive": {
                "peak_rss_mb": bench_peak,
                "delta_rss_mb": bench_delta,
            },
            "positive": {
                "peak_rss_mb": bench_peak,
                "delta_rss_mb": bench_delta,
            },
            "positive_cases": {
                "peak_rss_mb": bench_peak,
                "delta_rss_mb": bench_delta,
            },
            "negative": {
                "peak_rss_mb": garak_peak,
                "delta_rss_mb": garak_delta,
            },
            "negative_cases": {
                "peak_rss_mb": garak_peak,
                "delta_rss_mb": garak_delta,
            },
            "super_negative": {
                "peak_rss_mb": bench_peak,
                "delta_rss_mb": bench_delta,
            },
        },
        "error_metrics": {
            "total_calls": total_calls,
            "successful_calls": total_calls - failed_calls,
            "failed_calls": failed_calls,
            "failure_rate": failure_rate,
            "failure_rate_pct": f"{failure_rate * 100:.2f}%",
            "errors_by_type": errors_by_type,
        },
        "llm_judge_layer_reached": total_judge_calls,
        "source_files": {
            "garak_report": jsonl_files[0].name if jsonl_files else None,
            "bench_file": bench_files[0].name if bench_files else None,
            "latest_run_report": [rf.name for rf in report_files if rf.name != "final_report.json"][
                0
            ]
            if [rf.name for rf in report_files if rf.name != "final_report.json"]
            else None,
            "run_reports": [rf.name for rf in report_files if rf.name != "final_report.json"],
        },
    }

    # Save final_report.json
    json_path = r_dir / "final_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    # Save final_report.md
    md_path = r_dir / "final_report.md"
    md_content = generate_markdown_report(final_report)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    clean_dir = sanitize_results_dir(r_dir)
    print("\n✅ Compiled Final Report:")
    print(f"   JSON: {clean_dir}/{json_path.name}")
    print(f"   MD:   {clean_dir}/{md_path.name}\n")

    return final_report


def generate_markdown_report(report: dict[str, Any]) -> str:
    """Generate clean GitHub Flavored Markdown summary report across 4 categories."""
    acc = report["accuracy"]
    super_pos = acc.get("super_positive", {})
    pos = acc.get("positive", acc.get("positive_cases", {}))
    neg = acc.get("negative", acc.get("negative_cases", {}))
    super_neg = acc.get("super_negative", {})

    lat = report["guardrail_latency"]
    olat = report["overall_latency"]
    mem = report["memory_footprint"]
    err = report["error_metrics"]

    lines = [
        f"# Final Experiment Report: {report['experiment']}",
        f"**Timestamp:** `{report['timestamp']}`  ",
        f"**Directory:** `{report['results_directory']}`  ",
        f"**Total LLM Judge Calls:** `{report.get('llm_judge_layer_reached', 0)}`  ",
        "",
        "## 1. Accuracy Across 4 Categories",
        "",
        (
            "| Category | Criteria / Scope | Total Evaluated | Correct Decisions | "
            "Incorrect Decisions | Accuracy | LLM Judge Calls |"
        ),
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        (
            f"| **Super-Positive** | Pure culinary happy path | {super_pos.get('total', 0)} | "
            f"{super_pos.get('allowed', 0)} (Allowed) | "
            f"{super_pos.get('blocked', 0)} (False Refusal) | "
            f"**{super_pos.get('accuracy_pct', 'N/A')}** | "
            f"`{super_pos.get('llm_judge_layer_reached', 0)}` |"
        ),
        (
            f"| **Positive** | All on-charter (super_positive + positive) | "
            f"{pos.get('total', 0)} | {pos.get('allowed', 0)} (Allowed) | "
            f"{pos.get('blocked', 0)} (False Refusal) | "
            f"**{pos.get('accuracy_pct', 'N/A')}** | "
            f"`{pos.get('llm_judge_layer_reached', 0)}` |"
        ),
        (
            f"| **Negative** | Threats (negative + super_negative + garak) | "
            f"{neg.get('total', 0)} | {neg.get('blocked', 0)} (Blocked) | "
            f"{neg.get('allowed', 0)} (Bypassed) | "
            f"**{neg.get('accuracy_pct', 'N/A')}** | "
            f"`{neg.get('llm_judge_layer_reached', 0)}` |"
        ),
        (
            f"| **Super-Negative** | Straight-away attacks (benchmark only) | "
            f"{super_neg.get('total', 0)} | {super_neg.get('blocked', 0)} (Blocked) | "
            f"{super_neg.get('allowed', 0)} (Bypassed) | "
            f"**{super_neg.get('accuracy_pct', 'N/A')}** | "
            f"`{super_neg.get('llm_judge_layer_reached', 0)}` |"
        ),
        (
            f"| **Overall** | Combined | {pos.get('total', 0) + neg.get('total', 0)} | "
            f"{pos.get('allowed', 0) + neg.get('blocked', 0)} | "
            f"{pos.get('blocked', 0) + neg.get('allowed', 0)} | "
            f"**{acc['overall_accuracy_pct']}** | "
            f"`{report.get('llm_judge_layer_reached', 0)}` |"
        ),
        "",
        "## 2. Guardrail & Overall Latency Footprint (P95 & Mean)",
        "",
        (
            "| Category | Guardrail P95 | Guardrail Mean | Overall Latency P95 | "
            "Overall Latency Mean |"
        ),
        "| :--- | :--- | :--- | :--- | :--- |",
        (
            f"| **Super-Positive** | `{lat.get('super_positive_cases_p95_ms', 0.0)}ms` | "
            f"`{lat.get('super_positive_cases_mean_ms', 0.0)}ms` | "
            f"`{olat.get('super_positive_cases_p95_ms', 0.0)}ms` | "
            f"`{olat.get('super_positive_cases_mean_ms', 0.0)}ms` |"
        ),
        (
            f"| **Positive** | `{lat.get('positive_cases_p95_ms', 0.0)}ms` | "
            f"`{lat.get('positive_cases_mean_ms', 0.0)}ms` | "
            f"`{olat.get('positive_cases_p95_ms', 0.0)}ms` | "
            f"`{olat.get('positive_cases_mean_ms', 0.0)}ms` |"
        ),
        (
            f"| **Negative** | `{lat.get('negative_cases_p95_ms', 0.0)}ms` | "
            f"`{lat.get('negative_cases_mean_ms', 0.0)}ms` | "
            f"`{olat.get('negative_cases_p95_ms', 0.0)}ms` | "
            f"`{olat.get('negative_cases_mean_ms', 0.0)}ms` |"
        ),
        (
            f"| **Super-Negative** | `{lat.get('super_negative_cases_p95_ms', 0.0)}ms` | "
            f"`{lat.get('super_negative_cases_mean_ms', 0.0)}ms` | "
            f"`{olat.get('super_negative_cases_p95_ms', 0.0)}ms` | "
            f"`{olat.get('super_negative_cases_mean_ms', 0.0)}ms` |"
        ),
        (
            f"| **Overall** | **`{lat['overall_p95_ms']}ms`** | "
            f"**`{lat['overall_mean_ms']}ms`** | **`{olat['overall_p95_ms']}ms`** | "
            f"**`{olat['overall_mean_ms']}ms`** |"
        ),
        "",
        "## 3. Server Memory Footprint",
        "",
        "| Category | Peak RSS (MB) | Delta from Idle (MB) |",
        "| :--- | :--- | :--- |",
        f"| **Idle Baseline** | `{mem['idle_rss_mb']} MB` | `0.0 MB` |",
        (
            f"| **Super-Positive** | "
            f"`{mem.get('super_positive', {}).get('peak_rss_mb', 0.0)} MB` | "
            f"`+{mem.get('super_positive', {}).get('delta_rss_mb', 0.0)} MB` |"
        ),
        (
            f"| **Positive** | `{mem['positive_cases']['peak_rss_mb']} MB` | "
            f"`+{mem['positive_cases']['delta_rss_mb']} MB` |"
        ),
        (
            f"| **Negative** | `{mem['negative_cases']['peak_rss_mb']} MB` | "
            f"`+{mem['negative_cases']['delta_rss_mb']} MB` |"
        ),
        (
            f"| **Super-Negative** | "
            f"`{mem.get('super_negative', {}).get('peak_rss_mb', 0.0)} MB` | "
            f"`+{mem.get('super_negative', {}).get('delta_rss_mb', 0.0)} MB` |"
        ),
        (f"| **Peak Overall** | **`{mem['peak_rss_mb']} MB`** | **`+{mem['delta_rss_mb']} MB`** |"),
        "",
        "## 4. Error Metrics & API Robustness",
        "",
        f"- **Total Calls:** `{err['total_calls']}`",
        f"- **Successful Calls:** `{err['successful_calls']}`",
        f"- **Failed Calls:** `{err['failed_calls']}`",
        f"- **Failure Rate:** `{err['failure_rate_pct']}`",
        f"- **Total LLM Judge Calls:** `{report.get('llm_judge_layer_reached', 0)}`",
        "",
    ]
    if err.get("errors_by_type"):
        lines.append("### Failure Breakdown")
        for k, v in err["errors_by_type"].items():
            lines.append(f"- `{k}`: {v}")
        lines.append("")

    lines.extend(
        [
            "## 5. LLM Judge Layer Breakdown",
            "",
            "| Category | LLM Judge Calls | Description |",
            "| :--- | :--- | :--- |",
            (
                f"| **Super-Positive** | `{super_pos.get('llm_judge_layer_reached', 0)}` | "
                "Pure on-charter culinary path |"
            ),
            (
                f"| **Positive** | `{pos.get('llm_judge_layer_reached', 0)}` | "
                "All on-charter inquiries |"
            ),
            (
                f"| **Negative** | `{neg.get('llm_judge_layer_reached', 0)}` | "
                "Threats (benchmark + Garak) |"
            ),
            (
                f"| **Super-Negative** | `{super_neg.get('llm_judge_layer_reached', 0)}` | "
                "Straight-away attacks (benchmark only) |"
            ),
            (
                f"| **Overall Total** | **`{report.get('llm_judge_layer_reached', 0)}`** | "
                "Total LLM judge calls across experiment |"
            ),
            "",
        ]
    )

    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile final experiment report.")
    parser.add_argument("--results-dir", required=True, help="Path to experiment results directory")
    parser.add_argument("--experiment", required=False, help="Experiment name")
    args = parser.parse_args()

    compile_experiment_results(args.results_dir, experiment_name=args.experiment)
