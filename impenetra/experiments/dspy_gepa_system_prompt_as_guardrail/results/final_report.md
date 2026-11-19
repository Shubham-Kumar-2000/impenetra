# Final Experiment Report: dspy_gepa_system_prompt_as_guardrail
**Timestamp:** `20261120_002804`  
**Directory:** `impenetra/experiments/dspy_gepa_system_prompt_as_guardrail/results`  
**Total LLM Judge Calls:** `724`  

## 1. Accuracy Across Positive & Negative Cases

| Category | Criteria / Scope | Total Evaluated | Correct Decisions | Incorrect Decisions | Accuracy | LLM Judge Calls |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Positive** | On-charter benign inquiries (should NOT block) | 40 | 32 (Allowed) | 8 (False Refusal) | **80.0%** | `40` |
| **Negative** | Threats & off-charter queries (SHOULD block) | 684 | 605 (Blocked) | 79 (Bypassed) | **88.5%** | `684` |
| **Overall** | Combined | 724 | 637 | 87 | **88.0%** | `724` |

### Classification Metrics (Threat Detection)

| Metric | Value | Description |
| :--- | :--- | :--- |
| **Precision** | **`98.7%`** | TP / (TP + FP) — Threat accuracy among blocked prompts |
| **Recall (Sensitivity)** | **`88.5%`** | TP / (TP + FN) — Threat catch rate |
| **F1 Score** | **`93.3%`** | Harmonic mean of Precision and Recall |
| **False Positive Rate (FPR)** | **`20.0%`** | FP / (FP + TN) — Benign false refusal / overblocking rate |

## 2. Guardrail & Overall Latency Footprint (P95 & Mean)

| Category | Guardrail P95 | Guardrail Mean | Overall Latency P95 | Overall Latency Mean |
| :--- | :--- | :--- | :--- | :--- |
| **Positive** | `722.602ms` | `384.203ms` | `10621.007ms` | `3916.614ms` |
| **Negative** | `2516.919ms` | `688.992ms` | `11846.601ms` | `1133.177ms` |
| **Overall** | **`2516.919ms`** | **`536.6ms`** | **`11846.601ms`** | **`2524.9ms`** |

## 3. Server Memory Footprint

| Category | Peak RSS (MB) | Delta from Idle (MB) |
| :--- | :--- | :--- |
| **Idle Baseline** | `96.78 MB` | `0.0 MB` |
| **Positive** | `102.77 MB` | `+5.99 MB` |
| **Negative** | `99.25 MB` | `+2.47 MB` |
| **Peak Overall** | **`102.77 MB`** | **`+5.99 MB`** |

## 4. Error Metrics & API Robustness

- **Total Calls:** `724`
- **Successful Calls:** `724`
- **Failed Calls:** `0`
- **Failure Rate:** `0.00%`
- **Total LLM Judge Calls:** `724`

## 5. LLM Judge Layer Breakdown

| Category | LLM Judge Calls | Description |
| :--- | :--- | :--- |
| **Positive** | `40` | All on-charter inquiries |
| **Negative** | `684` | Threats & off-charter queries (benchmark + Garak) |
| **Overall Total** | **`724`** | Total LLM judge calls across experiment |
