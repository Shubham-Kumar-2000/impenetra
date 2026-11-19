# Final Experiment Report: system_prompt_as_guardrail
**Timestamp:** `20261119_182301`  
**Directory:** `impenetra/experiments/system_prompt_as_guardrail/results`  
**Total LLM Judge Calls:** `724`  

## 1. Accuracy Across Positive & Negative Cases

| Category | Criteria / Scope | Total Evaluated | Correct Decisions | Incorrect Decisions | Accuracy | LLM Judge Calls |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Positive** | On-charter benign inquiries (should NOT block) | 40 | 32 (Allowed) | 8 (False Refusal) | **80.0%** | `40` |
| **Negative** | Threats & off-charter queries (SHOULD block) | 684 | 608 (Blocked) | 76 (Bypassed) | **88.9%** | `684` |
| **Overall** | Combined | 724 | 640 | 84 | **88.4%** | `724` |

### Classification Metrics (Threat Detection)

| Metric | Value | Description |
| :--- | :--- | :--- |
| **Precision** | **`98.7%`** | TP / (TP + FP) — Threat accuracy among blocked prompts |
| **Recall (Sensitivity)** | **`88.9%`** | TP / (TP + FN) — Threat catch rate |
| **F1 Score** | **`93.5%`** | Harmonic mean of Precision and Recall |
| **False Positive Rate (FPR)** | **`20.0%`** | FP / (FP + TN) — Benign false refusal / overblocking rate |

## 2. Guardrail & Overall Latency Footprint (P95 & Mean)

| Category | Guardrail P95 | Guardrail Mean | Overall Latency P95 | Overall Latency Mean |
| :--- | :--- | :--- | :--- | :--- |
| **Positive** | `434.521ms` | `333.067ms` | `8679.648ms` | `3417.528ms` |
| **Negative** | `2465.427ms` | `601.82ms` | `9295.019ms` | `965.783ms` |
| **Overall** | **`2465.427ms`** | **`467.44ms`** | **`9295.019ms`** | **`2191.66ms`** |

## 3. Server Memory Footprint

| Category | Peak RSS (MB) | Delta from Idle (MB) |
| :--- | :--- | :--- |
| **Idle Baseline** | `77.77 MB` | `0.0 MB` |
| **Positive** | `102.25 MB` | `+24.48 MB` |
| **Negative** | `77.77 MB` | `+0.0 MB` |
| **Peak Overall** | **`102.25 MB`** | **`+24.48 MB`** |

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
