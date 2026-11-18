# Final Experiment Report: jev_systemone_choice_as_guardrail
**Timestamp:** `20261118_183739`  
**Directory:** `impenetra/experiments/jev_systemone_choice_as_guardrail/results`  
**Total LLM Judge Calls:** `0`  

## 1. Accuracy Across Positive & Negative Cases

| Category | Criteria / Scope | Total Evaluated | Correct Decisions | Incorrect Decisions | Accuracy | LLM Judge Calls |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Positive** | On-charter benign inquiries (should NOT block) | 40 | 36 (Allowed) | 4 (False Refusal) | **90.0%** | `0` |
| **Negative** | Threats & off-charter queries (SHOULD block) | 684 | 647 (Blocked) | 37 (Bypassed) | **94.6%** | `0` |
| **Overall** | Combined | 724 | 683 | 41 | **94.3%** | `0` |

### Classification Metrics (Threat Detection)

| Metric | Value | Description |
| :--- | :--- | :--- |
| **Precision** | **`99.4%`** | TP / (TP + FP) — Threat accuracy among blocked prompts |
| **Recall (Sensitivity)** | **`94.6%`** | TP / (TP + FN) — Threat catch rate |
| **F1 Score** | **`96.9%`** | Harmonic mean of Precision and Recall |
| **False Positive Rate (FPR)** | **`10.0%`** | FP / (FP + TN) — Benign false refusal / overblocking rate |

## 2. Guardrail & Overall Latency Footprint (P95 & Mean)

| Category | Guardrail P95 | Guardrail Mean | Overall Latency P95 | Overall Latency Mean |
| :--- | :--- | :--- | :--- | :--- |
| **Positive** | `373.671ms` | `296.922ms` | `8122.613ms` | `2885.521ms` |
| **Negative** | `371.248ms` | `290.424ms` | `7784.362ms` | `514.866ms` |
| **Overall** | **`373.671ms`** | **`293.67ms`** | **`8122.613ms`** | **`1700.19ms`** |

## 3. Server Memory Footprint

| Category | Peak RSS (MB) | Delta from Idle (MB) |
| :--- | :--- | :--- |
| **Idle Baseline** | `95.94 MB` | `0.0 MB` |
| **Positive** | `101.98 MB` | `+6.04 MB` |
| **Negative** | `102.09 MB` | `+6.15 MB` |
| **Peak Overall** | **`102.09 MB`** | **`+6.15 MB`** |

## 4. Error Metrics & API Robustness

- **Total Calls:** `724`
- **Successful Calls:** `724`
- **Failed Calls:** `0`
- **Failure Rate:** `0.00%`
- **Total LLM Judge Calls:** `0`

## 5. LLM Judge Layer Breakdown

| Category | LLM Judge Calls | Description |
| :--- | :--- | :--- |
| **Positive** | `0` | All on-charter inquiries |
| **Negative** | `0` | Threats & off-charter queries (benchmark + Garak) |
| **Overall Total** | **`0`** | Total LLM judge calls across experiment |
