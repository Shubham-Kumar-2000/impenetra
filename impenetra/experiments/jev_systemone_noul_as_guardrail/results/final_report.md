# Final Experiment Report: jev_systemone_noul_as_guardrail
**Timestamp:** `20261118_184432`  
**Directory:** `impenetra/experiments/jev_systemone_noul_as_guardrail/results`  
**Total LLM Judge Calls:** `0`  

## 1. Accuracy Across Positive & Negative Cases

| Category | Criteria / Scope | Total Evaluated | Correct Decisions | Incorrect Decisions | Accuracy | LLM Judge Calls |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Positive** | On-charter benign inquiries (should NOT block) | 40 | 37 (Allowed) | 3 (False Refusal) | **92.5%** | `0` |
| **Negative** | Threats & off-charter queries (SHOULD block) | 684 | 645 (Blocked) | 39 (Bypassed) | **94.3%** | `0` |
| **Overall** | Combined | 724 | 682 | 42 | **94.2%** | `0` |

### Classification Metrics (Threat Detection)

| Metric | Value | Description |
| :--- | :--- | :--- |
| **Precision** | **`99.5%`** | TP / (TP + FP) — Threat accuracy among blocked prompts |
| **Recall (Sensitivity)** | **`94.3%`** | TP / (TP + FN) — Threat catch rate |
| **F1 Score** | **`96.8%`** | Harmonic mean of Precision and Recall |
| **False Positive Rate (FPR)** | **`7.5%`** | FP / (FP + TN) — Benign false refusal / overblocking rate |

## 2. Guardrail & Overall Latency Footprint (P95 & Mean)

| Category | Guardrail P95 | Guardrail Mean | Overall Latency P95 | Overall Latency Mean |
| :--- | :--- | :--- | :--- | :--- |
| **Positive** | `369.53ms` | `286.209ms` | `7840.791ms` | `2945.89ms` |
| **Negative** | `375.06ms` | `287.921ms` | `7584.865ms` | `517.867ms` |
| **Overall** | **`375.06ms`** | **`287.06ms`** | **`7840.791ms`** | **`1731.88ms`** |

## 3. Server Memory Footprint

| Category | Peak RSS (MB) | Delta from Idle (MB) |
| :--- | :--- | :--- |
| **Idle Baseline** | `96.42 MB` | `0.0 MB` |
| **Positive** | `102.59 MB` | `+6.17 MB` |
| **Negative** | `102.62 MB` | `+6.2 MB` |
| **Peak Overall** | **`102.62 MB`** | **`+6.2 MB`** |

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
