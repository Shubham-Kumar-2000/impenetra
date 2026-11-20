# Final Experiment Report: vector_similarity_as_guardrail
**Timestamp:** `20261121_004157`  
**Directory:** `impenetra/experiments/vector_similarity_as_guardrail/results`  
**Total LLM Judge Calls:** `0`  

## 1. Accuracy Across Positive & Negative Cases

| Category | Criteria / Scope | Total Evaluated | Correct Decisions | Incorrect Decisions | Accuracy | LLM Judge Calls |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Positive** | On-charter benign inquiries (should NOT block) | 610 | 170 (Allowed) | 440 (False Refusal) | **27.9%** | `0` |
| **Negative** | Threats & off-charter queries (SHOULD block) | 5614 | 5498 (Blocked) | 116 (Bypassed) | **97.9%** | `0` |
| **Overall** | Combined | 6224 | 5668 | 556 | **91.1%** | `0` |

### Classification Metrics (Threat Detection)

| Metric | Value | Description |
| :--- | :--- | :--- |
| **Precision** | **`92.6%`** | TP / (TP + FP) — Threat accuracy among blocked prompts |
| **Recall (Sensitivity)** | **`97.9%`** | TP / (TP + FN) — Threat catch rate |
| **F1 Score** | **`95.2%`** | Harmonic mean of Precision and Recall |
| **False Positive Rate (FPR)** | **`72.1%`** | FP / (FP + TN) — Benign false refusal / overblocking rate |

## 2. Guardrail & Overall Latency Footprint (P95 & Mean)

| Category | Guardrail P95 | Guardrail Mean | Overall Latency P95 | Overall Latency Mean |
| :--- | :--- | :--- | :--- | :--- |
| **Positive** | `13.935ms` | `10.386ms` | `308.541ms` | `264.593ms` |
| **Negative** | `56.421ms` | `12.306ms` | `303.594ms` | `249.172ms` |
| **Overall** | **`56.421ms`** | **`11.35ms`** | **`308.541ms`** | **`256.88ms`** |

## 3. Server Memory Footprint

| Category | Peak RSS (MB) | Delta from Idle (MB) |
| :--- | :--- | :--- |
| **Idle Baseline** | `481.73 MB` | `0.0 MB` |
| **Positive** | `879.61 MB` | `+397.88 MB` |
| **Negative** | `481.73 MB` | `+0.0 MB` |
| **Peak Overall** | **`879.61 MB`** | **`+397.88 MB`** |

## 4. Error Metrics & API Robustness

- **Total Calls:** `6224`
- **Successful Calls:** `6224`
- **Failed Calls:** `0`
- **Failure Rate:** `0.00%`
- **Total LLM Judge Calls:** `0`

## 5. LLM Judge Layer Breakdown

| Category | LLM Judge Calls | Description |
| :--- | :--- | :--- |
| **Positive** | `0` | All on-charter inquiries |
| **Negative** | `0` | Threats & off-charter queries (benchmark + Garak) |
| **Overall Total** | **`0`** | Total LLM judge calls across experiment |
