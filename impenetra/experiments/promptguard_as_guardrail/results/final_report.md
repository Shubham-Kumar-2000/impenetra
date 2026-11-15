# Final Experiment Report: promptguard_as_guardrail
**Timestamp:** `20261115_062429`  
**Directory:** `impenetra/experiments/promptguard_as_guardrail/results`  

## 1. Accuracy Across Positive & Negative Cases

| Case Type | Criteria | Total Evaluated | Correct Decisions | Incorrect Decisions | Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Positive Cases** | Should NOT be blocked | 12 | 12 (Allowed) | 0 (False Refusal) | **100.0%** |
| **Negative Cases** | Should be blocked | 347 | 225 (Blocked) | 122 (Bypassed) | **64.8%** |
| **Overall** | Combined | 359 | 237 | 122 | **66.0%** |

## 2. Guardrail & Overall Latency Footprint (P95 & Mean)

| Case Type | Guardrail P95 | Guardrail Mean | Overall Latency P95 | Overall Latency Mean |
| :--- | :--- | :--- | :--- | :--- |
| **Positive Cases** | `196.507ms` | `132.268ms` | `199.927ms` | `134.306ms` |
| **Negative Cases** | `197.337ms` | `134.011ms` | `198.73ms` | `136.014ms` |
| **Overall** | **`197.337ms`** | **`133.14ms`** | **`199.927ms`** | **`135.16ms`** |

## 3. Server Memory Footprint

| Condition | Peak RSS (MB) | Delta from Idle (MB) |
| :--- | :--- | :--- |
| **Idle Baseline** | `101.81 MB` | `0.0 MB` |
| **Positive Cases** | `104.73 MB` | `+2.92 MB` |
| **Negative Cases** | `104.73 MB` | `+2.92 MB` |
| **Peak Overall** | **`104.73 MB`** | **`+2.92 MB`** |

## 4. Error Metrics & API Robustness

- **Total Calls:** `359`
- **Successful Calls:** `359`
- **Failed Calls:** `0`
- **Failure Rate:** `0.00%`
