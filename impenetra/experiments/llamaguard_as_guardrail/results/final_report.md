# Final Experiment Report: llamaguard_as_guardrail
**Timestamp:** `20261114_174950`  
**Directory:** `impenetra/experiments/llamaguard_as_guardrail/results`  

## 1. Accuracy Across Positive & Negative Cases

| Case Type | Criteria | Total Evaluated | Correct Decisions | Incorrect Decisions | Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Positive Cases** | Should NOT be blocked | 12 | 12 (Allowed) | 0 (False Refusal) | **100.0%** |
| **Negative Cases** | Should be blocked | 347 | 275 (Blocked) | 72 (Bypassed) | **79.3%** |
| **Overall** | Combined | 359 | 287 | 72 | **79.9%** |

## 2. Guardrail & Overall Latency Footprint (P95 & Mean)

| Case Type | Guardrail P95 | Guardrail Mean | Overall Latency P95 | Overall Latency Mean |
| :--- | :--- | :--- | :--- | :--- |
| **Positive Cases** | `534.686ms` | `402.908ms` | `534.709ms` | `404.562ms` |
| **Negative Cases** | `561.521ms` | `368.27ms` | `561.541ms` | `370.681ms` |
| **Overall** | **`561.521ms`** | **`385.59ms`** | **`561.541ms`** | **`387.62ms`** |

## 3. Server Memory Footprint

| Condition | Peak RSS (MB) | Delta from Idle (MB) |
| :--- | :--- | :--- |
| **Idle Baseline** | `110.5 MB` | `0.0 MB` |
| **Positive Cases** | `112.83 MB` | `+2.33 MB` |
| **Negative Cases** | `121.02 MB` | `+10.52 MB` |
| **Peak Overall** | **`121.02 MB`** | **`+10.52 MB`** |

## 4. Error Metrics & API Robustness

- **Total Calls:** `359`
- **Successful Calls:** `359`
- **Failed Calls:** `0`
- **Failure Rate:** `0.00%`
