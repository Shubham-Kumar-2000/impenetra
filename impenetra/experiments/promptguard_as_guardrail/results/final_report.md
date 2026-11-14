# Final Experiment Report: promptguard_as_guardrail
**Timestamp:** `20261115_051243`  
**Directory:** `impenetra/experiments/promptguard_as_guardrail/results`  

## 1. Accuracy Across Positive & Negative Cases

| Case Type | Criteria | Total Evaluated | Correct Decisions | Incorrect Decisions | Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Positive Cases** | Should NOT be blocked | 12 | 12 (Allowed) | 0 (False Refusal) | **100.0%** |
| **Negative Cases** | Should be blocked | 347 | 224 (Blocked) | 123 (Bypassed) | **64.6%** |
| **Overall** | Combined | 359 | 236 | 123 | **65.7%** |

## 2. Guardrail & Overall Latency Footprint (P95 & Mean)

| Case Type | Guardrail P95 | Guardrail Mean | Overall Latency P95 | Overall Latency Mean |
| :--- | :--- | :--- | :--- | :--- |
| **Positive Cases** | `524.768ms` | `451.067ms` | `528.346ms` | `453.626ms` |
| **Negative Cases** | `824.039ms` | `525.452ms` | `830.017ms` | `528.079ms` |
| **Overall** | **`824.039ms`** | **`488.26ms`** | **`830.017ms`** | **`490.85ms`** |

## 3. Server Memory Footprint

| Condition | Peak RSS (MB) | Delta from Idle (MB) |
| :--- | :--- | :--- |
| **Idle Baseline** | `102.0 MB` | `0.0 MB` |
| **Positive Cases** | `104.12 MB` | `+2.12 MB` |
| **Negative Cases** | `104.16 MB` | `+2.16 MB` |
| **Peak Overall** | **`104.16 MB`** | **`+2.16 MB`** |

## 4. Error Metrics & API Robustness

- **Total Calls:** `359`
- **Successful Calls:** `359`
- **Failed Calls:** `0`
- **Failure Rate:** `0.00%`
