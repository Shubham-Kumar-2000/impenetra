# Final Experiment Report: promptguard_as_guardrail
**Timestamp:** `20261115_080114`  
**Directory:** `impenetra/experiments/promptguard_as_guardrail/results`  

## 1. Accuracy Across Positive & Negative Cases

| Case Type | Criteria | Total Evaluated | Correct Decisions | Incorrect Decisions | Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Positive Cases** | Should NOT be blocked | 12 | 12 (Allowed) | 0 (False Refusal) | **100.0%** |
| **Negative Cases** | Should be blocked | 347 | 227 (Blocked) | 120 (Bypassed) | **65.4%** |
| **Overall** | Combined | 359 | 239 | 120 | **66.6%** |

## 2. Guardrail & Overall Latency Footprint (P95 & Mean)

| Case Type | Guardrail P95 | Guardrail Mean | Overall Latency P95 | Overall Latency Mean |
| :--- | :--- | :--- | :--- | :--- |
| **Positive Cases** | `438.953ms` | `332.121ms` | `443.531ms` | `334.952ms` |
| **Negative Cases** | `466.877ms` | `334.942ms` | `468.675ms` | `337.957ms` |
| **Overall** | **`466.877ms`** | **`333.53ms`** | **`468.675ms`** | **`336.45ms`** |

## 3. Server Memory Footprint

| Condition | Peak RSS (MB) | Delta from Idle (MB) |
| :--- | :--- | :--- |
| **Idle Baseline** | `101.98 MB` | `0.0 MB` |
| **Positive Cases** | `104.09 MB` | `+2.11 MB` |
| **Negative Cases** | `104.11 MB` | `+2.13 MB` |
| **Peak Overall** | **`104.11 MB`** | **`+2.13 MB`** |

## 4. Error Metrics & API Robustness

- **Total Calls:** `359`
- **Successful Calls:** `359`
- **Failed Calls:** `0`
- **Failure Rate:** `0.00%`
