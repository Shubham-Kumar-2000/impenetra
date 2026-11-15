# Final Experiment Report: nemo_guardrails
**Timestamp:** `20261115_134921`  
**Directory:** `impenetra/experiments/nemo_guardrails/results`  

## 1. Accuracy Across Positive & Negative Cases

| Case Type | Criteria | Total Evaluated | Correct Decisions | Incorrect Decisions | Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Positive Cases** | Should NOT be blocked | 12 | 11 (Allowed) | 1 (False Refusal) | **91.7%** |
| **Negative Cases** | Should be blocked | 347 | 322 (Blocked) | 25 (Bypassed) | **92.8%** |
| **Overall** | Combined | 359 | 333 | 26 | **92.8%** |

## 2. Guardrail & Overall Latency Footprint (P95 & Mean)

| Case Type | Guardrail P95 | Guardrail Mean | Overall Latency P95 | Overall Latency Mean |
| :--- | :--- | :--- | :--- | :--- |
| **Positive Cases** | `29458.19ms` | `23453.467ms` | `56783.892ms` | `29007.603ms` |
| **Negative Cases** | `29156.961ms` | `7634.352ms` | `29156.974ms` | `7634.849ms` |
| **Overall** | **`29458.19ms`** | **`15543.91ms`** | **`56783.892ms`** | **`18321.23ms`** |

## 3. Server Memory Footprint

| Condition | Peak RSS (MB) | Delta from Idle (MB) |
| :--- | :--- | :--- |
| **Idle Baseline** | `148.52 MB` | `0.0 MB` |
| **Positive Cases** | `202.66 MB` | `+54.14 MB` |
| **Negative Cases** | `213.89 MB` | `+65.37 MB` |
| **Peak Overall** | **`213.89 MB`** | **`+65.37 MB`** |

## 4. Error Metrics & API Robustness

- **Total Calls:** `359`
- **Successful Calls:** `359`
- **Failed Calls:** `0`
- **Failure Rate:** `0.00%`
