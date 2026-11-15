# Final Experiment Report: system_prompt_as_guardrail
**Timestamp:** `20261115_140626`  
**Directory:** `impenetra/experiments/system_prompt_as_guardrail/results`  

## 1. Accuracy Across Positive & Negative Cases

| Case Type | Criteria | Total Evaluated | Correct Decisions | Incorrect Decisions | Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Positive Cases** | Should NOT be blocked | 12 | 12 (Allowed) | 0 (False Refusal) | **100.0%** |
| **Negative Cases** | Should be blocked | 347 | 304 (Blocked) | 43 (Bypassed) | **87.6%** |
| **Overall** | Combined | 359 | 316 | 43 | **88.0%** |

## 2. Guardrail & Overall Latency Footprint (P95 & Mean)

| Case Type | Guardrail P95 | Guardrail Mean | Overall Latency P95 | Overall Latency Mean |
| :--- | :--- | :--- | :--- | :--- |
| **Positive Cases** | `605.268ms` | `380.254ms` | `611.408ms` | `386.05ms` |
| **Negative Cases** | `2779.621ms` | `719.41ms` | `2780.803ms` | `721.005ms` |
| **Overall** | **`2779.621ms`** | **`549.83ms`** | **`2780.803ms`** | **`553.53ms`** |

## 3. Server Memory Footprint

| Condition | Peak RSS (MB) | Delta from Idle (MB) |
| :--- | :--- | :--- |
| **Idle Baseline** | `96.47 MB` | `0.0 MB` |
| **Positive Cases** | `102.12 MB` | `+5.65 MB` |
| **Negative Cases** | `102.56 MB` | `+6.09 MB` |
| **Peak Overall** | **`102.56 MB`** | **`+6.09 MB`** |

## 4. Error Metrics & API Robustness

- **Total Calls:** `359`
- **Successful Calls:** `359`
- **Failed Calls:** `0`
- **Failure Rate:** `0.00%`
