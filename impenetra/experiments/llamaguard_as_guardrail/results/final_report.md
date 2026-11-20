# Final Experiment Report: llamaguard_as_guardrail
**Timestamp:** `20261118_013906`  
**Directory:** `impenetra/experiments/llamaguard_as_guardrail/results`  
**Total LLM Judge Calls:** `0`  

## 1. Accuracy Across Positive & Negative Cases

| Category | Criteria / Scope | Total Evaluated | Correct Decisions | Incorrect Decisions | Accuracy | LLM Judge Calls |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Positive** | On-charter benign inquiries (should NOT block) | 12 | 12 (Allowed) | 0 (False Refusal) | **100.0%** | `0` |
| **Negative** | Threats & off-charter queries (SHOULD block) | 347 | 275 (Blocked) | 72 (Bypassed) | **79.3%** | `0` |
| **Overall** | Combined | 359 | 287 | 72 | **79.9%** | `0` |

## 2. Guardrail & Overall Latency Footprint (P95 & Mean)

| Category | Guardrail P95 | Guardrail Mean | Overall Latency P95 | Overall Latency Mean |
| :--- | :--- | :--- | :--- | :--- |
| **Positive** | `390.245ms` | `320.465ms` | `395.116ms` | `327.945ms` |
| **Negative** | `561.521ms` | `373.417ms` | `561.541ms` | `376.066ms` |
| **Overall** | **`561.521ms`** | **`346.94ms`** | **`561.541ms`** | **`352.01ms`** |

## 3. Server Memory Footprint

| Category | Peak RSS (MB) | Delta from Idle (MB) |
| :--- | :--- | :--- |
| **Idle Baseline** | `110.5 MB` | `0.0 MB` |
| **Positive** | `112.83 MB` | `+2.33 MB` |
| **Negative** | `121.02 MB` | `+10.52 MB` |
| **Peak Overall** | **`121.02 MB`** | **`+10.52 MB`** |

## 4. Error Metrics & API Robustness

- **Total Calls:** `359`
- **Successful Calls:** `359`
- **Failed Calls:** `0`
- **Failure Rate:** `0.00%`
- **Total LLM Judge Calls:** `0`

## 5. LLM Judge Layer Breakdown

| Category | LLM Judge Calls | Description |
| :--- | :--- | :--- |
| **Positive** | `0` | All on-charter inquiries |
| **Negative** | `0` | Threats & off-charter queries (benchmark + Garak) |
| **Overall Total** | **`0`** | Total LLM judge calls across experiment |
