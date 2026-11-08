# Paper1 Reviewer-Response Checklist

| Reviewer issue | Revision support | Status before formal run |
|---|---|---|
| Long-term RL versus one-slot decisions | `horizon` E1 and `per_slot` comparison | Code ready; results pending |
| Cost critic and discount consistency | `horizon`: generic critics, `gamma=0/0.99` | Code ready; not a full structured-critic one-factor ablation |
| Strong constrained-RL baseline | `cpo` with split actor, `gamma=0/0.99` | Code ready; results pending |
| Critic structure versus SAC update rule | `fair_critics` | Code ready; results pending |
| Fixed penalty comparison | structured beta sweep `{2,2.5,3,3.5,4.5}` | Code ready; equal-failure comparison is an analysis, not direct training |
| Per-slot numerical reference | `per_slot` actor, critic descent, full-information optimizer | Code ready; finite local optimizer, not global optimum |
| AP-dependent channel model | `ap` learned AP, best-channel AP, AP-independent gains | Code ready; results pending |
| Model mismatch | `mismatch` MEC capacity, PA efficiency, cycles/bit, MEC sharing | Code ready; results pending |
| Decision latency | `latency` plus optimizer decision times | Code ready; measures computation path, not full system latency |
| More statistical reporting | five paired deployments where declared, paired deltas, SD, nominal CI | Code ready; five pairs do not establish equivalence by themselves |
| Device-level reliability | existing device P95 diagnostics | Still unresolved; must remain in limitations |
| DVFS | fixed heterogeneous CPU profiles remain | Unresolved; no DVFS experiment added |
| Long-horizon safety guarantee | immediate critics and conservative wording | Must not claim a hard CMDP guarantee |
| Historical main-result provenance | separate historical main checkout required | Not mixed with revision branch |
