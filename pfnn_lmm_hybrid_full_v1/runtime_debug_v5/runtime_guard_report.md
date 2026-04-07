# Runtime Guard Report

## Gate Status
- `teacher selector + teacher residual`: finite on all scenes
- `teacher selector + learned residual`: finite on all scenes
- `learned selector + zero residual`: finite on all scenes
- `learned selector + learned residual`: finite on all scenes

## What Changed
- switch severity score from feature/control/phase-contact/contact-mismatch deltas
- segmented residual carry: full / damped / hard reset depending on severity and runtime guards
- learned-selector hysteresis with dwell time and score margin
- frame-0 forced retrieval and per-frame anchor re-expression remain in place
- residual-stepper now uses training-matched control semantics and is hard-clamped to its training-domain norm

## Key Result
- The previous late-stage residual blow-up is gone. In `runtime_debug_v5`, `learned selector + learned residual` no longer accumulates residual norms into the old `55-66` range.
- Remaining divergence events are dominated by immediate contact-slip spikes on uphill/downhill/rough scenes, and those same spikes are already present in `teacher selector + teacher residual` and `learned selector + zero residual`.
- That means the residual carry bug has been separated from a baseline scene/contact start-up issue.

## First Divergence Snapshots
### teacher_selector_teacher_residual
- flat: no divergence event
- uphill: frame 1, slip 86.425163, residual_after 0.000000, stepper_input 0.000000, switch_severity 0.000000, hard_reset 0
- downhill: frame 1, slip 84.142311, residual_after 0.000000, stepper_input 0.000000, switch_severity 0.000000, hard_reset 0
- rough: frame 1, slip 84.208687, residual_after 0.000000, stepper_input 0.000000, switch_severity 0.000000, hard_reset 0
- obstacle: frame 325, slip 63.076004, residual_after 0.000000, stepper_input 0.000000, switch_severity 7.379500, hard_reset 0

### teacher_selector_learned_residual
- flat: no divergence event
- uphill: frame 1, slip 85.907913, residual_after 0.556364, stepper_input 27.363863, switch_severity 0.000000, hard_reset 0
- downhill: frame 1, slip 83.699966, residual_after 0.556364, stepper_input 27.363863, switch_severity 0.000000, hard_reset 0
- rough: frame 1, slip 83.773293, residual_after 0.556364, stepper_input 27.363863, switch_severity 0.000000, hard_reset 0
- obstacle: no divergence event

### learned_selector_zero_residual
- flat: frame 685, slip 58.571472, residual_after 0.000000, stepper_input 0.000000, switch_severity 5.029471, hard_reset 0
- uphill: frame 1, slip 86.425163, residual_after 0.000000, stepper_input 0.000000, switch_severity 0.000000, hard_reset 0
- downhill: frame 1, slip 84.142311, residual_after 0.000000, stepper_input 0.000000, switch_severity 0.000000, hard_reset 0
- rough: frame 1, slip 84.208687, residual_after 0.000000, stepper_input 0.000000, switch_severity 0.000000, hard_reset 0
- obstacle: no divergence event

### learned_selector_learned_residual
- flat: no divergence event
- uphill: frame 1, slip 85.907913, residual_after 0.556364, stepper_input 27.363863, switch_severity 0.000000, hard_reset 0
- downhill: frame 1, slip 83.699966, residual_after 0.556364, stepper_input 27.363863, switch_severity 0.000000, hard_reset 0
- rough: frame 1, slip 83.773293, residual_after 0.556364, stepper_input 27.363863, switch_severity 0.000000, hard_reset 0
- obstacle: no divergence event

## Interpretation
- Hybrid closure is now behaving as a guarded piecewise-local system: large-switch residual carry is no longer the first failure mode.
- The next runtime-only issue is contact handling around the first slope/rough-ground support phase, not residual manifold escape.
