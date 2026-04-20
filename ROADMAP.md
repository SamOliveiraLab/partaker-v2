# PARTAKER Roadmap — from raw ND2 to digital twin

Living checklist. Tick items as we finish them. Add/rename freely.

---

## Phase 1 — Data ingest
- [x] Load ND2 files (multi-position, multi-channel, multi-timepoint)
- [x] Experiment abstraction with position/channel metadata
- [x] `PositionsMismatchError` + truncating positions in Experiment
- [ ] Robust handling of heterogeneous acquisitions (mixed channels across positions)

## Phase 2 — Segmentation
- [x] Cellpose / Omnipose integration
- [x] Per-position cached masks
- [ ] QC pass: segmentation review UI (flag bad frames, edit masks)
- [ ] Benchmark cellpose vs omnipose on our strains

## Phase 3 — Tracking & lineage
- [x] btrack integration (BayesianTracker + GLPK)
- [x] Trackastra integration (transformer, greedy / greedy_nodiv / ilp)
- [x] Per-position tracking cache, with algorithm-switch prompt
- [x] Progress dialog lifecycle fixed
- [x] btrack GLPK infinite-loop fix (`tm_lim` + `mip_gap`)
- [ ] btrack division/merge step (drop `mip_gap`, accept `feasible`, keep `tm_lim`)
- [ ] Side-by-side btrack vs Trackastra validation on a ground-truth clip
- [ ] Manual lineage editor (merge/split tracks, fix mis-links)

## Phase 3.5 — Analysis split (paper-1 deck, Figure 2)
From here the pipeline forks into two parallel analysis approaches. The UI makes this explicit via two tabs in the Tracking page.

- [x] Tracking tab split into **Cell View** + **Environment View** tabs (QTabWidget)
- [x] `tracks_to_grid` helper (pixel tracks → per-grid per-timepoint state)
- [x] Environment View tab: density map, mean-speed map, grid controls, time selector
- [ ] Environment View: COMSOL hydrodynamic variable map
- [ ] Environment View: grid↔grid neighbour correlations
- [ ] Environment View: cell-state features per grid (mean size, dividing fraction, FL mean)
- [ ] Cell View: absorb Lineage Trees + Motility as internal tabs of `CellViewDialog` (currently still modal sub-dialogs)
- [x] **Animated Cell View** — click a cell, watch its life play back with ROI-aware chamber canvas, spotlight + trail, trajectory panel, morphology filmstrip, playback controls, division-aware spotlight, dwell-heatmap COMSOL placeholder, live stats ticker

## Phase 4 — Cell metrics (Approach 2: Cell View)
- [x] Basic per-cell metrics (area, intensity, position)
- [ ] Fix `cell_history._build_cell_record` → pass `position=` to `metrics_service.query_optimized`
- [ ] Growth-rate estimation per cell
- [ ] Division-time distribution
- [ ] Cell-cycle phase inference

## Phase 5 — Fluorescence / reporter quantification
- [x] RPU button wiring
- [ ] RPU calibration workflow (standard curve, per-channel)
- [ ] Background subtraction robust across positions
- [ ] Bleed-through correction
- [ ] Multi-reporter ratiometric analysis

## Phase 6 — Population & lineage analytics (Approach 2) + grid dynamics (Approach 1)
- [ ] Lineage trees with expression overlays
- [ ] Sister-cell correlations
- [ ] Mother-machine-style kymographs (if applicable)
- [ ] Population-level growth curves from single-cell data
- [ ] Heritability / memory quantification

## Phase 7 — Model fitting
- [ ] Growth models (exponential, logistic, with noise)
- [ ] Gene regulation models (Hill, repression, activation)
- [ ] Stochastic parameter inference
- [ ] Parameter sensitivity + identifiability analysis

## Phase 8 — Digital twin
- [ ] Agent-based simulator seeded from fitted parameters
- [ ] Reproduce observed lineage statistics in silico
- [ ] Perturbation prediction (inducer dose, strain variant)
- [ ] Validation loop: simulate → compare to held-out experiment → refit
- [ ] Interactive twin inside the PARTAKER UI

---

## Cross-cutting
- [ ] Unify `partaker` (v1) and `partaker-v2` — decide deprecation plan for v1
- [ ] Commit pending v2 changes (`tracking.py`, `tracking_widget.py`, `pyproject.toml`, `uv.lock`)
- [ ] Packaging / distribution for lab members
- [ ] Test suite (at minimum: tracking, metrics, RPU)
- [ ] Docs for lab onboarding
