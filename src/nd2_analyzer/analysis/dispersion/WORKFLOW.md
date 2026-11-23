# Dispersion territory and breakup measurement

This is the workflow approved for integration on November 22, 2026. A future local-darkness classifier is separate work. Existing reviewed-mask imports remain available; fresh automatic runs use the rules below. Seeding and Maturation are independent.

## Geometry and identity

1. Exclude the 15-pixel analysis border. Normalize for outer-boundary detection, blur 5×5, use Otsu plus the intensity-control adjustment. Defaults: intensity 39%, ellipse kernel 37 pixels, minimum outer area 0.006 times the analysis image area. Follow the selected whole colony by overlap.
2. Initialize territories from local darkness (32-pixel window), sigma-2 smoothing, prominence 0.025, peak spacing 18 pixels, minimum area 120 pixels and minimum support 0.5. Blend validated transported density with weight 0.35; enhance local relief with sigma 18 and gain 0.65. Require two observations for peak births/retirements.
3. Recover mixed dark/light initial pockets using contrast, relief, neighborhood and checked residual-motion support. Initial motion samples at most 1,200 corners with 6-pixel minimum spacing. Initialization uses reflected 15-pixel padding of the cropped image, preserving the reviewed experiment's filter boundary condition. Later central-compartment boundary refinement uses the real source-frame context.
4. Track texture with checked affine correspondence, appearance comparisons and displaced negative controls. Retention requires belief at least 0.65. Changes exceeding an area factor of 1.3 need independent material-change evidence. Partial recovery uses locally checked patches rather than extrapolating the entire unsupported footprint.
5. Add persistent coarse structures using scales 5/8/12, rolling median and MAD, contrast at least 5, stability ratio 2, polarity agreement 0.8, at least five observations and agreement across parameter perturbations. Confirmation is retrospective and uses observation counts, not fixed biological times.
6. Merge a newly detected fragment with a partially observed parent only when repeated proximity, predicted parent overlap, common motion and an unambiguous parent agree. Maximum proximity gap is 8 pixels; require at least four proximity observations and three consistent motion intervals. No colony ID or image coordinate is privileged.
7. Partition neighboring territories jointly. Repair narrow opposed seams (nearest distance at most 6 pixels, combined distance at most 10, opposing cosine at most −0.25). Reassign narrow pink remnants and small enclosed pockets. Keep the dominant broad residual compartment and its 12-pixel interface; remote residual pixels compete between established territories.
8. Smooth shared edges with sigma 1 and a 3-pixel band, protecting component cores/topology and limiting area changes to 3%. Finish seams to convergence while preserving the original broad pink core. Independently tracked texture points can reconcile a supported new identity or recover a missing original identity within existing nonpink support.
9. Repair thin exterior-facing pink collars only beside initially exterior-touching colonies with strong full correspondence (fine NCC at least 0.65, coarse NCC at least 0.75). Use width 0.3 times equivalent colony radius, opposing boundary geometry and the side away from the broad pink core. Attach additions to the supporting colony. Absorb severed residual caps. Reuse the pre-repair broad-core protection during final seam closure.

Pink means central residual/unresolved material, not a demonstrated motile-cell class. Every valid in-scope biofilm pixel belongs to one territory or this compartment. Numeric IDs have no `N` prefix. Tracking restarts across capture gaps or excluded focus frames; excluded frames have no inferred identities or breakup measurements.

## Current inner dark-area measurement: 50%

Automatic dispersion now uses `intensity_50` for measured inner area and breakup scores. The verifier's numerical detector is shared through `analysis/colony_regions.py`; its GUI method delegates to the same function.

- Take the native image's 15-pixel-cropped analysis interior.
- Normalize that entire crop independently in each frame to uint8 0–255 (`cv2.normalize`, `NORM_MINMAX`).
- Apply the fixed 5×5 Gaussian blur with sigma argument 0.
- Keep blurred values **<=128**, equivalent to **50%** on the inner dark-cell slider, inside the current outer boundary.
- Morphology kernel 1 is an identity operation. Keep all eight-connected components of at least **1 pixel**.
- Score each final current colony territory as **1 − largest connected dark area / total dark area**. No dark area produces an unavailable score.

This detector reconstructs the original notebook's saved inner masks exactly across all 17 reviewed frames. Outer-boundary settings remain 39%/37 pixels. Local-z evidence used inside territory tracking is unchanged; `raw_dark_mask` remains a separate diagnostic. The measured area, GUI overlay, cell-region map and breakup graph use the new 50% mask.

The tool continues to score its current colony territories. The original notebook's main score uses a transported initial footprint and anchor-selected body, so exact dark-mask reproduction does not imply identical score curves under those different spatial definitions. The motion-support classification experiment is not installed.

The GUI identifies the existing slider as **Boundary threshold** and shows the fixed **Inner dark area: 50% (128/255), minimum area 1, morphology 1** separately. Fresh verifier defaults are also 50/1/1; explicitly saved user settings remain respected there. Existing result folders retain their measurement method and display its corresponding label. Restart the GUI and run a new analysis to obtain the updated measurement; old saved results are not rewritten.

The `dispersion_regions_motion_v5` manifest records the detector's threshold, normalization, blur, morphology, area minimum, connectivity, spatial domain and score formula. `tests/dispersion_reference.json` retains the earlier raw-0.8 checkpoint and adds `inner50_material_sha256` plus the 50%-mask colony-27 reference on the accepted territory stack. The recording regression checks the new inner-mask hashes exactly and the raw diagnostic hashes separately, and independently verifies every graph row against its actual final masks. `PARTAKER_PREVIOUS_DISPERSION` optionally points to a previous result folder for a pixel-exact territory-preservation check.

## Earlier 0.8 measurement (retained for comparison)

For cropped raw intensity I:

- mu = Gaussian(I, sigma=7)
- sd = sqrt(max(0, Gaussian(I², sigma=7) − mu²))
- z = (mu − Gaussian(I, sigma=0.6)) / max(5, sd)
- Keep z > 0.8 inside the current outer boundary; remove eight-connected objects smaller than 4 pixels.
- For each final current colony territory, score = 1 − largest connected dark area / total dark area. No dark area means an unavailable score, not zero.

Background statistics use unsmoothed raw intensity. This was the earlier reviewed detector; it must not silently change to the alternative detector that smooths before calculating background statistics. The measurement is also different from the original notebook's primary score, which uses its saved dark mask and transported initial footprint.

The GUI worker exports numeric rows and final masks, without rendering plots or movies. The existing graph displays the selected colony's data and remains empty until selection. Time navigation stays in Dispersion. Each run records settings, formulas, provenance and identity decisions in its result manifest/CSVs.

## Verification

`tests/dispersion_reference.json` preserves the approved recording's image hash, output-mask hashes, dark-mask hashes, and all 665 reference graph rows. The recording is 17 frames at 66.00–70.00 hours; these timestamps are test data, not rules.

Fast numerical tests: `python -m unittest discover -s tests -p 'test_dispersion*.py'`.

The full TIFF regression is opt-in: set `PARTAKER_TIFF_HOURS` to the reviewed TIFF_Hours directory and run the same command. Native GUI behavior must additionally be tested in a working Qt session.

### Earlier 0.8 integration validation, November 22, 2026

The complete backend path (`prepare_recording` → automatic territories → final masks → numeric export) ran on Python 3.11.15/macOS arm64 with numpy 1.26.4, scipy 1.11.4, OpenCV 4.11.0 and scikit-image 0.25.2. The approved experiment was generated on Linux x86_64/Python 3.12.14 with the same numerical package versions.

| Check | Integrated result |
| --- | --- |
| Frame-0 comparison with approved experiment | Pixel exact; 52 colonies |
| Whole-series agreement with approved territory masks | 99.8681% of biofilm pixels |
| Worst single-frame agreement | 99.4327% |
| Breakup graph rows independently verified against final masks | 664 / 664 |
| Initial-region mean IoU against original notebook | 97.0215% |
| Later initial-colony mean IoU against original notebook | 75.4674% |
| Pink pooled IoU against original notebook | 75.0713% |
| Colony 43 | Observed in all 17 frames; four territory pixels differ from approved experiment across the entire series |
| Colony 27, 66 h | 0.8272552783109405 |
| Colony 27, 70 h | 0.8549848942598187 (approved experiment: 0.8552036199095022) |
| Colony 27 maximum absolute score difference across frames | 0.0050126345858053645 |

Later-frame reproduction is close, not pixel identical. One approved observation (region 45 at frame 5, 67.25 h) is unresolved in this runtime, producing 664 rather than 665 colony observations. Checked fit values differ slightly between environments; forcing a single OpenCV thread did not remove the differences. No identity-specific rescue was added to conceal this discrepancy.

The portable regression retains the approved masks and rows. It requires exact frame-0 geometry and all 17 raw-dark masks, at least 99.5% overall territory agreement and 99% in each frame, at least 99% observation correspondence, and colony-27 score differences at most 0.01. Every exported row must additionally match the actual final masks to 12 decimal places. These comparison tolerances explicitly accommodate the observed runtime differences; they are not a claim of identical masks or validation on other datasets.

Fast checks cover translation, positive/negative structure evidence, merging, unsupported fragments, shared edges, protected residual caps, cancellation, focus/capture gaps, current measurement metadata and GUI selection behavior. Native Qt rendering could not be checked in the execution session because Qt aborts on its NEON processor-feature check. No Qt dependencies or other analysis tabs were changed to work around this.

### 50% inner-area integration validation

Validated on branch `partaker-2-dispersion` with the complete TIFF_Hours recording and default `intensity_50` settings:

- All 22 automated tests passed, including the full recording regression (about 253 seconds).
- All 17 computed inner-area masks match the original notebook saved masks exactly (100% IoU, zero differing pixels). Prepared inner masks and measured masks agree.
- All 17 colony territory maps are pixel-identical to the preceding installed v4 run. The territory-rule source was not modified.
- All 664 exported microcolony breakup rows were independently checked against their final measured masks; raw local-z diagnostic masks also retain their earlier hashes.
- Colony 27 current-territory scores: 0.0 at 66 h and 0.29073482428115016 at 70 h. The notebook's transported-footprint score at 70 h is 0.3393329804129169; the geometry/scoring distinction remains explicit.
- The shared detector matches the previous GUI numerical method exactly in six parameter/dtype controls, including uint16 input.
- Native GUI execution remains unverified in this execution session: Qt aborts with a NEON processor-feature requirement. No Qt dependencies were changed. GUI run-mode wiring is covered by an automated check; actual window rendering still requires a working native Qt session.

Restart the GUI from this checkout and run a fresh dispersion analysis. The existing boundary slider remains at 39%; the fixed inner detector is displayed separately as 50% (128/255). Previously saved result folders retain their original scores and detector labels.

### Analysis setup and display

Import loads the recording. **Run dispersion analysis…** opens a modal setup window with boundary threshold, frame range, and position/frame preview controls. Opening the window computes its initial preview. Moving a slider or changing the preview frame only changes a draft; **Update preview** explicitly recomputes the boundary. Clicking the preview chooses a whole-biofilm seed for that position. **Confirm and run** validates the range and refreshes a stale preview before starting; **Cancel** discards the draft.

The main page shows the completed run's boundary threshold and frame range as text. **Set up new analysis…** opens a prefilled setup window. Existing results and review controls remain available while a replacement runs; cancellation or worker failure leaves them intact. A successful replacement updates the displayed results and settings together. The fixed 50% inner-area detector and all numerical measurements are unchanged.

Both result image views hide the 15-pixel excluded margin. The stage image uses full-frame axes with cropped limits; the main viewer records the crop offset before applying zoom, so colony selection still uses full-frame coordinates. Other tabs retain their normal image bounds. Source images, masks, and the calculation exclusion remain unchanged.

Headless checks cover draft isolation, explicit preview wiring, range validation, worker snapshots, cancellation/failure, incomplete-result rejection, and display-coordinate mapping through ROI intersections and zoom/reset. The setup preview was additionally checked on TIFF_Hours frames 0, 8, and 16: its boundary edges exactly match `whole_colony_mask` at 39%. Native Qt rendering still requires a working GUI session; this execution environment has a Qt NEON processor-feature error. The full 17-frame numerical regression was not repeated for this display-only change.

### Current review overlays

The full-field viewer keeps all territory colors. The selected-colony close-up retains only that colony's color; neighboring territories use the saved grayscale frame. The Breakup overlay colors each microcolony's largest eight-connected measured dark component yellow and its other measured components blue. In the central compartment, only measured dark pixels are blue, not the entire territory.

Position and Show overlays share one visible control row. Colony selection uses image clicks; time selection uses the timeline or breakup graph. Detector details remain in result metadata rather than the review text.

The cyan diamond marks the exact current endpoint of the white trail, rather than a separately recomputed territory centroid. The white tracking trail follows the notebook's reference-point advection approach: accepted local raw-image affine fits move a reference point, with at most six consecutive supported positions. Unsupported fits, missing observations, focus/capture gaps and association ambiguity restart the trail. It is distinct from a segmentation-centroid trajectory or the optional instantaneous texture arrows. The existing checked motion fitter is reused, and fits are cached for review; no scores, masks or saved result files are rewritten. The display stroke is 1.5 points in white with a 3.5-point dark underlay, independent of image zoom. TIFF_Hours reference colony 27 supported 15 of 16 interval fits in the review check; the unsupported interval restarted the trail.
