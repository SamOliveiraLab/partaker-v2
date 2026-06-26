"""
Headless Partaker 2 EPS pipeline on Nona's E. coli data.

Calls Jianna's REAL code (eps_analysis.EPSAnalysisWidget Moreau mode +
colony_separator.ColonySeparator) with no GUI. No GPU needed.

Channel map (verified by image inspection):
  file _0 -> phase/bacteria  (our channel index 0; Moreau cell mask uses this)
  file _2 -> EPS dye          (our channel index 1; the ground-truth matrix)
  file _1 -> RFP cells, only at t0 (ignored here)
"""

import os, sys, argparse
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import PySide6
os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH",
                      os.path.join(os.path.dirname(PySide6.__file__), "Qt", "plugins"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import tifffile
from types import SimpleNamespace
from pathlib import Path

import figure_style as fs

DATA = Path("/Volumes/Extreme SSD/Partaker-2/Partaker-Induced-RFP-Ecoli-EPS-NONA-data")
AGES = [0, 24, 48, 96]          # T index 0..3
POSITIONS = [0, 1, 2]           # P index 0..2
CELL_FILE_CH = 0                # file _0 = phase  -> our C0
EPS_FILE_CH = 2                 # file _2 = EPS    -> our C1
PIXEL_SIZE_UM = 0.10            # TODO confirm with Nona; area-fraction is px-size independent

def tif_path(p, age, file_ch):
    return DATA / f"pos{p}-t{age}" / f"pos{p}_t{age}_{file_ch}.tif"

def build_imagedata():
    """Build the (T,P,C,Y,X) array (C0=phase, C1=EPS) and install the ImageData singleton."""
    # reference shape
    ref = tifffile.imread(tif_path(0, 0, CELL_FILE_CH))
    H, W = ref.shape
    arr = np.zeros((len(AGES), len(POSITIONS), 2, H, W), dtype=np.uint16)
    for ti, age in enumerate(AGES):
        for pi, p in enumerate(POSITIONS):
            cp, ep = tif_path(p, age, CELL_FILE_CH), tif_path(p, age, EPS_FILE_CH)
            if cp.exists(): arr[ti, pi, 0] = tifffile.imread(cp)
            if ep.exists(): arr[ti, pi, 1] = tifffile.imread(ep)
    from nd2_analyzer.data.image_data import ImageData
    img = ImageData.create_instance(data=arr, path=str(tif_path(0, 0, EPS_FILE_CH)))
    img.voxel_size = SimpleNamespace(x=PIXEL_SIZE_UM, y=PIXEL_SIZE_UM, z=1.0)
    return img, arr

def make_headless_eps():
    """Rebind EPSAnalysisWidget's REAL algorithm methods onto a plain (non-Qt) object.

    Identical code to the widget, but no QWidget base -> no QApplication / Qt platform
    plugin needed. We copy the Moreau class-constants and the pure algorithm methods.
    """
    from nd2_analyzer.ui.biofilms.eps_analysis import EPSAnalysisWidget as W

    class EPSAlgo:
        pass

    for const in ["MOREAU_LOCAL_AREA_PERCENT", "MOREAU_THRESHOLD_SCOPE",
                  "MOREAU_EPS_OPENING_RADIUS_PX", "MOREAU_EPS_CLOSING_RADIUS_PX",
                  "MOREAU_EPS_DILATION_RADIUS_PX", "MOREAU_SMOOTHING_SIGMA"]:
        setattr(EPSAlgo, const, getattr(W, const))
    for meth in ["fixed_or_otsu_threshold", "get_pixel_size_um", "make_moreau_cell_masks",
                 "blockwise_local_otsu_eps_mask", "apply_eps_channel_morphology",
                 "segment_eps_moreau", "segment_eps_partaker",
                 "is_moreau_method", "is_partaker_method", "get_analysis_method"]:
        setattr(EPSAlgo, meth, getattr(W, meth))

    h = EPSAlgo()
    h.close_dialate = SimpleNamespace(isChecked=lambda: True)    # apply Moreau 2px dilation
    h.gaus_back_corr = SimpleNamespace(isChecked=lambda: False)  # default: no bg subtraction
    h.radio_moreau = SimpleNamespace(isChecked=lambda: True)     # force Moreau mode
    h.radio_partaker = SimpleNamespace(isChecked=lambda: False)
    return h

def run_one(img, h, ti, pi):
    """Run Jianna's Moreau EPS segmentation on one (time-index, position-index)."""
    frame_eps = img.get(ti, pi, 1).astype(float)
    gaussian_corrected = np.clip(frame_eps.copy(), 0, None)   # gaus_back_corr off -> copy
    smoothed, eps_mask, m = h.segment_eps_moreau(
        gaussian_corrected=gaussian_corrected, image=frame_eps, time=ti, position=pi)
    return frame_eps, eps_mask, m

def detect_colonies(img, ti, pi, intensity_threshold=50, min_colony_size=80):
    from nd2_analyzer.ui.biofilms.colony_separator import ColonySeparator
    sep = ColonySeparator()
    sep.update_parameters(intensity_threshold=intensity_threshold, min_colony_size=min_colony_size)
    return sep.detect_colonies_otsu(img.get(ti, pi, 1))

def run_all(img, h):
    """Run the full EPS->colony pipeline over every (age, position). Returns rows + raw stash."""
    import csv
    rows, stash = [], {}
    for ti, age in enumerate(AGES):
        for pi, p in enumerate(POSITIONS):
            frame_eps, eps_mask, m = run_one(img, h, ti, pi)
            colonies = detect_colonies(img, ti, pi)
            areas_um2 = [c["area"] * PIXEL_SIZE_UM**2 for c in colonies]
            eps_occupancy = 100.0 * float(eps_mask.sum()) / eps_mask.size
            rows.append({
                "position": p, "age_h": age,
                "areal_fraction_rho_pct": round(m["areal_fraction_percent"], 3),
                "eps_encased_pct": round(m["fraction_vps_coverage"], 3),
                "eps_occupancy_pct": round(eps_occupancy, 3),
                "eps_threshold": round(m["eps_threshold_used"], 2),
                "n_colonies": len(colonies),
                "mean_colony_area_um2": round(float(np.mean(areas_um2)), 3) if areas_um2 else 0.0,
            })
            stash[(age, p)] = {"eps_raw": frame_eps, "eps_mask": eps_mask, "areas_um2": areas_um2, "colonies": colonies}
            print(f"  done age={age:>2}h pos={p}  rho={m['areal_fraction_percent']:.1f}%  "
                  f"encased={m['fraction_vps_coverage']:.1f}%  colonies={len(colonies)}")
    out = Path(__file__).parent / "out"
    with open(out / "eps_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("  wrote", out / "eps_metrics.csv")
    return rows, stash

def _series_by_age(rows, key):
    """Return (ages, list-of-values-per-age) for mean/sd plotting."""
    per = {a: [] for a in AGES}
    for r in rows:
        per[r["age_h"]].append(r[key])
    return AGES, [per[a] for a in AGES]

def figure_dynamics(rows, key, ylabel, title, fname):
    import matplotlib.pyplot as plt
    fs.apply_style()
    ages, groups = _series_by_age(rows, key)
    means = [np.mean(g) for g in groups]; sds = [np.std(g) for g in groups]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.errorbar(ages, means, yerr=sds, fmt="-o", color=fs.PALETTE[0], lw=2.2, ms=8,
                capsize=5, capthick=1.6, ecolor=fs.PALETTE[0], zorder=3)
    for a, g in zip(ages, groups):                          # individual positions as dots
        ax.scatter([a]*len(g), g, color=fs.PALETTE[1], s=34, alpha=0.7, zorder=4)
    ax.set_xlabel("biofilm age (hours)"); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.set_xticks(ages); ax.set_xlim(-4, 100); ax.set_ylim(bottom=0)
    fs.despine(ax)
    return fs.save(fig, Path(__file__).parent / "out" / fname)

def figure_colony_area_dist(stash, fname):
    import matplotlib.pyplot as plt
    fs.apply_style()
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    data = []
    for i, age in enumerate(AGES):
        pooled = []
        for p in POSITIONS:
            pooled += stash[(age, p)]["areas_um2"]
        data.append(np.log10(np.array(pooled) + 1e-9) if pooled else np.array([]))
    parts = ax.violinplot([d for d in data], positions=range(len(AGES)), showmeans=True, widths=0.8)
    for b in parts["bodies"]:
        b.set_facecolor(fs.PALETTE[0]); b.set_alpha(0.55); b.set_edgecolor("black")
    for key_ in ("cmeans", "cmaxes", "cmins", "cbars"):
        if key_ in parts: parts[key_].set_color("black")
    ax.set_xticks(range(len(AGES))); ax.set_xticklabels([f"{a} h" for a in AGES])
    ax.set_xlabel("biofilm age"); ax.set_ylabel("colony area  (log₁₀ µm²)")
    ax.set_title("Per-colony size distribution over time")
    fs.despine(ax)
    return fs.save(fig, Path(__file__).parent / "out" / fname)

def figure_raw_to_quant(stash, position, fname):
    """Sam's 'raw -> quantify' visual: EPS raw (top) and EPS mask (bottom) across ages."""
    import matplotlib.pyplot as plt
    fs.apply_style()
    fig, ax = plt.subplots(2, len(AGES), figsize=(4*len(AGES), 8))
    for j, age in enumerate(AGES):
        s = stash[(age, position)]
        ax[0, j].imshow(fs.stretch(s["eps_raw"]), cmap="gray"); ax[0, j].set_title(f"{age} h")
        ax[1, j].imshow(s["eps_mask"], cmap="magma")
        fs.style_image_ax(ax[0, j]); fs.style_image_ax(ax[1, j])
    ax[0, 0].set_ylabel("EPS channel", fontsize=14); ax[1, 0].set_ylabel("EPS mask", fontsize=14)
    fs.add_scale_bar(ax[0, 0], stash[(AGES[0], position)]["eps_raw"].shape[1], PIXEL_SIZE_UM)
    return fs.save(fig, Path(__file__).parent / "out" / fname)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--one", action="store_true", help="run only pos2/t48 as a validation")
    args = ap.parse_args()

    print("Loading Nona data + installing ImageData singleton ...")
    img, arr = build_imagedata()
    print("  array shape (T,P,C,Y,X):", arr.shape)
    h = make_headless_eps()

    if args.one:
        ti, pi = AGES.index(48), POSITIONS.index(2)
        print(f"\n=== validation run: pos2, t48 (T={ti}, P={pi}) ===")
        frame_eps, eps_mask, m = run_one(img, h, ti, pi)
        print("  metrics:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()})

        # colony detection on EPS channel (real ColonySeparator)
        from nd2_analyzer.ui.biofilms.colony_separator import ColonySeparator
        sep = ColonySeparator(); sep.update_parameters(intensity_threshold=50, min_colony_size=80)
        colonies = sep.detect_colonies_otsu(img.get(ti, pi, 1))
        print(f"  colonies detected: {len(colonies)}")

        # overlay figure (publication style)
        import matplotlib.pyplot as plt
        fs.apply_style()
        fig, ax = plt.subplots(1, 3, figsize=(16, 6))
        ax[0].imshow(fs.stretch(frame_eps), cmap="gray"); ax[0].set_title("EPS channel (raw)")
        ax[1].imshow(eps_mask, cmap="magma"); ax[1].set_title("EPS mask (Moreau)")
        ax[2].imshow(fs.stretch(frame_eps), cmap="gray")
        for c in colonies:
            cnt = c["contour"]; ax[2].plot(np.append(cnt[:,0], cnt[0,0]), np.append(cnt[:,1], cnt[0,1]), lw=1.2, color="#00E5FF")
        ax[2].set_title(f"colonies (n={len(colonies)})")
        for a in ax: fs.style_image_ax(a)
        fs.add_scale_bar(ax[0], frame_eps.shape[1], PIXEL_SIZE_UM)
        out = fs.save(fig, Path(__file__).parent / "out" / "validate_pos2_t48")
        print("  saved overlay:", out)
        print("\nVALIDATION OK")
    else:
        print("\n=== FULL RUN: all ages x positions ===")
        rows, stash = run_all(img, h)
        print("\nBuilding publication figures ...")
        f1 = figure_dynamics(rows, "eps_encased_pct",
                             "EPS-encased fraction (%)",
                             "Biofilm matrix coverage over time",
                             "fig1_eps_encased_vs_age")
        f2 = figure_dynamics(rows, "areal_fraction_rho_pct",
                             "cell areal fraction ρ (%)",
                             "Cell coverage over time",
                             "fig2_areal_fraction_vs_age")
        f3 = figure_dynamics(rows, "n_colonies",
                             "number of colonies",
                             "Colony count over time",
                             "fig3_colony_count_vs_age")
        f4 = figure_colony_area_dist(stash, "fig4_colony_area_distribution")
        f5 = figure_raw_to_quant(stash, position=2, fname="fig5_raw_to_quant_pos2")
        print("  figures:", f1, f2, f3, f4, f5, sep="\n   ")
        print("\nFULL RUN OK")
