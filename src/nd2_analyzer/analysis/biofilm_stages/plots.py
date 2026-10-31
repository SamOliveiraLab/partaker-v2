"""Drawing for the biofilm stage pages, in the journal style (one message per panel, markers at every frame, Okabe-Ito
colours, light grey stages). numpy + matplotlib only; the pages, the exports and the tests draw through these."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from nd2_analyzer.analysis.tracking.film_render import house_rc
from .study import BiofilmStudy, Curve, StageHours

BLUE, GREEN, GREY, INK, CURSOR = "#0072B2", "#009E73", "#7f7f7f", "#1a1a1a", "#555555"
STAGE = ["#f4f4f4", "#e4e4e4", "#d4d4d4"]
PASTEL = np.array([[0.35, 0.60, 0.85], [0.30, 0.75, 0.55], [0.80, 0.50, 0.75], [0.60, 0.70, 0.90], [0.90, 0.80, 0.35],
                   [0.45, 0.80, 0.85], [0.85, 0.55, 0.60], [0.55, 0.55, 0.80]])
MK = dict(marker="o", ms=4.5, mfc="white", mew=1.6)
MK_REF = dict(marker="s", ms=3.8, mfc="white", mew=1.3, ls="--")


def colony_colour(k: int) -> np.ndarray:
    return PASTEL[int(k) % len(PASTEL)]


# ---------------------------------------------------------------------------------------------- the strip
def draw_strip(ax, sh: StageHours, cursor_h: float, highlight: Optional[str] = None):
    """The spine of every stage page: hours, stages shaded, Th1 / Th2, the attachment block, the cursor."""
    from matplotlib.patches import Rectangle
    x0, x1 = sh.first_h - 4.5, sh.last_h + 2
    ax.set_xlim(x0, x1); ax.set_ylim(0, 1); ax.set_yticks([])
    ticks = [t for t in range(int(sh.first_h), int(sh.last_h) + 1, 6)]
    ax.set_xticks(ticks); ax.tick_params(axis="x", labelsize=9, length=4)
    for s in ("left", "top", "right"): ax.spines[s].set_visible(False)
    ax.add_patch(Rectangle((x0 + 0.3, 0.2), 3.0, 0.6, color="#e9e9e9"))
    ax.text(x0 + 1.8, 0.5, "attachment\n3 h", ha="center", va="center", fontsize=7.5, color="#555555")
    stages = sh.stages()
    for i, (name, a, b) in enumerate(stages):
        ax.add_patch(Rectangle((a, 0.2), b - a, 0.6, color=STAGE[min(i + 1, 2)], ec="none"))
        label = name + ("   (Th2 not reached: growth has not stopped by the end)" if sh.th2_h is None and i == len(stages) - 1 else "")
        ax.text((a + b) / 2, 0.5, label, ha="center", va="center", fontsize=8, color=INK if highlight is None or name.split()[-1].lower().startswith(highlight) else "#777777")
    if sh.th1_h is not None:
        ax.axvline(sh.th1_h, color=INK, ls="--", lw=1.2); ax.text(sh.th1_h - 0.3, 0.86, f"Th1  {sh.th1_h:.1f} h", ha="right", va="bottom", fontsize=8.5, fontweight="bold")
    if sh.th2_h is not None:
        ax.axvline(sh.th2_h, color=INK, ls="--", lw=1.2); ax.text(sh.th2_h - 0.3, 0.86, f"Th2  {sh.th2_h:.1f} h", ha="right", va="bottom", fontsize=8.5, fontweight="bold")
    ax.axvline(cursor_h, color=BLUE, lw=2.2); ax.text(cursor_h + 0.3, 0.86, f"{cursor_h:.1f} h", ha="left", va="bottom", fontsize=8.5, color=BLUE, fontweight="bold")
    ax.text(1.0, 1.0, "time after inoculation (h)", transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="#555555")


# ---------------------------------------------------------------------------------------------- helpers
def _curve(ax, c: Curve, colour, label=None, ref=False, band=True):
    if band and c.sd is not None:
        ax.fill_between(c.hours, c.mean - c.sd, c.mean + c.sd, color=colour, alpha=0.15, lw=0)
    ax.plot(c.hours, c.mean, color=colour, lw=2, label=label, **(MK_REF if ref else MK))


def _cursor_value(ax, c: Curve, cursor_h, colour, fmt="{:.0f}", suffix="", dy=-22):
    k = int(np.argmin(np.abs(c.hours - cursor_h)))
    ax.annotate(fmt.format(c.mean[k]) + suffix, (c.hours[k], c.mean[k]), xytext=(8, dy), textcoords="offset points", fontsize=9, color=colour, fontweight="bold")


def _stage_bands(ax, sh: StageHours, ymax):
    from matplotlib.patches import Rectangle
    for i, (name, a, b) in enumerate(sh.stages()):
        ax.add_patch(Rectangle((a, 0), b - a, ymax, color=STAGE[min(i + 1, 2)], zorder=0))
        ax.text(a + 0.5, ymax * 0.95, name.split()[0], fontsize=11, fontweight="bold", color="#666666", va="center")


def _lines(ax, sh: StageHours, cursor_h):
    if sh.th1_h is not None: ax.axvline(sh.th1_h, color=INK, ls="--", lw=1)
    if sh.th2_h is not None: ax.axvline(sh.th2_h, color=INK, ls="--", lw=1)
    ax.axvline(cursor_h, color=CURSOR, lw=1)
    ax.set_xlim(sh.first_h - 1, sh.last_h + 1)


def gray(ph):
    lo, hi = np.percentile(ph, (1, 99.5)); return np.clip((np.asarray(ph, float) - lo) / max(hi - lo, 1e-6), 0, 1)


def colony_view(fr: Dict, layers=("cells", "colonies", "matrix")) -> np.ndarray:
    """The image panel: phase on grey; every cell in its own tint when 'cells' is on; microcolonies one colour each; matrix outlined."""
    from skimage.segmentation import find_boundaries
    ph = fr.get("phase")
    if ph is None:
        base = np.zeros((*fr["cells"].shape, 3)) if "cells" in fr else np.zeros((64, 64, 3))
    else:
        base = np.repeat(gray(ph)[..., None], 3, -1) * 0.85
    rgb = base.copy()
    if "cells" in layers and "cells" in fr and "colonies" not in layers:
        cells = fr["cells"]
        for lab in np.unique(cells):
            if lab: sel = cells == lab; rgb[sel] = 0.4 * rgb[sel] + 0.6 * colony_colour(lab * 31)
        rgb[find_boundaries(cells, mode="inner")] *= 0.5
    if "colonies" in layers and "colonies" in fr:
        col = fr["colonies"]
        for k in np.unique(col):
            if k: sel = col == k; rgb[sel] = 0.45 * rgb[sel] + 0.55 * colony_colour(k)
    if "matrix" in layers and "matrix" in fr and fr["matrix"].any():
        rgb[find_boundaries(fr["matrix"], mode="inner")] = np.array([0, 158, 115]) / 255
    if "dye" in layers and "dye" in fr:
        d = fr["dye"] / 255.0; rgb = np.clip(rgb * 0.6 + np.stack([np.zeros_like(d), d, d * 0.7], -1), 0, 1)
    return np.clip(rgb, 0, 1)


def scalebar(ax, h, um, pixel_um):
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((20, h - 30), um / pixel_um, 6, color="w")); ax.text(20, h - 38, f"{um:g} um", color="w", fontsize=8)


# ---------------------------------------------------------------------------------------------- pages
def draw_maturation(fig, study: BiofilmStudy, cursor_h: float, fr: Optional[Dict], p: int, layers=("cells", "colonies", "matrix")):
    """Growth states: the image at the cursor, the main panel (cells in microcolonies with the stages), matrix gives Th1,
    cell-area rate gives Th2."""
    import matplotlib
    sh = study.stage_hours()
    with matplotlib.rc_context(house_rc()):
        fig.clear()
        a = fig.add_axes([0.02, 0.10, 0.40, 0.84]); a.set_xticks([]); a.set_yticks([])
        if fr is not None and ("phase" in fr or "cells" in fr):
            img = colony_view(fr, layers); a.imshow(img, interpolation="nearest"); scalebar(a, img.shape[0], 10, study.pixel_um)
            n = len([k for k in np.unique(fr.get("colonies", [])) if k])
            a.set_title(f"position {p} at {fr['hour']:.1f} h   {n} microcolonies in view", loc="left")
        else:
            a.set_facecolor("black"); a.text(0.5, 0.5, "no frame: open the recording and the masks\n(Import study)", color="w", ha="center", va="center", transform=a.transAxes)
        inc = study.cells_in_microcolonies(); mat = study.matrix_pct(); rate = study.area_rate()
        b = fig.add_axes([0.52, 0.52, 0.45, 0.38]); c = fig.add_axes([0.52, 0.10, 0.19, 0.22]); d = fig.add_axes([0.78, 0.10, 0.19, 0.22])
        if inc is not None:
            _stage_bands(b, sh, 112); _curve(b, inc, BLUE); _cursor_value(b, inc, cursor_h, BLUE, suffix=f" % at {cursor_h:.0f} h")
            b.set_ylim(0, 112); b.set_ylabel("cells in microcolonies (%)"); b.set_xlabel("time after inoculation (h)")
            b.set_title("cells in microcolonies, mean with SD of the kept positions", loc="left", fontsize=9.5, fontweight="normal", color=CURSOR)
        if mat is not None:
            _curve(c, mat, GREEN); c.set_ylabel("matrix (% of frame)"); c.set_xlabel("time (h)"); c.set_title("gives Th1", loc="left", fontsize=9.5, fontweight="normal", color=CURSOR)
            _cursor_value(c, mat, cursor_h, GREEN, fmt="{:.1f}", suffix=" %", dy=14)
        if rate is not None:
            _curve(d, rate, GREY); d.set_ylabel("cell area rate (% per h)"); d.set_xlabel("time (h)"); d.set_title("gives Th2", loc="left", fontsize=9.5, fontweight="normal", color=CURSOR)
            d.axhline(0.1 * rate.mean.max(), color=GREY, ls="--", lw=1); d.text(sh.last_h, 0.1 * rate.mean.max(), "a tenth of the peak", fontsize=8, color=GREY, va="bottom", ha="right")
        for ax in (b, c, d): _lines(ax, sh, cursor_h)
    return {"image": a, "main": b, "matrix": c, "rate": d}


def draw_microcolonies(fig, study: BiofilmStudy, cursor_h: float, fr: Optional[Dict], p: int):
    """The rule in pictures: the colony view at the cursor, colonies per field with the merge point, grouped against crowded,
    mergers per hour and the gap to the nearest neighbour."""
    import matplotlib
    sh = study.stage_hours()
    with matplotlib.rc_context(house_rc()):
        fig.clear()
        a = fig.add_axes([0.02, 0.10, 0.40, 0.84]); a.set_xticks([]); a.set_yticks([])
        if fr is not None and "cells" in fr:
            img = colony_view(fr, ("colonies",)); a.imshow(img, interpolation="nearest"); scalebar(a, img.shape[0], 10, study.pixel_um)
            n = len([k for k in np.unique(fr.get("colonies", [])) if k]); a.set_title(f"position {p} at {fr['hour']:.1f} h   {n} microcolonies, one colour each", loc="left")
        else:
            a.set_facecolor("black"); a.text(0.5, 0.5, "no frame", color="w", ha="center", va="center", transform=a.transAxes)
        b = fig.add_axes([0.52, 0.58, 0.45, 0.32]); c = fig.add_axes([0.52, 0.10, 0.19, 0.32]); d = fig.add_axes([0.78, 0.10, 0.19, 0.32])
        npf = study.colonies_per_field(); inc = study.cells_in_microcolonies(); rnd = study.cells_in_microcolonies_random()
        if npf is not None:
            _curve(b, npf, BLUE); b.set_ylabel("microcolonies per field"); b.set_title("the count peaks, then the colonies merge into one layer", loc="left", fontsize=9.5, fontweight="normal", color=CURSOR)
            if sh.merge_h: b.axvline(sh.merge_h, color=INK, ls=":", lw=1.2); b.text(sh.merge_h + 0.5, npf.mean.max() * 0.95, f"merge point {sh.merge_h:.0f} h", fontsize=8.5, va="top")
            _cursor_value(b, npf, cursor_h, BLUE, fmt="{:.1f}", dy=10); b.set_xlabel("time after inoculation (h)")
        if inc is not None and rnd is not None:
            _curve(c, inc, BLUE, label="observed"); _curve(c, rnd, GREY, label="same cells placed at random", ref=True, band=False)
            c.set_ylabel("cells in microcolonies (%)"); c.set_xlabel("time (h)"); c.legend(loc="lower right", fontsize=8); c.set_ylim(0, 105)
            c.set_title("grouped, or just crowded", loc="left", fontsize=9.5, fontweight="normal", color=CURSOR)
        ev = study.events(); dyn = study.dynamics()
        if ev is not None and len(ev):
            kept = study.kept(); e = ev[ev.position.isin(kept)] if kept else ev
            g = e.groupby("hours"); m = g.lost_by_merging.mean() / g.dt_h.mean(); s = (g.lost_by_merging.std(ddof=1) / g.dt_h.mean()).fillna(0)
            cm = Curve(m.index.values.astype(float), m.values, s.values)
            _curve(d, cm, GREY); d.set_ylabel("mergers per hour"); d.set_xlabel("time (h)")
            d.set_title("microcolonies lost by merging", loc="left", fontsize=9.5, fontweight="normal", color=CURSOR)
            if dyn.get("merge_peak_h"): d.text(dyn["merge_peak_h"] + 0.5, dyn["merge_peak_per_h"], f"peak {dyn['merge_peak_per_h']:.1f} per h", fontsize=8.5, va="bottom")
        for ax in (b, c, d): _lines(ax, sh, cursor_h)
    return {"image": a, "count": b, "random": c, "mergers": d}


def draw_matrix_view(fig, study: BiofilmStudy, cursor_h: float, fr: Optional[Dict], p: int, threshold: int = 40, crop: int = 800):
    """The four premise tiles on one crop at the cursor's hour, the threshold's cost at this hour, the position gate."""
    import matplotlib
    from matplotlib.patches import Rectangle
    sh = study.stage_hours()
    with matplotlib.rc_context(house_rc()):
        fig.clear()
        if fr is None or "cells" not in fr:
            ax = fig.add_axes([0.1, 0.3, 0.8, 0.4]); ax.axis("off"); ax.text(0.5, 0.5, "no frame: open the recording and the masks", ha="center", va="center"); return {}
        H, W = fr["cells"].shape; r0 = max(H // 2 - crop // 2, 0); c0 = max(W // 2 - crop // 2, 0); sl = (slice(r0, r0 + crop), slice(c0, c0 + crop))
        sub = {k: (v[sl] if isinstance(v, np.ndarray) and v.ndim == 2 else v) for k, v in fr.items()}
        dye = sub.get("dye"); matrix = sub.get("matrix")
        tiles = [("Cell", colony_view(sub, ("cells",)), f"{len(np.unique(sub['cells'])) - 1} cells"),
                 ("Microcolony", colony_view(sub, ("colonies",)), f"{len([k for k in np.unique(sub.get('colonies', [])) if k])} microcolonies")]
        if dye is not None:
            tiles.append(("EPS", np.stack([np.zeros_like(dye, float), dye / 255.0, dye / 255.0 * 0.6], -1), f"mean dye {dye.mean():.0f} (8 bit)"))
            thr = np.zeros((*dye.shape, 3)); thr[dye >= threshold] = np.array([0.25, 0.45, 0.38])
            if matrix is not None: thr[matrix] = np.array([0.2, 0.92, 0.65])
            tiles.append(("Threshold", thr, f"matrix {100 * matrix.mean():.1f} % of the crop" if matrix is not None else f"dye >= {threshold}"))
        axes = {}
        for i, (name, img, sub_t) in enumerate(tiles):
            ax = fig.add_axes([0.03 + i * 0.24, 0.40, 0.22, 0.52]); ax.imshow(img, interpolation="nearest"); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{name}   {sub_t}", loc="left", fontsize=9.5)
            if i == 0: scalebar(ax, img.shape[0], 10, study.pixel_um)
            axes[name] = ax
        e = fig.add_axes([0.07, 0.09, 0.38, 0.22])
        if dye is not None:
            T = np.arange(10, 90, 5); frac = [100 * (dye >= x).mean() for x in T]
            e.plot(T, frac, color=GREEN, lw=2, **MK); e.axvline(threshold, color=INK, ls="--", lw=1.2)
            e.text(threshold + 2, max(frac) * 0.92, f"{threshold} = mean + 5 SD of the 24 h dye\n(validated: at 24 h the dye is background)" if threshold == 40 else f"{threshold}: a what-if; 40 is the validated value", fontsize=8.5)
            e.set_xlabel("threshold on the processed dye (8 bit)"); e.set_ylabel("dye above it (% of crop)"); e.set_title("the cost of the threshold, at this hour", loc="left", fontsize=9.5, fontweight="normal", color=CURSOR)
        g = fig.add_axes([0.54, 0.09, 0.43, 0.22]); gate = study.gate(); tab = gate.get("table", {}); kept = set(gate.get("kept", []))
        g.set_title("position gate: dye mask never over 90 % of the frame, dye contrast never under 2.3", loc="left", fontsize=9.5, fontweight="normal", color=CURSOR)
        n = len(tab)
        for i, (key, row) in enumerate(sorted(tab.items(), key=lambda kv: int(kv[0]))):
            ok = int(key) in kept; g.add_patch(Rectangle((i, 0), 0.92, 1, color="#e3f1ea" if ok else "#ececec"))
            g.text(i + 0.46, 0.72, f"P{key}", ha="center", fontsize=8.5, fontweight="bold"); g.text(i + 0.46, 0.42, f"{row['mask_max_pct']:.0f} %", ha="center", fontsize=7.5)
            g.text(i + 0.46, 0.15, f"{row['contrast_min']:.2f}", ha="center", fontsize=7.5, color=INK if ok else GREY)
        g.set_xlim(0, max(n, 1)); g.set_ylim(0, 1); g.axis("off")
        g.text(0, -0.1, "top: largest dye mask; bottom: lowest dye contrast. Green kept, grey dropped.", transform=g.transAxes, fontsize=7.5, color="#555555", va="top")
        axes["cost"] = e; axes["gate"] = g
    return axes


def draw_matrix_density(fig, study: BiofilmStudy, cursor_h: float):
    import matplotlib
    sh = study.stage_hours()
    with matplotlib.rc_context(house_rc()):
        fig.clear()
        a = fig.add_axes([0.08, 0.15, 0.38, 0.72]); b = fig.add_axes([0.58, 0.15, 0.38, 0.72])
        cov = study.cells_covered_pct(); mpc = study.matrix_per_cell(); corr = study.tile_correlation()
        if cov is not None:
            _curve(a, cov, GREEN); a.set_ylabel("cells covered by matrix (%)"); a.set_xlabel("time after inoculation (h)"); a.set_ylim(0, 105)
            a.set_title("how many cells the matrix covers", loc="left", fontsize=9.5, fontweight="normal", color=CURSOR); _cursor_value(a, cov, cursor_h, GREEN, suffix=" %")
        if corr is not None:
            _curve(b, corr, GREY); b.set_ylabel("dye against cells, 4 um tiles (Spearman r)"); b.set_xlabel("time after inoculation (h)"); b.set_ylim(-0.2, 1)
            b.axhline(0, color=GREY, lw=0.8); b.set_title("does the matrix sit where the cells are", loc="left", fontsize=9.5, fontweight="normal", color=CURSOR)
        for ax in (a, b): _lines(ax, sh, cursor_h)
    return {"covered": a, "corr": b}


def draw_seeding(fig, study: BiofilmStudy, cursor_h: float):
    """Seeding: the 3 h frame of our recording (after attachment) and the lag-time comparison with Lee et al. on their data."""
    import matplotlib
    lit = study.literature()
    with matplotlib.rc_context(house_rc()):
        fig.clear()
        a = fig.add_axes([0.02, 0.10, 0.40, 0.84]); a.set_xticks([]); a.set_yticks([])
        f3 = study.frame_3h()
        if f3 is not None and "phase" in f3:
            a.imshow(gray(f3["phase"]), cmap="gray", interpolation="nearest"); scalebar(a, f3["phase"].shape[0], 10, study.pixel_um)
            a.set_title("our E. coli at 3 h, after attachment (phase contrast)", loc="left")
        else:
            a.set_facecolor("black"); a.text(0.5, 0.5, "no 3 h frame in this study", color="w", ha="center", va="center", transform=a.transAxes)
        b = fig.add_axes([0.55, 0.15, 0.42, 0.70])
        lags = lit.get("lag_times", [])
        if lags:
            pub = np.array([r["published_h"] for r in lags]); ptk = np.array([r["partaker_h"] for r in lags])
            lim = (min(pub.min(), ptk.min()) - 2, max(pub.max(), ptk.max()) + 2)
            b.plot(lim, lim, color=GREY, lw=1, ls="--"); b.plot(pub, ptk, "o", color=BLUE, ms=6, mfc="white", mew=1.8)
            for r in lags: b.annotate(r["recording"].split(" (")[0], (r["published_h"], r["partaker_h"]), xytext=(6, 4), textcoords="offset points", fontsize=7.5, color="#444444")
            b.set_xlabel("lag time, Lee et al. 2018 (h)"); b.set_ylabel("lag time, Partaker on their masks (h)"); b.set_xlim(*lim); b.set_ylim(*lim); b.set_aspect("equal")
            b.set_title(f"the surface population takes off: {len(lags)} recordings, all within {np.abs(pub - ptk).max():.2f} h", loc="left", fontsize=9.5, fontweight="normal", color=CURSOR)
        else:
            b.axis("off"); b.text(0.5, 0.5, "import the literature file to compare lag times", ha="center", va="center")
    return {"image": a, "lag": b}


def draw_dispersion(fig, study: BiofilmStudy, cursor_h: float):
    """Dispersion is not measured in this run: say so, then show the candidate signals the tool tracks."""
    import matplotlib
    sh = study.stage_hours()
    with matplotlib.rc_context(house_rc()):
        fig.clear()
        d1 = study.dispersion_table("D1"); d2 = study.dispersion_table("D2")
        a = fig.add_axes([0.08, 0.15, 0.38, 0.70]); b = fig.add_axes([0.58, 0.15, 0.38, 0.70])
        if d1 is not None and "free_cells_m" in d1:
            c = Curve(d1.hours.values.astype(float), d1.free_cells_m.values.astype(float), d1.free_cells_s.fillna(0).values.astype(float))
            _curve(a, c, GREY); a.set_ylabel("free cells per field"); a.set_xlabel("time after inoculation (h)")
            a.set_title("D1  cells outside any microcolony (would rise at dispersion)", loc="left", fontsize=9.5, fontweight="normal", color=CURSOR)
            _lines(a, sh, cursor_h)
        else:
            a.axis("off"); a.text(0.5, 0.5, "no D1 table", ha="center")
        if d2 is not None and "disappeared" in d2:
            b.plot(d2.hours, d2.disappeared, color=GREY, lw=2, label="microcolonies that disappear", **MK); b.plot(d2.hours, d2.new, color=BLUE, lw=2, label="new microcolonies", **MK_REF)
            b.set_ylabel("per 1.5 h, all kept positions"); b.set_xlabel("time after inoculation (h)"); b.legend(loc="upper right", fontsize=8)
            b.set_title("D2  microcolonies that vanish, against new ones", loc="left", fontsize=9.5, fontweight="normal", color=CURSOR); _lines(b, sh, cursor_h)
        else:
            b.axis("off"); b.text(0.5, 0.5, "no D2 table", ha="center")
    return {"D1": a, "D2": b}
