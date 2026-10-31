"""The twin page's picture: the real cells and the twin's cells at the same frame, side by side, and the count and area
curves of the real cells against the runs. numpy + matplotlib only, shared by the tab, the export and the tests."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from nd2_analyzer.analysis.tracking.film_render import GREY, house_rc, id_colour

SIM = "#1a863a"          # the twin's green (no orange anywhere)
REAL = "#000000"


def real_panel(mask_roi: np.ndarray) -> np.ndarray:
    """The real cells inside the ROI box, each in its own colour, dark seams, black background."""
    from skimage.segmentation import find_boundaries
    m = np.asarray(mask_roi); rgb = np.zeros((*m.shape, 3), np.float32)
    labs = np.unique(m); labs = labs[labs > 0]
    for L in labs:
        rgb[m == L] = id_colour(int(L)) * 0.8
    rgb[find_boundaries(m, mode="inner")] *= 0.35
    return np.clip(rgb, 0, 1)


def cell_colour(aid) -> np.ndarray:
    """One fixed colour per twin cell, the same rule as the real cells' tag colours (hashed from the cell's id)."""
    import hashlib
    return id_colour(int(hashlib.md5(str(aid).encode()).hexdigest()[:8], 16) % 100_000)


def draw_sim(ax, cells, w_um: float, h_um: float, fig_dpi: float):
    """The twin's capsules (x, y, total length, radius, angle in um) on black, to scale, one colour per cell like the real ones."""
    ax.set_facecolor("black"); ax.set_aspect("equal")
    ax.set_xlim(0, w_um); ax.set_ylim(h_um, 0); ax.set_xticks([]); ax.set_yticks([])
    # points per um once the axes has its size (the figure must be laid out first)
    bb = ax.get_window_extent(); ppu = (bb.height / h_um) * 72.0 / fig_dpi if bb.height > 0 else 4.0
    for aid, x, y, L, r, ang in cells:
        half = max(L - 2 * r, 0.0) / 2
        dx, dy = half * np.cos(ang), half * np.sin(ang)
        ax.plot([x - dx, x + dx], [y - dy, y + dy], color=cell_colour(aid) * 0.8, lw=max(2 * r * ppu, 0.5), solid_capstyle="round", solid_joinstyle="round")


def draw_twin(fig, t: int, hours: np.ndarray, mask_roi: Optional[np.ndarray], sim_cells, w_um: float, h_um: float,
              real_counts, real_areas, sim_counts: Optional[np.ndarray] = None, sim_areas: Optional[np.ndarray] = None,
              rules_text: str = "", message: str = ""):
    """One frame of the twin page. sim_counts / sim_areas: runs x frames (None before a run)."""
    import matplotlib
    with matplotlib.rc_context(house_rc()):
        fig.clear()
        gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 0.42, 2.2], left=0.02, right=0.985, top=0.92, bottom=0.14, wspace=0.10, hspace=0.45)
        a1 = fig.add_subplot(gs[:, 0]); a2 = fig.add_subplot(gs[:, 1]); c = fig.add_subplot(gs[0, 3]); d = fig.add_subplot(gs[1, 3])
        rc = np.asarray(real_counts, float); ra = np.asarray(real_areas, float)
        if mask_roi is not None:
            a1.imshow(real_panel(mask_roi), interpolation="nearest", extent=(0, w_um, h_um, 0))
        a1.set_xticks([]); a1.set_yticks([]); a1.set_facecolor("black")
        a1.set_title(f"real cells   {int(rc[t]) if t < len(rc) else ''}", loc="left")
        fig.canvas.draw_idle() if hasattr(fig, "canvas") and fig.canvas is not None else None
        if sim_cells is not None:
            draw_sim(a2, sim_cells, w_um, h_um, fig.dpi)
            a2.set_title(f"twin   {len(sim_cells)}", loc="left")
        else:
            a2.set_facecolor("black"); a2.set_xticks([]); a2.set_yticks([]); a2.set_xlim(0, w_um); a2.set_ylim(h_um, 0); a2.set_aspect("equal")
            a2.set_title("twin   not run yet", loc="left")
            if message:
                a2.text(0.5, 0.5, message, transform=a2.transAxes, color="white", ha="center", va="center", wrap=True)
        c.plot(hours[: len(rc)], rc, "o", color=REAL, ms=3, label="real cells")
        d.plot(hours[: len(ra)], ra / max(ra[0], 1e-9), "o", color=REAL, ms=3)
        if sim_counts is not None and len(sim_counts):
            m, s = sim_counts.mean(0), (sim_counts.std(0, ddof=1) if len(sim_counts) > 1 else np.zeros(sim_counts.shape[1]))
            hh = hours[: len(m)]
            c.fill_between(hh, m - s, m + s, color=SIM, alpha=0.2, lw=0); c.plot(hh, m, color=SIM, lw=1.8, label=f"twin, mean of {len(sim_counts)} run{'s' if len(sim_counts) != 1 else ''}")
            am = sim_areas.mean(0); asd = sim_areas.std(0, ddof=1) if len(sim_areas) > 1 else np.zeros_like(am)
            d.fill_between(hh, (am - asd) / am[0], (am + asd) / am[0], color=SIM, alpha=0.2, lw=0); d.plot(hh, am / am[0], color=SIM, lw=1.8)
        for ax in (c, d):
            ax.axvline(float(hours[t]), color=GREY, lw=1)
        c.set_ylabel("cells in chamber"); c.legend(frameon=False, loc="upper left"); c.set_xticklabels([])
        d.set_ylabel("area / area at 0 h"); d.set_xlabel("time (h)")
        for ax in (a1, a2):
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_linewidth(0.8); sp.set_color("#555555")
        if rules_text:
            fig.text(0.03, 0.015, rules_text, fontsize=8.5, color="#333333")
        return a1, a2, c, d
