"""
Shared publication-figure style for Partaker 2.

Hard rules (non-negotiable, per project standard):
  - no gridlines
  - no top/right spines, no box frame
  - legible fonts (large, no overlap), axis labels with units
  - no image distortion (aspect='equal' for microscopy, never stretch)
  - high resolution (300 dpi raster + vector PDF)
  - tight bounding box on save

Every figure in this project routes through apply_style() + save().
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from pathlib import Path

# A calm, colour-blind-safe palette (Okabe-Ito).
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]


def apply_style():
    """Set global rcParams enforcing the publication rules."""
    plt.rcParams.update({
        # no grid
        "axes.grid": False,
        # no top/right frame
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.1,
        # legible type
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        # ticks outward, readable
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        # resolution
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,   # editable text in vector output
        "ps.fonttype": 42,
        "axes.prop_cycle": plt.cycler(color=PALETTE),
    })


def despine(ax):
    """Remove top/right spines and any grid from a plot axis."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    return ax


def style_image_ax(ax):
    """Configure an axis that shows a microscopy image: no axes, no distortion."""
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


def add_scale_bar(ax, image_width_px, pixel_size_um, bar_um=20, loc="lower right",
                  color="white", pad_frac=0.05, height_frac=0.012, label=True):
    """Draw a real scale bar on an image axis (no axes ticks needed)."""
    if not pixel_size_um or pixel_size_um <= 0:
        return  # cannot place a calibrated bar without a pixel size
    bar_px = bar_um / pixel_size_um
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    H = abs(ylim[0] - ylim[1])
    W = abs(xlim[1] - xlim[0])
    pad_x, pad_y = W * pad_frac, H * pad_frac
    bar_h = H * height_frac
    if "right" in loc:
        x0 = max(xlim) - pad_x - bar_px
    else:
        x0 = min(xlim) + pad_x
    if "lower" in loc:
        y0 = max(ylim) - pad_y - bar_h
    else:
        y0 = min(ylim) + pad_y
    ax.add_patch(plt.Rectangle((x0, y0), bar_px, bar_h, color=color, ec="none"))
    if label:
        ax.text(x0 + bar_px / 2, y0 - bar_h * 1.2, f"{bar_um} µm",
                color=color, ha="center", va="bottom", fontsize=12)


def save(fig, out_path, also_pdf=True):
    """Save a figure to PNG (300 dpi) and a vector PDF, tight bbox."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight",
                facecolor="white")
    if also_pdf:
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    return str(out_path.with_suffix(".png"))


def stretch(img, lo=1, hi=99):
    """Percentile contrast stretch to [0,1] for display (no distortion)."""
    img = img.astype(np.float32)
    p_lo, p_hi = np.percentile(img, lo), np.percentile(img, hi)
    return np.clip((img - p_lo) / (p_hi - p_lo + 1e-6), 0, 1)
