#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Interactive STM lattice-strain analysis using geometric phase analysis (GPA).

The reported strain is relative to a user-selected reference region.  This is
important for STM: scanner calibration, thermal drift, creep, and affine image
distortion cannot be separated reliably from a spatially uniform lattice strain
in a single topograph.
"""

from __future__ import annotations

from itertools import permutations, product
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np

# nanonispy <= 1.1 still uses NumPy aliases removed in NumPy 2.x.
if not hasattr(np, "float"):
    np.float = np.float64
if not hasattr(np, "int"):
    np.int = np.int_
if not hasattr(np, "bool"):
    np.bool = np.bool_

import matplotlib

# The application itself uses Tkinter.  Explicitly use Matplotlib's Tk backend
# as well; otherwise machines with PySide/PyQt installed may auto-select a Qt
# backend.  Mixing a Tk main loop with Qt figures is especially unreliable in
# a PyInstaller windowed executable and can make loading a file appear to do
# nothing when the first plot is created.
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, hsv_to_rgb
from matplotlib.patches import Circle, Rectangle
from matplotlib.widgets import Button, RectangleSelector, Slider
import nanonispy as nap
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    correlate,
    gaussian_filter,
    generic_filter,
    map_coordinates,
    maximum_filter,
)
from scipy.interpolate import griddata
from scipy.optimize import least_squares, linear_sum_assignment
from scipy.signal.windows import tukey
from skimage.restoration import unwrap_phase


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------
img: np.ndarray | None = None
img_full: np.ndarray | None = None
scan_images_full: dict[str, np.ndarray] = {}
img_processed: np.ndarray | None = None
fft_shifted: np.ndarray | None = None
source_path: Path | None = None

# Peak coordinates are (row, column), and may be sub-pixel values.
peaks_selected: list[np.ndarray] | None = None
peak_widths: list[dict[str, np.ndarray | float]] | None = None
recommended_sigma: float | None = None
reciprocal_condition_number: float | None = None
reference_roi: tuple[int, int, int, int] | None = None  # y1, y2, x1, x2
reference_selection_fig = None
unit_cell_selection_fig = None
unit_cell_origin_xy: np.ndarray | None = None
unit_cell_basis_xy: np.ndarray | None = None
unit_cell_vertices_xy: np.ndarray | None = None
unit_cell_snap_fraction: float | None = None
analysis_region_fig = None
analysis_roi_full: tuple[int, int, int, int] | None = None
analysis_region_confirmed = False
full_scan_size_xy = (1.0, 1.0)

# Physical pixel sizes (dx, dy), normally metres/pixel for SXM data.
pixel_size_xy = (1.0, 1.0)
scan_size_xy = (1.0, 1.0)
spatial_unit = "pixel"
zoom_factor = 1

fig_res = None
result_artists: dict[str, object] = {}
last_results: dict[str, np.ndarray] | None = None
sigma_slider = None
smooth_slider = None
median_slider = None
app_closing = False
lattice_a_var = None
lattice_b_var = None
lattice_angle_var = None
scan_info_var = None
auto_mask_var = None
analysis_channel_var = None
analysis_method_var = None
scan_direction_var = None
source_channel = ""
active_scan_direction = "forward"
direction_states: dict[str, dict] = {}
backward_x_flipped = False
reference_correction_var = None
strain_reference_points_xy_nm: list[np.ndarray] = []
strain_reference_radius_nm = 0.5
strain_reference_selection_fig = None

# Validation defaults.  They are deliberately explicit so advanced users can
# change them without altering the physical calculation functions.
CONFIDENCE_THRESHOLD = 0.18
PHASE_JUMP_DILATION = 2
DERIVATIVE_MASK_EROSION = 2
DERIVATIVE_RADIUS = 2
MAX_ABS_STRAIN = 0.20  # small-strain GPA is not reliable beyond about 20%


def configure_plot_fonts() -> None:
    """Choose an installed CJK font while keeping equations in MathText."""
    installed = {font.name for font in font_manager.fontManager.ttflist}
    candidates = (
        "Microsoft YaHei",
        "Microsoft YaHei UI",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
    )
    chosen = next((name for name in candidates if name in installed), "DejaVu Sans")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [chosen, "DejaVu Sans"]
    plt.rcParams["mathtext.fontset"] = "dejavusans"
    plt.rcParams["axes.unicode_minus"] = False


configure_plot_fonts()


gwyddion_cmap = LinearSegmentedColormap.from_list(
    "gwyddion_fft",
    [
        (0.0, "black"),
        (0.2, "#220033"),
        (0.4, "#550088"),
        (0.6, "#aa00aa"),
        (0.8, "#ff66cc"),
        (1.0, "#ffffff"),
    ],
)

# Nanonis "Nanox" palette for real-space STM/topography displays.  This only
# maps raw scalar values to screen colours; all analysis continues to use the
# unchanged floating-point image arrays.
nanox_cmap = LinearSegmentedColormap.from_list(
    "nanox",
    np.asarray(
        ((0, 0, 0), (115, 33, 0), (180, 119, 0), (255, 255, 255)),
        dtype=float,
    ) / 255.0,
    N=256,
)
nanox_cmap.set_over("red")
nanox_cmap.set_under("blue")


# ---------------------------------------------------------------------------
# Loading and preprocessing
# ---------------------------------------------------------------------------
def _replace_nonfinite(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=float)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        raise ValueError("图像中没有有限数值。")
    if finite.size != data.size:
        data = np.nan_to_num(
            data,
            nan=float(np.median(finite)),
            posinf=float(np.max(finite)),
            neginf=float(np.min(finite)),
        )
    return data


def detrend_image(data: np.ndarray) -> np.ndarray:
    """Remove the best-fit plane without suppressing the atomic Bragg signal."""
    data = _replace_nonfinite(data)
    h, w = data.shape
    yy, xx = np.mgrid[:h, :w]
    design = np.column_stack((xx.ravel(), yy.ravel(), np.ones(data.size)))
    coef, *_ = np.linalg.lstsq(design, data.ravel(), rcond=None)
    plane = coef[0] * xx + coef[1] * yy + coef[2]
    return data - plane


def prepare_fft(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Use v1-style background subtraction plus a gentle anti-wrap edge taper."""
    z = detrend_image(data)
    high_passed = z - gaussian_filter(z, sigma=20.0)
    h, w = high_passed.shape
    edge_taper = np.outer(tukey(h, alpha=0.12), tukey(w, alpha=0.12))
    processed = high_passed * edge_taper
    return processed, np.fft.fftshift(np.fft.fft2(processed))


def _empty_direction_state() -> dict:
    return {
        "peaks_selected": None,
        "peak_widths": None,
        "recommended_sigma": None,
        "reciprocal_condition_number": None,
        "unit_cell_origin_xy": None,
        "unit_cell_basis_xy": None,
        "unit_cell_vertices_xy": None,
        "unit_cell_snap_fraction": None,
        "calibrated": False,
    }


def save_active_direction_state() -> None:
    """Save FFT/cell choices for the currently displayed scan direction."""
    if active_scan_direction not in ("forward", "backward"):
        return
    state = direction_states.setdefault(active_scan_direction, _empty_direction_state())
    state.update(
        {
            "peaks_selected": None if peaks_selected is None else [np.asarray(p).copy() for p in peaks_selected],
            "peak_widths": peak_widths,
            "recommended_sigma": recommended_sigma,
            "reciprocal_condition_number": reciprocal_condition_number,
            "unit_cell_origin_xy": None if unit_cell_origin_xy is None else np.asarray(unit_cell_origin_xy).copy(),
            "unit_cell_basis_xy": None if unit_cell_basis_xy is None else np.asarray(unit_cell_basis_xy).copy(),
            "unit_cell_vertices_xy": None if unit_cell_vertices_xy is None else np.asarray(unit_cell_vertices_xy).copy(),
            "unit_cell_snap_fraction": unit_cell_snap_fraction,
        }
    )


def activate_scan_direction(direction: str, save_current: bool = True) -> None:
    """Switch all legacy single-image globals to one stored scan direction."""
    global active_scan_direction, img_full, img, img_processed, fft_shifted
    global peaks_selected, peak_widths, recommended_sigma, reciprocal_condition_number
    global unit_cell_origin_xy, unit_cell_basis_xy, unit_cell_vertices_xy, unit_cell_snap_fraction
    direction = str(direction).lower()
    if direction not in scan_images_full:
        raise RuntimeError(f"The SXM file has no {direction} image available.")
    if save_current and img_full is not None:
        save_active_direction_state()
    active_scan_direction = direction
    img_full = np.asarray(scan_images_full[direction], dtype=float)
    if analysis_roi_full is None:
        img = img_full.copy()
    else:
        y1, y2, x1, x2 = map(int, analysis_roi_full)
        img = np.asarray(img_full[y1:y2, x1:x2], dtype=float).copy()
    img_processed, fft_shifted = prepare_fft(img)
    state = direction_states.setdefault(direction, _empty_direction_state())
    peaks_selected = state["peaks_selected"]
    peak_widths = state["peak_widths"]
    recommended_sigma = state["recommended_sigma"]
    reciprocal_condition_number = state["reciprocal_condition_number"]
    unit_cell_origin_xy = state["unit_cell_origin_xy"]
    unit_cell_basis_xy = state["unit_cell_basis_xy"]
    unit_cell_vertices_xy = state["unit_cell_vertices_xy"]
    unit_cell_snap_fraction = state["unit_cell_snap_fraction"]


def selected_scan_directions() -> list[str]:
    mode = "forward" if scan_direction_var is None else str(scan_direction_var.get()).lower()
    return ["forward", "backward"] if mode == "both" else [mode]


def align_backward_orientation(forward: np.ndarray, backward: np.ndarray) -> tuple[np.ndarray, bool]:
    """Choose the backward x orientation that best matches the forward topograph."""
    def normalized_feature(data):
        feature = detrend_image(data)
        feature = feature - gaussian_filter(feature, sigma=max(2.0, min(data.shape) / 20.0))
        scale = float(np.std(feature))
        return feature / max(scale, np.finfo(float).eps)

    forward_feature = normalized_feature(forward)
    candidates = (np.asarray(backward, float), np.flip(backward, axis=1))
    scores = [
        float(np.mean(forward_feature * normalized_feature(candidate)))
        for candidate in candidates
    ]
    flipped = scores[1] > scores[0]
    return np.asarray(candidates[int(flipped)], dtype=float).copy(), flipped


def _read_sxm(
    path: str, preferred_channel: str = "Z"
) -> tuple[dict[str, np.ndarray], tuple[float, float], str, dict]:
    sxm = nap.read.Scan(path)
    channels = list(sxm.signals)
    if not channels:
        raise ValueError("The SXM file contains no image channels.")

    def find_channel(name):
        exact = next((key for key in channels if key.casefold() == name.casefold()), None)
        if exact is not None:
            return exact
        return next((key for key in channels if name.casefold() in key.casefold()), None)

    requested = str(preferred_channel or "Z")
    channel = find_channel(requested)
    if channel is None:
        fallback_name = "Z" if requested.casefold() == "current" else "Current"
        channel = find_channel(fallback_name) or channels[0]
    data = np.asarray(sxm.signals[channel]["forward"], dtype=float)
    backward_raw = np.asarray(sxm.signals[channel]["backward"], dtype=float)
    if data.ndim != 2:
        raise ValueError(f"SXM 通道 {channel!r} 不是二维图像：{data.shape}")
    if backward_raw.shape != data.shape:
        raise ValueError(
            f"Forward/backward image shapes differ: {data.shape} versus {backward_raw.shape}."
        )
    backward, backward_flipped = align_backward_orientation(data, backward_raw)
    images = {"forward": data, "backward": backward}

    header_pixels = np.asarray(sxm.header.get("scan_pixels", []), dtype=int)
    if header_pixels.size >= 2:
        expected_shape = (int(header_pixels[1]), int(header_pixels[0]))
        if data.shape != expected_shape:
            raise ValueError(
                f"SXM header says {header_pixels[0]} x {header_pixels[1]} pixels, "
                f"but the Z array is {data.shape[1]} x {data.shape[0]}."
            )

    # Nanonis SCAN_RANGE is (x range, y range) in SI metres.  For FFT
    # calibration the DFT sampling interval is L/N, not the plotting span L/(N-1).
    scan_range = np.asarray(sxm.header.get("scan_range", []), dtype=float)
    if scan_range.size >= 2 and np.all(np.isfinite(scan_range[:2])):
        lx = abs(float(scan_range[0]))
        ly = abs(float(scan_range[1]))
        dx = lx / data.shape[1]
        dy = ly / data.shape[0]
        if dx > 0 and dy > 0:
            metadata = {
                "scan_size_xy": (lx, ly),
                "scan_pixels_xy": (data.shape[1], data.shape[0]),
                "scan_angle_deg": float(sxm.header.get("scan_angle", 0.0)),
                "channel": channel,
                "requested_channel": requested,
                "backward_x_flipped": backward_flipped,
            }
            return images, (dx, dy), "m", metadata
    metadata = {
        "scan_size_xy": (float(data.shape[1]), float(data.shape[0])),
        "scan_pixels_xy": (data.shape[1], data.shape[0]),
        "scan_angle_deg": float(sxm.header.get("scan_angle", 0.0)),
        "channel": channel,
        "requested_channel": requested,
        "backward_x_flipped": backward_flipped,
    }
    return images, (1.0, 1.0), "pixel", metadata


def load_file() -> None:
    global img, img_full, scan_images_full, img_processed, fft_shifted, source_path
    global peaks_selected, peak_widths, recommended_sigma, reciprocal_condition_number
    global reference_roi, pixel_size_xy, scan_size_xy, full_scan_size_xy, spatial_unit, last_results
    global reference_selection_fig, unit_cell_selection_fig, fig_res, result_artists
    global unit_cell_origin_xy, unit_cell_basis_xy, unit_cell_vertices_xy
    global unit_cell_snap_fraction
    global analysis_region_fig, analysis_roi_full, analysis_region_confirmed
    global source_channel, direction_states, active_scan_direction, backward_x_flipped
    global strain_reference_points_xy_nm, strain_reference_selection_fig

    path = filedialog.askopenfilename(
        title="选择 Nanonis SXM 文件",
        filetypes=[("Nanonis SXM", "*.sxm"), ("All files", "*.*")],
    )
    if not path:
        return

    try:
        preferred_channel = (
            analysis_channel_var.get() if analysis_channel_var is not None else "Z"
        )
        images, spacing, unit, metadata = _read_sxm(path, preferred_channel)
    except Exception as exc:
        messagebox.showerror("读取失败", str(exc))
        return

    plt.close("all")
    reference_selection_fig = None
    unit_cell_selection_fig = None
    analysis_region_fig = None
    fig_res = None
    result_artists = {}
    scan_images_full = {
        key: _replace_nonfinite(value) for key, value in images.items()
    }
    direction_states = {key: _empty_direction_state() for key in scan_images_full}
    requested_directions = selected_scan_directions()
    active_scan_direction = requested_directions[0]
    img_full = scan_images_full[active_scan_direction]
    img = img_full.copy()
    img_processed, fft_shifted = prepare_fft(img)
    source_path = Path(path)
    source_channel = str(metadata["channel"])
    backward_x_flipped = bool(metadata.get("backward_x_flipped", False))
    pixel_size_xy = spacing
    scan_size_xy = metadata["scan_size_xy"]
    full_scan_size_xy = metadata["scan_size_xy"]
    spatial_unit = unit
    peaks_selected = None
    peak_widths = None
    recommended_sigma = None
    reciprocal_condition_number = None
    reference_roi = None
    unit_cell_origin_xy = None
    unit_cell_basis_xy = None
    unit_cell_vertices_xy = None
    unit_cell_snap_fraction = None
    analysis_roi_full = None
    analysis_region_confirmed = False
    strain_reference_points_xy_nm = []
    strain_reference_selection_fig = None
    last_results = None
    if auto_mask_var is not None:
        auto_mask_var.set(False)

    if unit == "m":
        lx_nm, ly_nm = np.asarray(scan_size_xy) * 1e9
        if scan_info_var is not None:
            scan_info_var.set(
                f"SXM: {img.shape[1]} x {img.shape[0]} px, "
                f"{lx_nm:.6g} x {ly_nm:.6g} nm; channel: {source_channel}"
            )
    elif scan_info_var is not None:
        scan_info_var.set(
            f"SXM: {img.shape[1]} x {img.shape[0]} px; "
            f"channel: {source_channel}; physical size unavailable"
        )
    if source_channel.casefold() != str(preferred_channel).casefold():
        messagebox.showwarning(
            "Analysis channel",
            f"Requested channel '{preferred_channel}' was not found. Using '{source_channel}'.",
        )

    if not app_closing:
        select_analysis_region()


def _apply_analysis_region(roi) -> None:
    """Crop the full topograph and rebuild every downstream FFT-dependent state."""
    global img, img_processed, fft_shifted, scan_size_xy
    global peaks_selected, peak_widths, recommended_sigma, reciprocal_condition_number
    global reference_roi, last_results, fig_res
    global unit_cell_origin_xy, unit_cell_basis_xy, unit_cell_vertices_xy
    global unit_cell_snap_fraction
    global analysis_roi_full, analysis_region_confirmed, direction_states
    global strain_reference_points_xy_nm
    if img_full is None:
        raise RuntimeError("No full STM image is loaded.")
    y1, y2, x1, x2 = map(int, roi)
    if y2-y1 < 32 or x2-x1 < 32:
        raise ValueError("The analysis region must be at least 32 x 32 pixels.")
    cropped = np.asarray(img_full[y1:y2, x1:x2], dtype=float).copy()
    processed, transformed = prepare_fft(cropped)
    img, img_processed, fft_shifted = cropped, processed, transformed
    dx, dy = map(float, pixel_size_xy)
    scan_size_xy = (cropped.shape[1]*dx, cropped.shape[0]*dy)
    analysis_roi_full = (y1, y2, x1, x2)
    analysis_region_confirmed = True
    direction_states = {key: _empty_direction_state() for key in scan_images_full}
    strain_reference_points_xy_nm = []
    peaks_selected = None
    peak_widths = None
    recommended_sigma = None
    reciprocal_condition_number = None
    reference_roi = None
    last_results = None
    unit_cell_origin_xy = None
    unit_cell_basis_xy = None
    unit_cell_vertices_xy = None
    unit_cell_snap_fraction = None
    if fig_res is not None:
        plt.close(fig_res)
        fig_res = None
    if auto_mask_var is not None:
        auto_mask_var.set(False)
    if scan_info_var is not None:
        if spatial_unit == "m":
            lx_nm, ly_nm = np.asarray(scan_size_xy)*1e9
            scan_info_var.set(
                f"ROI: {cropped.shape[1]} x {cropped.shape[0]} px, "
                f"{lx_nm:.5g} x {ly_nm:.5g} nm; channel: {source_channel}; "
                f"full pixels x={x1}:{x2}, y={y1}:{y2}"
            )
        else:
            scan_info_var.set(
                f"ROI: {cropped.shape[1]} x {cropped.shape[0]} px; "
                f"channel: {source_channel}; full pixels x={x1}:{x2}, y={y1}:{y2}"
            )


def select_analysis_region() -> None:
    """Select a defect-free ROI on the full image before constructing its FFT."""
    global analysis_region_fig
    if img_full is None:
        messagebox.showwarning("Analysis region", "Load an SXM image first.")
        return
    if analysis_region_fig is not None:
        try:
            plt.close(analysis_region_fig)
        except Exception:
            pass
    height, width = img_full.shape
    if spatial_unit == "m":
        lx, ly = np.asarray(full_scan_size_xy, dtype=float)*1e9
        extent = (0.0, lx, ly, 0.0)
        xlabel, ylabel = "x (nm)", "y (nm)"
    else:
        lx, ly = float(width), float(height)
        extent = (0.0, lx, ly, 0.0)
        xlabel, ylabel = "x (pixel)", "y (pixel)"
    state = {"roi": None}
    fig, ax = plt.subplots(figsize=(9, 8), num="Select defect-free analysis region")
    analysis_region_fig = fig
    fig.subplots_adjust(bottom=0.16)
    ax.imshow(img_full, cmap=nanox_cmap, origin="upper", extent=extent)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"{source_channel}: drag a rectangle around the defect-free test region, then confirm"
    )

    def on_select(eclick, erelease):
        if None in (eclick.xdata, eclick.ydata, erelease.xdata, erelease.ydata):
            return
        xa, xb = sorted((float(eclick.xdata), float(erelease.xdata)))
        ya, yb = sorted((float(eclick.ydata), float(erelease.ydata)))
        x1 = int(np.clip(np.floor(xa/lx*width), 0, width-1))
        x2 = int(np.clip(np.ceil(xb/lx*width), x1+1, width))
        y1 = int(np.clip(np.floor(ya/ly*height), 0, height-1))
        y2 = int(np.clip(np.ceil(yb/ly*height), y1+1, height))
        state["roi"] = (y1, y2, x1, x2)
        ax.set_title(f"Selected {x2-x1} x {y2-y1} px - click Confirm region")
        fig.canvas.draw_idle()

    selector = RectangleSelector(
        ax, on_select, useblit=True, button=[1], minspanx=12, minspany=12,
        spancoords="pixels", interactive=True,
        props=dict(facecolor="yellow", edgecolor="yellow", alpha=0.22, fill=True),
    )

    def finish(roi):
        try:
            _apply_analysis_region(roi)
        except Exception as exc:
            messagebox.showerror("Analysis region", str(exc))
            return
        plt.close(fig)
        show_image()

    def confirm(_event):
        if state["roi"] is None:
            messagebox.showwarning("Analysis region", "Draw a rectangular test region first.")
            return
        finish(state["roi"])

    def use_full(_event):
        finish((0, height, 0, width))

    confirm_ax = fig.add_axes([0.56, 0.045, 0.19, 0.065])
    full_ax = fig.add_axes([0.77, 0.045, 0.15, 0.065])
    confirm_button = Button(confirm_ax, "Confirm region")
    full_button = Button(full_ax, "Use full image")
    confirm_button.on_clicked(confirm)
    full_button.on_clicked(use_full)
    fig._analysis_widgets = (selector, confirm_button, full_button)

    def on_close(_event):
        global analysis_region_fig
        if analysis_region_fig is fig:
            analysis_region_fig = None

    fig.canvas.mpl_connect("close_event", on_close)
    plt.show(block=False)


def select_strain_reference_patches() -> None:
    """Select common physical patches assumed to be strain-free."""
    global strain_reference_selection_fig
    if img is None or not analysis_region_confirmed:
        messagebox.showwarning("Strain reference", "Select and confirm the analysis region first.")
        return
    if spatial_unit != "m":
        messagebox.showwarning("Strain reference", "Physical SXM calibration is required.")
        return
    if strain_reference_selection_fig is not None:
        try:
            plt.close(strain_reference_selection_fig)
        except Exception:
            pass
    lx_nm, ly_nm = np.asarray(scan_size_xy, dtype=float) * 1e9
    extent = (0.0, lx_nm, ly_nm, 0.0)
    if selected_scan_directions() == ["forward", "backward"]:
        y1, y2, x1, x2 = map(int, analysis_roi_full)
        shown = 0.5 * (
            scan_images_full["forward"][y1:y2, x1:x2]
            + scan_images_full["backward"][y1:y2, x1:x2]
        )
        image_label = "Forward/backward mean"
    else:
        shown = img
        image_label = active_scan_direction.capitalize()
    try:
        a_nm, b_nm, _angle = read_lattice_inputs()
        suggested_radius = float(np.clip(0.75 * max(a_nm, b_nm), 0.15, 0.15 * min(lx_nm, ly_nm)))
    except Exception:
        suggested_radius = float(np.clip(0.06 * min(lx_nm, ly_nm), 0.15, 1.5))
    state = {
        "points": [np.asarray(point, float).copy() for point in strain_reference_points_xy_nm],
        "radius": float(strain_reference_radius_nm if strain_reference_points_xy_nm else suggested_radius),
        "artists": [],
    }
    fig, ax = plt.subplots(figsize=(9.5, 8), num="Select strain-free reference patches")
    strain_reference_selection_fig = fig
    fig.subplots_adjust(bottom=0.18)
    ax.imshow(shown, cmap=nanox_cmap, origin="upper", extent=extent)
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")

    def redraw():
        for artist in state["artists"]:
            artist.remove()
        state["artists"] = []
        for index, point in enumerate(state["points"], start=1):
            circle = Circle(point, state["radius"], facecolor="lime", edgecolor="cyan",
                            linewidth=1.5, alpha=0.22)
            ax.add_patch(circle)
            label = ax.text(point[0], point[1], str(index), color="white", fontsize=9,
                            ha="center", va="center", fontweight="bold")
            state["artists"].extend((circle, label))
        mode = "drift surface" if reference_correction_var is None else reference_correction_var.get()
        ax.set_title(
            f"{image_label}: left-click to add; right-click to remove nearest\n"
            f"{len(state['points'])} patches, radius={state['radius']:.3g} nm, mode={mode}"
        )
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes is not ax or event.xdata is None or event.ydata is None:
            return
        point = np.array([event.xdata, event.ydata], dtype=float)
        if event.button == 1:
            if (state["radius"] <= point[0] <= lx_nm - state["radius"]
                    and state["radius"] <= point[1] <= ly_nm - state["radius"]):
                state["points"].append(point)
        elif event.button == 3 and state["points"]:
            distances = np.linalg.norm(np.asarray(state["points"]) - point, axis=1)
            state["points"].pop(int(np.argmin(distances)))
        redraw()

    radius_ax = fig.add_axes([0.12, 0.075, 0.38, 0.035])
    radius_slider = Slider(
        radius_ax, "Patch radius (nm)", 0.10, max(0.12, 0.18 * min(lx_nm, ly_nm)),
        valinit=np.clip(state["radius"], 0.10, max(0.12, 0.18 * min(lx_nm, ly_nm))),
    )

    def update_radius(value):
        state["radius"] = float(value)
        redraw()

    def clear(_event):
        state["points"].clear()
        redraw()

    def confirm(_event):
        global strain_reference_points_xy_nm, strain_reference_radius_nm
        mode = "none" if reference_correction_var is None else reference_correction_var.get()
        minimum = 6 if mode == "drift" else 1
        if len(state["points"]) < minimum:
            messagebox.showwarning(
                "Strain reference",
                f"{mode} correction requires at least {minimum} reference patches. "
                "For drift correction, distribute them around the defect and across the image.",
            )
            return
        strain_reference_points_xy_nm = [point.copy() for point in state["points"]]
        strain_reference_radius_nm = float(state["radius"])
        plt.close(fig)
        messagebox.showinfo(
            "Strain reference",
            f"Saved {len(strain_reference_points_xy_nm)} common reference patches "
            f"with radius {strain_reference_radius_nm:.3g} nm.",
        )

    clear_ax = fig.add_axes([0.57, 0.055, 0.15, 0.065])
    confirm_ax = fig.add_axes([0.75, 0.055, 0.18, 0.065])
    clear_button = Button(clear_ax, "Clear")
    confirm_button = Button(confirm_ax, "Confirm patches")
    clear_button.on_clicked(clear)
    confirm_button.on_clicked(confirm)
    radius_slider.on_changed(update_radius)
    fig.canvas.mpl_connect("button_press_event", on_click)
    fig._reference_widgets = (radius_slider, clear_button, confirm_button)

    def on_close(_event):
        global strain_reference_selection_fig
        if strain_reference_selection_fig is fig:
            strain_reference_selection_fig = None

    fig.canvas.mpl_connect("close_event", on_close)
    redraw()
    plt.show(block=False)


def show_image() -> None:
    if app_closing:
        return
    if img is None:
        messagebox.showwarning("尚未载入", "请先载入 SXM 图像。")
        return
    fig, ax = plt.subplots(figsize=(8, 7))
    if spatial_unit == "m":
        lx_nm, ly_nm = np.asarray(scan_size_xy) * 1e9
        extent = (0.0, lx_nm, ly_nm, 0.0)
        im = ax.imshow(img, cmap=nanox_cmap, origin="upper", extent=extent)
        ax.set_xlabel("x (nm)")
        ax.set_ylabel("y (nm)")
        ax.set_title(
            f"STM {source_channel} {active_scan_direction} image "
            f"({lx_nm:.6g} x {ly_nm:.6g} nm)"
        )
    else:
        im = ax.imshow(img, cmap=nanox_cmap, origin="upper")
        ax.set_xlabel("x pixel")
        ax.set_ylabel("y pixel")
        ax.set_title(f"STM {source_channel} {active_scan_direction} image")
    fig.colorbar(im, ax=ax, label="Z signal")
    fig.tight_layout()
    # Tk owns the application event loop; never start a nested blocking loop.
    plt.show(block=False)
    fig.canvas.draw_idle()


# ---------------------------------------------------------------------------
# FFT display and Bragg-peak selection
# ---------------------------------------------------------------------------
def fft_display_array(F: np.ndarray) -> np.ndarray:
    """Log-amplitude FFT display, close to Gwyddion's plain FFT view.

    A per-pixel local-contrast booster was tried here to expose weak Bragg
    peaks, but it self-normalizes pure background noise to order 1 everywhere
    (noise divided by a local estimate of its own scale), lighting up the
    whole frame as speckle instead of leaving it flat like Gwyddion's display.
    """
    amp = np.log1p(np.abs(F))
    h, w = amp.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    radius2 = (xx - cx) ** 2 + (yy - cy) ** 2
    high_pass = 1.0 - np.exp(-radius2 / (2.0 * 10.0**2))
    view = amp * high_pass
    lo, hi = np.percentile(view, (20.0, 99.8))
    view = np.clip((view-lo)/max(hi-lo, 1e-12), 0.0, 1.0)
    return gaussian_filter(view, 0.35)


def _fft_crop(data: np.ndarray) -> tuple[np.ndarray, int, int]:
    h, w = data.shape
    cy, cx = h // 2, w // 2
    half = max(8, int(min(h, w) // (2 * zoom_factor)))
    y1, y2 = max(0, cy - half), min(h, cy + half)
    x1, x2 = max(0, cx - half), min(w, cx + half)
    return data[y1:y2, x1:x2], y1, x1


def show_fft() -> None:
    if app_closing:
        return
    if fft_shifted is None:
        messagebox.showwarning("尚未载入", "请先载入 SXM 图像。")
        return
    display = fft_display_array(fft_shifted)
    crop, y1, x1 = _fft_crop(display)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(crop, cmap=gwyddion_cmap, origin="upper", interpolation="bilinear")
    ax.set_title(f"FFT {active_scan_direction} ({zoom_factor}x zoom)")
    ax.set_xlabel(f"FFT column + {x1}")
    ax.set_ylabel(f"FFT row + {y1}")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    plt.show(block=False)
    fig.canvas.draw_idle()


def set_zoom(value: int) -> None:
    global zoom_factor
    zoom_factor = int(value)
    if fft_shifted is not None:
        show_fft()


def estimate_peak_width(F: np.ndarray, center_rc, radius: int = 5) -> dict:
    """Estimate a Bragg peak centroid and intensity-covariance width."""
    h, w = F.shape
    r, c = map(float, center_rc)
    r0, c0 = int(round(r)), int(round(c))
    q1, q2 = max(0, r0 - radius), min(h, r0 + radius + 1)
    p1, p2 = max(0, c0 - radius), min(w, c0 + radius + 1)
    power = np.abs(F[q1:q2, p1:p2]) ** 2
    if power.size < 9:
        raise ValueError("Bragg peak is too close to the FFT boundary.")
    background = float(np.percentile(power, 35.0))
    weights = np.maximum(power - background, 0.0)
    # Reject a broad noise pedestal without truncating the central peak tails.
    weights[weights < 0.05 * np.max(weights)] = 0.0
    rr, cc = np.mgrid[q1:q2, p1:p2]
    total = float(np.sum(weights))
    if total <= 0:
        raise ValueError("Could not estimate the selected Bragg peak width.")
    refined = np.array([np.sum(rr * weights) / total, np.sum(cc * weights) / total])
    dr = rr - refined[0]
    dc = cc - refined[1]
    covariance = np.array(
        [
            [np.sum(weights * dr * dr), np.sum(weights * dr * dc)],
            [np.sum(weights * dr * dc), np.sum(weights * dc * dc)],
        ]
    ) / total
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.05**2)
    sigma_axes = np.sqrt(eigenvalues)
    return {
        "center_rc": refined,
        "covariance_rc": covariance,
        "sigma_axes_pixel": sigma_axes,
        "fwhm_axes_pixel": 2.35482 * sigma_axes,
        "background_power": background,
    }


def refine_peak(F: np.ndarray, initial_rc, search_radius: int = 5, return_info=False):
    """Snap a manual click to a local maximum, then refine it to sub-pixel precision."""
    h, w = F.shape
    r, c = (int(round(initial_rc[0])), int(round(initial_rc[1])))
    r1, r2 = max(0, r - search_radius), min(h, r + search_radius + 1)
    c1, c2 = max(0, c - search_radius), min(w, c + search_radius + 1)
    local = np.abs(F[r1:r2, c1:c2])
    if local.size == 0:
        raise ValueError("选择点位于 FFT 范围之外。")
    mr, mc = np.unravel_index(np.argmax(local), local.shape)
    info = estimate_peak_width(
        F, (r1 + mr, c1 + mc), radius=max(3, min(5, int(search_radius)))
    )
    return info if return_info else np.asarray(info["center_rc"])


def weak_peak_centroid(F, clicked_rc, radius=5) -> dict:
    """Click-constrained subpixel centroid for a weak peak on a sloping FFT background."""
    amplitude = np.abs(np.asarray(F, dtype=complex))
    log_amp = np.log1p(amplitude)
    feature = gaussian_filter(log_amp, 0.65) - gaussian_filter(log_amp, 2.8)
    h, w = amplitude.shape
    cr, cc = map(float, clicked_rc)
    r0, c0 = int(round(cr)), int(round(cc))
    q = int(radius)
    y1, y2 = max(0, r0-q), min(h, r0+q+1)
    x1, x2 = max(0, c0-q), min(w, c0+q+1)
    patch = feature[y1:y2, x1:x2]
    yy, xx = np.mgrid[y1:y2, x1:x2]
    if patch.size < 9:
        raise ValueError("Weak Bragg peak is too close to the FFT boundary.")
    noise = 1.4826*np.median(np.abs(patch-np.median(patch))) + 1e-12
    significance = float((np.max(patch)-np.median(patch))/noise)
    # A Gaussian click prior prevents a weak peak from snapping to an unrelated
    # nearby maximum while still allowing several pixels of manual error.
    prior = np.exp(-((yy-cr)**2+(xx-cc)**2)/(2.0*2.4**2))
    weights = np.maximum(patch-np.percentile(patch, 45.0), 0.0)*prior
    total = float(np.sum(weights))
    if total <= 0 or significance < 1.35:
        raise ValueError(
            f"No statistically distinct weak FFT peak near the click (SNR={significance:.2f})."
        )
    center = np.array([np.sum(yy*weights)/total, np.sum(xx*weights)/total])
    if np.linalg.norm(center-np.asarray(clicked_rc, float)) > 3.5:
        raise ValueError("Weak-peak centroid moved too far from the manual click.")
    dr, dc = yy-center[0], xx-center[1]
    covariance = np.array([
        [np.sum(weights*dr*dr), np.sum(weights*dr*dc)],
        [np.sum(weights*dr*dc), np.sum(weights*dc*dc)],
    ])/total
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.25**2)
    sigma_axes = np.sqrt(eigenvalues)
    effective_n = total**2/max(float(np.sum(weights**2)), 1e-12)
    uncertainty = sigma_axes/np.sqrt(max(effective_n, 1.0))
    mr, mc = np.unravel_index(np.argmax(patch*prior), patch.shape)
    return {
        "clicked_rc": np.asarray(clicked_rc, dtype=float),
        "integer_maximum_rc": np.asarray((y1+mr, x1+mc), dtype=float),
        "center_rc": center,
        "center_uncertainty_rc": uncertainty,
        "sigma_axes_pixel": sigma_axes,
        "fwhm_axes_pixel": 2.35482*sigma_axes,
        "rho": np.asarray(0.0),
        "r_squared": np.asarray(np.nan),
        "peak_significance": np.asarray(significance),
        "fit_method": np.asarray("weak-centroid"),
        "fit_window": np.asarray((y1, y2, x1, x2), dtype=int),
    }


def fit_bragg_peak_2d(F, initial_rc, search_radius=5, fit_radius=4) -> dict:
    """Constrained sub-pixel elliptical-Gaussian fit around a clicked FFT peak."""
    amplitude = np.abs(np.asarray(F))
    h, w = amplitude.shape
    clicked = np.asarray(initial_rc, dtype=float)
    r = int(round(clicked[0]))
    c = int(round(clicked[1]))
    s = int(search_radius)
    r1, r2 = max(0, r - s), min(h, r + s + 1)
    c1, c2 = max(0, c - s), min(w, c + s + 1)
    log_amplitude = np.log1p(amplitude)
    peak_feature = gaussian_filter(log_amplitude, 0.65) - gaussian_filter(log_amplitude, 2.8)
    local = peak_feature[r1:r2, c1:c2]
    if local.size < 9:
        raise ValueError("Clicked peak is too close to the FFT boundary.")
    mr, mc = np.unravel_index(np.argmax(local), local.shape)
    maximum_rc = np.array([r1 + mr, c1 + mc], dtype=float)

    q = int(fit_radius)
    y1, y2 = max(0, int(maximum_rc[0]) - q), min(h, int(maximum_rc[0]) + q + 1)
    x1, x2 = max(0, int(maximum_rc[1]) - q), min(w, int(maximum_rc[1]) + q + 1)
    patch = amplitude[y1:y2, x1:x2].astype(float)
    yy, xx = np.mgrid[y1:y2, x1:x2]
    xoff = xx - maximum_rc[1]
    yoff = yy - maximum_rc[0]
    low = float(np.percentile(patch, 15.0))
    scale = max(float(np.max(patch) - low), np.finfo(float).eps)
    observed = (patch - low) / scale

    # background, amplitude, x0, y0, sigma_x, sigma_y, correlation
    initial = np.array([0.0, 1.0, 0.0, 0.0, 1.2, 1.2, 0.0])
    lower = np.array([-0.5, 0.05, -2.0, -2.0, 0.35, 0.35, -0.85])
    upper = np.array([0.8, 2.5, 2.0, 2.0, float(q), float(q), 0.85])

    def model(parameters):
        background, height, x0, y0, sx, sy, rho = parameters
        dx0 = xoff - x0
        dy0 = yoff - y0
        exponent = (
            (dx0 / sx) ** 2
            - 2.0 * rho * dx0 * dy0 / (sx * sy)
            + (dy0 / sy) ** 2
        ) / (2.0 * (1.0 - rho**2))
        return background + height * np.exp(-exponent)

    fit = least_squares(
        lambda parameters: (model(parameters) - observed).ravel(),
        initial,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=0.05,
        max_nfev=1500,
    )
    parameters = fit.x
    fitted = model(parameters)
    residual = observed - fitted
    total_variance = float(np.sum((observed - np.mean(observed)) ** 2))
    r_squared = 1.0 - float(np.sum(residual**2)) / max(total_variance, 1e-15)
    center_rc = maximum_rc + np.array([parameters[3], parameters[2]])
    if not fit.success or r_squared < 0.20 or np.linalg.norm(center_rc - maximum_rc) > 2.1:
        return weak_peak_centroid(F, clicked, radius=max(4, int(search_radius)))

    center_uncertainty = np.full(2, np.nan)
    degrees_freedom = observed.size - parameters.size
    if degrees_freedom > 0 and fit.jac.shape[0] >= fit.jac.shape[1]:
        try:
            covariance = np.linalg.inv(fit.jac.T @ fit.jac)
            covariance *= float(np.sum(fit.fun**2) / degrees_freedom)
            center_uncertainty = np.sqrt(
                np.maximum([covariance[3, 3], covariance[2, 2]], 0.0)
            )
        except np.linalg.LinAlgError:
            pass

    sigma_axes = np.asarray([parameters[5], parameters[4]], dtype=float)
    return {
        "clicked_rc": clicked,
        "integer_maximum_rc": maximum_rc,
        "center_rc": center_rc,
        "center_uncertainty_rc": center_uncertainty,
        "sigma_axes_pixel": sigma_axes,
        "fwhm_axes_pixel": 2.35482 * sigma_axes,
        "rho": np.asarray(parameters[6]),
        "r_squared": np.asarray(r_squared),
        "peak_significance": np.asarray(np.nan),
        "fit_method": np.asarray("gaussian-2d"),
        "fit_window": np.asarray((y1, y2, x1, x2), dtype=int),
    }


def reciprocal_vector_from_peak(shape, peak_rc) -> np.ndarray:
    """Return (gx, gy) in radians per image pixel."""
    h, w = shape
    cy, cx = h // 2, w // 2
    r, c = np.asarray(peak_rc, dtype=float)
    return np.array([2.0 * np.pi * (c - cx) / w, 2.0 * np.pi * (r - cy) / h])


def theoretical_direct_basis(a_nm: float, b_nm: float, angle_deg: float) -> np.ndarray:
    """Return the two theoretical direct-lattice vectors as matrix columns."""
    if a_nm <= 0 or b_nm <= 0:
        raise ValueError("Lattice constants a and b must be positive.")
    if not 5.0 < angle_deg < 175.0:
        raise ValueError("Lattice angle must be between 5 and 175 degrees.")
    angle = np.deg2rad(angle_deg)
    a = a_nm * 1e-9
    b = b_nm * 1e-9
    return np.array([[a, b * np.cos(angle)], [0.0, b * np.sin(angle)]], dtype=float)


def measured_reciprocal_vectors_physical(shape, peaks, spacing_xy) -> np.ndarray:
    """Return selected reciprocal rows in rad/m using the SXM scan calibration."""
    h, w = shape
    dx, dy = map(float, spacing_xy)
    if dx <= 0 or dy <= 0:
        raise ValueError("SXM pixel calibration is unavailable.")
    cy, cx = h // 2, w // 2
    rows = []
    for row, col in np.asarray(peaks, dtype=float):
        rows.append(
            [2.0 * np.pi * (col - cx) / (w * dx), 2.0 * np.pi * (row - cy) / (h * dy)]
        )
    return np.asarray(rows, dtype=float)


def reciprocal_pair_measurements(reciprocal_rows) -> dict:
    """Return FFT periods and the sign-independent acute peak angle."""
    G = np.asarray(reciprocal_rows, dtype=float)
    norms = np.linalg.norm(G, axis=1)
    if G.shape != (2, 2) or np.any(norms <= 0):
        raise ValueError("Two finite reciprocal vectors are required.")
    periods = 2.0 * np.pi / norms
    cosine = abs(float(np.dot(G[0], G[1]) / (norms[0] * norms[1])))
    acute_angle = float(np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0))))
    return {"periods": periods, "acute_angle_deg": acute_angle}


def match_theoretical_lattice(measured_G, a_nm: float, b_nm: float, angle_deg: float) -> dict:
    """Match peak order/sign and recover spatial stretch relative to theory."""
    measured_G = np.asarray(measured_G, dtype=float)
    if measured_G.shape != (2, 2) or abs(np.linalg.det(measured_G)) < 1e-20:
        raise ValueError("Selected reciprocal vectors cannot define a 2D lattice.")
    direct_theory = theoretical_direct_basis(a_nm, b_nm, angle_deg)
    reciprocal_theory = 2.0 * np.pi * np.linalg.inv(direct_theory)
    best = None
    for order in permutations((0, 1)):
        for signs in product((-1.0, 1.0), repeat=2):
            candidate_G = measured_G[list(order)] * np.asarray(signs)[:, None]
            if abs(np.linalg.det(candidate_G)) < 1e-20:
                continue
            direct_measured = 2.0 * np.pi * np.linalg.inv(candidate_G)
            deformation = direct_measured @ np.linalg.inv(direct_theory)
            left_vectors, singular_values, right_vectors_t = np.linalg.svd(deformation)
            left_stretch = left_vectors @ np.diag(singular_values) @ left_vectors.T
            # Scanner-coordinate correction.  F = V R maps an ideal lattice to
            # the measured one; V contains x/y scale and shear while R is the
            # physical sample rotation, which must be preserved.
            coordinate_correction = (
                left_vectors @ np.diag(1.0 / singular_values) @ left_vectors.T
            )
            spatial_strain = left_stretch - np.eye(2)
            score = float(np.linalg.norm(spatial_strain, ord="fro"))
            item = {
                "measured_reciprocal": candidate_G,
                "theoretical_reciprocal": reciprocal_theory,
                "measured_direct": direct_measured,
                "theoretical_direct": direct_theory,
                "deformation_gradient": deformation,
                "left_stretch": left_stretch,
                "coordinate_correction": coordinate_correction,
                "spatial_strain": spatial_strain,
                "principal_stretches": singular_values,
                "order": np.asarray(order, dtype=int),
                "signs": np.asarray(signs, dtype=float),
                "match_score": score,
            }
            measured_info = reciprocal_pair_measurements(candidate_G)
            theoretical_info = reciprocal_pair_measurements(reciprocal_theory)
            item["measured_fft_periods"] = measured_info["periods"]
            item["measured_direct_lengths"] = np.linalg.norm(direct_measured, axis=0)
            direct_cosine = float(
                np.dot(direct_measured[:, 0], direct_measured[:, 1])
                / np.prod(item["measured_direct_lengths"])
            )
            item["measured_direct_angle_deg"] = float(
                np.degrees(np.arccos(np.clip(direct_cosine, -1.0, 1.0)))
            )
            item["theoretical_fft_periods"] = theoretical_info["periods"]
            item["measured_peak_acute_angle_deg"] = measured_info["acute_angle_deg"]
            item["theoretical_peak_acute_angle_deg"] = theoretical_info["acute_angle_deg"]
            if best is None or score < best["match_score"]:
                best = item
    if best is None:
        raise ValueError("Could not match the selected peaks to the theoretical lattice.")
    return best


def reciprocal_geometry(shape, peaks) -> dict:
    """Return reciprocal-matrix geometry and conditioning diagnostics."""
    if peaks is None or len(peaks) != 2:
        raise ValueError("必须选择两个 Bragg 峰。")
    g1 = reciprocal_vector_from_peak(shape, peaks[0])
    g2 = reciprocal_vector_from_peak(shape, peaks[1])
    G = np.vstack((g1, g2))
    scale = np.linalg.norm(g1) * np.linalg.norm(g2)
    if scale == 0:
        raise ValueError("A selected Bragg peak is at the FFT origin.")
    sin_angle = abs(float(np.linalg.det(G))) / scale
    normalized = G / np.linalg.norm(G, axis=1)[:, None]
    return {
        "matrix": G,
        "condition_number": float(np.linalg.cond(G)),
        "normalized_condition_number": float(np.linalg.cond(normalized)),
        "sin_angle": sin_angle,
    }


def _validate_peaks(shape, peaks) -> float:
    geometry = reciprocal_geometry(shape, peaks)
    if geometry["sin_angle"] < 0.08 or geometry["normalized_condition_number"] > 25.0:
        raise ValueError("两个峰近乎共线；请从两组不同晶格方向各选一个 Bragg 峰。")
    if geometry["condition_number"] > 50.0:
        raise ValueError(
            f"Reciprocal-vector matrix is ill-conditioned "
            f"(condition number {geometry['condition_number']:.1f})."
        )
    return geometry["condition_number"]


def recommend_mask_sigma(shape, peaks, widths) -> float:
    """Recommend a mask sigma from measured peak width and peak separation."""
    center = np.array([shape[0] // 2, shape[1] // 2], dtype=float)
    offsets = [np.asarray(p, dtype=float) - center for p in peaks]
    separations = [np.linalg.norm(v) for v in offsets]
    separations.extend(
        [np.linalg.norm(offsets[0] - offsets[1]), np.linalg.norm(offsets[0] + offsets[1])]
    )
    # A mask only as wide as the fitted carrier peak over-smooths the real-space
    # field.  Three measured intensity FWHM captures the peak and nearby strain
    # sidebands while the separation cap prevents overlap with DC/other peaks.
    width_based = max(
        3.0, 3.0 * max(float(np.max(w["fwhm_axes_pixel"])) for w in widths)
    )
    isolation_limit = 0.35 * min(separations)
    return float(np.clip(width_based, 2.0, max(2.0, isolation_limit)))


def auto_select_bragg_peaks(F, spacing_xy, a_nm, b_nm, angle_deg) -> dict:
    """Find the two first-order FFT peaks that best match the entered lattice."""
    h, w = F.shape
    dx, dy = map(float, spacing_xy)
    direct = theoretical_direct_basis(a_nm, b_nm, angle_deg)
    theoretical_G = 2.0 * np.pi * np.linalg.inv(direct)
    expected_q = np.linalg.norm(theoretical_G, axis=1)
    expected_angle = reciprocal_pair_measurements(theoretical_G)["acute_angle_deg"]

    cy, cx = h // 2, w // 2
    rr, cc = np.mgrid[:h, :w]
    qx = 2.0 * np.pi * (cc - cx) / (w * dx)
    qy = 2.0 * np.pi * (rr - cy) / (h * dy)
    qmag = np.hypot(qx, qy)
    nyquist_radius = np.hypot(np.pi / dx, np.pi / dy)
    if np.min(expected_q) > nyquist_radius:
        raise ValueError(
            "The entered lattice period is beyond the image Nyquist limit; "
            "the STM pixel density is insufficient."
        )

    amplitude = np.abs(F)
    local_max = amplitude == maximum_filter(amplitude, size=5, mode="nearest")
    # Cast a slightly wider net than the final 5% acceptance so pixel-grid
    # quantization does not exclude the true candidate before scoring.
    tolerance = 0.08
    shell = np.zeros(F.shape, dtype=bool)
    for target in expected_q:
        shell |= np.abs(qmag - target) <= tolerance * target
    shell &= qmag > 0.25 * np.min(expected_q)
    candidates = np.argwhere(local_max & shell)
    if candidates.shape[0] < 2:
        raise ValueError(
            "No pair of FFT peaks was found near the theoretical reciprocal-lattice radius. "
            "Check a, b, angle, surface reconstruction, and image resolution."
        )

    strengths = amplitude[candidates[:, 0], candidates[:, 1]]
    order = np.argsort(strengths)[::-1][:80]
    candidates = candidates[order]
    strengths = strengths[order]
    log_strength = np.log1p(strengths)
    strength_scale = max(float(np.max(log_strength)), 1e-12)
    best = None
    for i in range(len(candidates)):
        g_i = np.array([qx[tuple(candidates[i])], qy[tuple(candidates[i])]])
        for j in range(i + 1, len(candidates)):
            g_j = np.array([qx[tuple(candidates[j])], qy[tuple(candidates[j])]])
            G = np.vstack((g_i, g_j))
            measured = reciprocal_pair_measurements(G)
            angle_error = abs(measured["acute_angle_deg"] - expected_angle)
            if angle_error > 30.0:
                continue
            measured_q = np.linalg.norm(G, axis=1)
            length_score = min(
                np.sum(((measured_q - expected_q) / expected_q) ** 2),
                np.sum(((measured_q[::-1] - expected_q) / expected_q) ** 2),
            )
            if np.sqrt(length_score) > 0.05:
                continue
            amplitude_bonus = 0.05 * (log_strength[i] + log_strength[j]) / strength_scale
            score = float(length_score + (angle_error / 15.0) ** 2 - amplitude_bonus)
            if best is None or score < best["score"]:
                best = {"score": score, "coords": (candidates[i], candidates[j])}
    if best is None:
        raise ValueError(
            "FFT peaks exist near the expected periods, but no non-collinear pair matches "
            "the entered lattice angle within 30 degrees."
        )

    peak_info = [refine_peak(F, coord, search_radius=4, return_info=True) for coord in best["coords"]]
    peaks = [np.asarray(item["center_rc"], dtype=float) for item in peak_info]
    condition = _validate_peaks(F.shape, peaks)
    sigma = recommend_mask_sigma(F.shape, peaks, peak_info)
    measured_G = measured_reciprocal_vectors_physical(F.shape, peaks, spacing_xy)
    measured = reciprocal_pair_measurements(measured_G)
    refined_q = np.linalg.norm(measured_G, axis=1)
    refined_length_error = min(
        np.sqrt(np.sum(((refined_q - expected_q) / expected_q) ** 2)),
        np.sqrt(np.sum(((refined_q[::-1] - expected_q) / expected_q) ** 2)),
    )
    refined_angle_error = abs(measured["acute_angle_deg"] - expected_angle)
    if refined_length_error > 0.05 or refined_angle_error > 30.0:
        periods_nm = measured["periods"] * 1e9
        raise ValueError(
            "Automatic peak refinement left the theoretical search region "
            f"(periods {periods_nm[0]:.4g}, {periods_nm[1]:.4g} nm; "
            f"acute angle {measured['acute_angle_deg']:.1f} deg). "
            "The atomic Bragg peaks are not sufficiently distinct from background or harmonics."
        )
    return {
        "peaks": peaks,
        "peak_info": peak_info,
        "condition_number": condition,
        "recommended_sigma": sigma,
        "measured_fft_periods": measured["periods"],
        "measured_peak_acute_angle_deg": measured["acute_angle_deg"],
        "selection_score": np.asarray(best["score"]),
    }


def pick_g_vectors() -> None:
    global peaks_selected, peak_widths, recommended_sigma, reciprocal_condition_number
    if fft_shifted is None:
        messagebox.showwarning("Analysis region", "Load an SXM image and confirm a test region first.")
        return

    display = fft_display_array(fft_shifted)
    crop, y1, x1 = _fft_crop(display)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(crop, cmap=gwyddion_cmap, origin="upper", interpolation="bilinear")
    ax.set_title(
        f"{active_scan_direction.capitalize()}: click two non-collinear Bragg peaks"
    )
    pts = plt.ginput(2, timeout=0)
    if app_closing:
        plt.close(fig)
        return
    if len(pts) != 2:
        plt.close(fig)
        messagebox.showerror("选择失败", "需要恰好选择两个 Bragg 峰。")
        return

    try:
        clicked = [np.asarray((y + y1, x + x1), dtype=float) for x, y in pts]
        # Use the clicked position as-is (no sub-pixel fit snapping it
        # elsewhere); only the local width around it is measured, to
        # recommend a demodulation mask sigma.
        info = []
        for point in clicked:
            width_info = dict(estimate_peak_width(fft_shifted, point, radius=5))
            width_info["center_rc"] = point.copy()
            info.append(width_info)
        selected = [np.asarray(item["center_rc"], dtype=float) for item in info]
        condition = _validate_peaks(fft_shifted.shape, selected)
        center = np.array([fft_shifted.shape[0] // 2, fft_shifted.shape[1] // 2])
        offsets = [p - center for p in selected]
        separations = [np.linalg.norm(v) for v in offsets]
        separations.extend((np.linalg.norm(offsets[0] - offsets[1]),
                            np.linalg.norm(offsets[0] + offsets[1])))
        sigma_rec = recommend_mask_sigma(fft_shifted.shape, selected, info)
    except Exception as exc:
        messagebox.showerror("选择失败", str(exc))
        return

    peaks_selected = selected
    peak_widths = info
    recommended_sigma = sigma_rec
    reciprocal_condition_number = condition
    text = (
        "Two Bragg peaks set from your clicks (no sub-pixel fit).\n"
        f"Automatic mask sigma: {sigma_rec:.1f}\n"
        f"Peak-pair condition number: {condition:.2f}"
    )
    for index, item in enumerate(info, start=1):
        text += f"\nPeak {index}: row={item['center_rc'][0]:.3f}, col={item['center_rc'][1]:.3f}"
    if spatial_unit == "m":
        measured_G = measured_reciprocal_vectors_physical(
            fft_shifted.shape, selected, pixel_size_xy
        )
        measured = reciprocal_pair_measurements(measured_G)
        periods_nm = measured["periods"] * 1e9
        text += (
            f"\nFFT periods: {periods_nm[0]:.4g} nm, {periods_nm[1]:.4g} nm"
            f"\nAcute peak angle: {measured['acute_angle_deg']:.2f} deg"
        )
    for point in selected:
        ax.plot(point[1] - x1, point[0] - y1, marker="x", color="yellow", ms=10, mew=1.8)
    ax.set_title("Yellow x: your selected Bragg peaks")
    fig.canvas.draw_idle()
    plt.show(block=False)
    messagebox.showinfo("Bragg 峰已选定", text)
    select_unit_cell_frame()


def pick_g_vectors_fitted() -> None:
    """Click two Bragg peaks, then snap each to a sub-pixel elliptical-Gaussian fit."""
    global peaks_selected, peak_widths, recommended_sigma, reciprocal_condition_number
    if fft_shifted is None:
        messagebox.showwarning("Analysis region", "Load an SXM image and confirm a test region first.")
        return

    display = fft_display_array(fft_shifted)
    crop, y1, x1 = _fft_crop(display)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(crop, cmap=gwyddion_cmap, origin="upper", interpolation="bilinear")
    ax.set_title(
        f"{active_scan_direction.capitalize()}: click two non-collinear Bragg peaks"
    )
    pts = plt.ginput(2, timeout=0)
    if app_closing:
        plt.close(fig)
        return
    if len(pts) != 2:
        plt.close(fig)
        messagebox.showerror("选择失败", "需要恰好选择两个 Bragg 峰。")
        return

    try:
        clicked = [np.asarray((y + y1, x + x1), dtype=float) for x, y in pts]
        info = [fit_bragg_peak_2d(fft_shifted, point) for point in clicked]
        selected = [np.asarray(item["center_rc"], dtype=float) for item in info]
        condition = _validate_peaks(fft_shifted.shape, selected)
        center = np.array([fft_shifted.shape[0] // 2, fft_shifted.shape[1] // 2])
        offsets = [p - center for p in selected]
        separations = [np.linalg.norm(v) for v in offsets]
        separations.extend((np.linalg.norm(offsets[0] - offsets[1]),
                            np.linalg.norm(offsets[0] + offsets[1])))
        sigma_rec = recommend_mask_sigma(fft_shifted.shape, selected, info)
    except Exception as exc:
        messagebox.showerror("选择失败", str(exc))
        return

    peaks_selected = selected
    peak_widths = info
    recommended_sigma = sigma_rec
    reciprocal_condition_number = condition
    text = (
        "Two Bragg peaks fitted successfully.\n"
        f"Automatic mask sigma: {sigma_rec:.1f}\n"
        f"Peak-pair condition number: {condition:.2f}"
    )
    for index, item in enumerate(info, start=1):
        uncertainty = np.asarray(item["center_uncertainty_rc"], dtype=float)
        uncertainty_text = (
            f"±{np.nanmax(uncertainty):.3f}px" if np.any(np.isfinite(uncertainty)) else "uncertainty n/a"
        )
        method = str(np.asarray(item.get("fit_method", "gaussian-2d")))
        r2 = float(item["r_squared"])
        quality_text = (
            f"R²={r2:.3f}" if np.isfinite(r2)
            else f"local SNR={float(item.get('peak_significance', np.nan)):.2f}"
        )
        text += (
            f"\nPeak {index}: row={item['center_rc'][0]:.3f}, "
            f"col={item['center_rc'][1]:.3f}, {quality_text}, "
            f"{uncertainty_text}, method={method}"
        )
    if spatial_unit == "m":
        measured_G = measured_reciprocal_vectors_physical(
            fft_shifted.shape, selected, pixel_size_xy
        )
        measured = reciprocal_pair_measurements(measured_G)
        periods_nm = measured["periods"] * 1e9
        text += (
            f"\nFFT periods: {periods_nm[0]:.4g} nm, {periods_nm[1]:.4g} nm"
            f"\nAcute peak angle: {measured['acute_angle_deg']:.2f} deg"
        )
    for point in clicked:
        ax.plot(point[1] - x1, point[0] - y1, marker="x", color="yellow", ms=8, mew=1.5)
    for point in selected:
        ax.plot(point[1] - x1, point[0] - y1, marker="+", color="cyan", ms=12, mew=1.8)
    ax.set_title("Yellow x: click; cyan +: fitted sub-pixel peak")
    fig.canvas.draw_idle()
    plt.show(block=False)
    messagebox.showinfo("Bragg 峰已选定", text)
    select_unit_cell_frame()


def fft_direct_basis_in_image_coordinates() -> np.ndarray:
    """Return FFT-derived direct-lattice basis columns in displayed x/y units."""
    if img is None or peaks_selected is None:
        raise RuntimeError("Load an image and select two Bragg peaks first.")
    g_pixel = np.vstack(
        [reciprocal_vector_from_peak(img.shape, peak) for peak in peaks_selected]
    )
    if abs(np.linalg.det(g_pixel)) < 1e-10:
        raise ValueError("The selected Bragg peaks cannot define a 2D unit cell.")
    basis_pixel = 2.0 * np.pi * np.linalg.inv(g_pixel)
    if spatial_unit == "m":
        dx_nm, dy_nm = np.asarray(pixel_size_xy, dtype=float) * 1e9
        return np.diag([dx_nm, dy_nm]) @ basis_pixel
    return basis_pixel


def select_unit_cell_frame() -> None:
    """Overlay an FFT-derived cell on the STM image and let the user translate it."""
    global unit_cell_selection_fig
    if img is None or peaks_selected is None:
        messagebox.showwarning("Unit cell", "Select two Bragg peaks first.")
        return
    try:
        basis = fft_direct_basis_in_image_coordinates()
    except Exception as exc:
        messagebox.showerror("Unit-cell frame", str(exc))
        return
    if unit_cell_selection_fig is not None:
        try:
            plt.close(unit_cell_selection_fig)
        except Exception:
            pass

    if spatial_unit == "m":
        lx, ly = np.asarray(scan_size_xy, dtype=float) * 1e9
        extent = (0.0, lx, ly, 0.0)
        unit_label = "nm"
    else:
        ly, lx = img.shape
        extent = (0.0, float(lx), float(ly), 0.0)
        unit_label = "pixel"

    initial_origin = np.array([0.5 * lx, 0.5 * ly]) - 0.5 * (
        basis[:, 0] + basis[:, 1]
    )
    state = {"origin": initial_origin.copy(), "initial": initial_origin.copy(),
             "dragging": False, "last": None}
    fig, ax = plt.subplots(figsize=(9, 8), num="Align FFT-derived unit cell")
    unit_cell_selection_fig = fig
    fig.subplots_adjust(bottom=0.16)
    ax.imshow(img, cmap=nanox_cmap, origin="upper", extent=extent)
    ax.set_xlabel(f"x ({unit_label})")
    ax.set_ylabel(f"y ({unit_label})")
    ax.set_title(
        f"{source_channel}: drag the yellow unit-cell frame onto one clear cell, then confirm"
    )
    main_line, = ax.plot([], [], color="yellow", linewidth=2.4, marker="o", markersize=5)
    neighbour_lines = [ax.plot([], [], color="cyan", linewidth=0.8, alpha=0.35)[0]
                       for _ in range(8)]
    corner_texts = [ax.text(0, 0, label, color="yellow", fontsize=9,
                            ha="left", va="bottom")
                    for label in ("O", "a", "a+b", "b")]

    def vertices(origin):
        a_vec, b_vec = basis[:, 0], basis[:, 1]
        return np.asarray((origin, origin + a_vec, origin + a_vec + b_vec, origin + b_vec))

    def redraw():
        cell = vertices(state["origin"])
        closed = np.vstack((cell, cell[0]))
        main_line.set_data(closed[:, 0], closed[:, 1])
        for text_item, point in zip(corner_texts, cell):
            text_item.set_position(point)
        offsets = [i * basis[:, 0] + j * basis[:, 1]
                   for j in (-1, 0, 1) for i in (-1, 0, 1) if (i, j) != (0, 0)]
        for line, offset in zip(neighbour_lines, offsets):
            neighbour = vertices(state["origin"] + offset)
            neighbour = np.vstack((neighbour, neighbour[0]))
            line.set_data(neighbour[:, 0], neighbour[:, 1])
        fig.canvas.draw_idle()

    def point_near_cell(x, y):
        try:
            uv = np.linalg.solve(basis, np.array([x, y]) - state["origin"])
        except np.linalg.LinAlgError:
            return False
        return bool(np.all(uv >= -0.25) and np.all(uv <= 1.25))

    def on_press(event):
        if event.inaxes is not ax or event.button != 1 or event.xdata is None:
            return
        if point_near_cell(float(event.xdata), float(event.ydata)):
            state["dragging"] = True
            state["last"] = np.array([event.xdata, event.ydata], dtype=float)

    def on_motion(event):
        if not state["dragging"] or event.inaxes is not ax or event.xdata is None:
            return
        current = np.array([event.xdata, event.ydata], dtype=float)
        state["origin"] += current - state["last"]
        state["last"] = current
        redraw()

    def on_release(_event):
        state["dragging"] = False
        state["last"] = None

    def confirm(_event):
        global unit_cell_origin_xy, unit_cell_basis_xy, unit_cell_vertices_xy
        global unit_cell_snap_fraction
        if spatial_unit == "m":
            display_per_pixel = np.asarray(pixel_size_xy, dtype=float)*1e9
        else:
            display_per_pixel = np.ones(2, dtype=float)
        origin_px = state["origin"]/display_per_pixel
        basis_px = np.diag(1.0/display_per_pixel)@basis
        spacing_px = float(min(np.linalg.norm(basis_px[:, 0]),
                               np.linalg.norm(basis_px[:, 1])))
        expected_count = int(max(8, img.size/max(abs(np.linalg.det(basis_px)), 1.0)))
        try:
            detected_px, _scores = detect_atomic_spots(img, spacing_px, expected_count)
        except Exception as exc:
            messagebox.showerror("Unit-cell snap", f"Bright-point detection failed: {exc}")
            return
        distances = np.linalg.norm(detected_px-origin_px, axis=1)
        nearest_index = int(np.argmin(distances))
        snap_fraction = float(distances[nearest_index]/max(spacing_px, 1e-12))
        if snap_fraction > 0.10:
            messagebox.showwarning(
                "Unit-cell snap",
                f"The nearest detected bright point is {100*snap_fraction:.2f}% of a lattice "
                "spacing away (limit: 10%). Move the yellow O corner closer and confirm again.",
            )
            return
        snapped_origin = detected_px[nearest_index]*display_per_pixel
        state["origin"] = snapped_origin
        redraw()
        unit_cell_origin_xy = snapped_origin.copy()
        unit_cell_basis_xy = basis.copy()
        unit_cell_vertices_xy = vertices(unit_cell_origin_xy)
        unit_cell_snap_fraction = snap_fraction
        save_active_direction_state()
        direction_states[active_scan_direction]["calibrated"] = True
        messagebox.showinfo(
            "Unit cell saved",
            f"FFT-derived unit cell saved in {unit_label}.\n"
            f"Origin: ({unit_cell_origin_xy[0]:.4g}, {unit_cell_origin_xy[1]:.4g})\n"
            f"Auto-snap correction: {100*snap_fraction:.2f}% of lattice spacing.\n"
            "The frame position is now available for lattice-site fitting.",
        )
        plt.close(fig)
        requested = selected_scan_directions()
        if (
            requested == ["forward", "backward"]
            and active_scan_direction == "forward"
            and not direction_states["backward"]["calibrated"]
        ):
            activate_scan_direction("backward")
            show_image()
            messagebox.showinfo(
                "Backward calibration",
                "Forward FFT and unit cell are saved. Now pick two Bragg peaks for the "
                "backward image and confirm its FFT-derived unit cell. The same analysis "
                "region is being used.",
            )

    def reset(_event):
        state["origin"] = state["initial"].copy()
        redraw()

    confirm_ax = fig.add_axes([0.56, 0.045, 0.18, 0.065])
    reset_ax = fig.add_axes([0.76, 0.045, 0.16, 0.065])
    confirm_button = Button(confirm_ax, "Confirm cell")
    reset_button = Button(reset_ax, "Reset position")
    confirm_button.on_clicked(confirm)
    reset_button.on_clicked(reset)
    fig._unit_cell_buttons = (confirm_button, reset_button)
    fig._unit_cell_state = state
    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)

    def on_close(_event):
        global unit_cell_selection_fig
        if unit_cell_selection_fig is fig:
            unit_cell_selection_fig = None

    fig.canvas.mpl_connect("close_event", on_close)
    redraw()
    plt.show(block=False)


# ---------------------------------------------------------------------------
# Reference selection and GPA core
# ---------------------------------------------------------------------------
def select_reference_region() -> None:
    global reference_roi, reference_selection_fig
    if img is None:
        messagebox.showwarning("尚未载入", "请先载入 SXM 图像。")
        return

    # Tk already owns the GUI event loop, so this window must be non-blocking.
    # Close an older selector window before opening a new one.
    if reference_selection_fig is not None:
        try:
            plt.close(reference_selection_fig)
        except Exception:
            pass

    state = {"roi": None, "start": None, "current": None, "patch": None}
    fig, ax = plt.subplots(figsize=(9, 7))
    reference_selection_fig = fig
    ax.imshow(img, cmap=nanox_cmap, origin="upper")
    ax.set_title("按住鼠标左键拖动；松开即保存，随后可按 Enter 或关闭窗口")

    def clamp_point(x, y):
        x = float(np.clip(x, 0, img.shape[1] - 1))
        y = float(np.clip(y, 0, img.shape[0] - 1))
        return x, y

    def update_rectangle(x, y):
        if state["start"] is None:
            return
        x0, y0 = state["start"]
        x, y = clamp_point(x, y)
        state["current"] = (x, y)
        x1, x2 = sorted((x0, x))
        y1, y2 = sorted((y0, y))
        patch = state["patch"]
        patch.set_xy((x1, y1))
        patch.set_width(x2 - x1)
        patch.set_height(y2 - y1)
        fig.canvas.draw_idle()

    def on_press(event):
        if event.inaxes is not ax or event.button != 1:
            return
        if event.xdata is None or event.ydata is None:
            return
        start = clamp_point(event.xdata, event.ydata)
        state["start"] = start
        state["current"] = start
        state["roi"] = None
        if state["patch"] is not None:
            state["patch"].remove()
        state["patch"] = Rectangle(
            start, 0, 0, fill=True, facecolor="lime", alpha=0.18,
            edgecolor="lime", linewidth=1.8
        )
        ax.add_patch(state["patch"])
        fig.canvas.draw_idle()

    def on_motion(event):
        if state["start"] is None or event.xdata is None or event.ydata is None:
            return
        update_rectangle(event.xdata, event.ydata)

    def on_release(event):
        global reference_roi
        # Some Matplotlib backends report button=None on mouse release.
        if event.button not in (None, 1) or state["start"] is None:
            return
        if event.xdata is not None and event.ydata is not None:
            update_rectangle(event.xdata, event.ydata)
        x0, y0 = state["start"]
        x, y = state["current"]
        state["start"] = None
        x1, x2 = sorted((int(round(x0)), int(round(x))))
        y1, y2 = sorted((int(round(y0)), int(round(y))))
        x1, x2 = max(0, x1), min(img.shape[1], x2)
        y1, y2 = max(0, y1), min(img.shape[0], y2)
        if x2 - x1 >= 8 and y2 - y1 >= 8:
            state["roi"] = (y1, y2, x1, x2)
            # Save immediately: closing this window must not discard the ROI.
            reference_roi = state["roi"]
            ax.set_title(
                f"已保存参考区：x={x1}:{x2}, y={y1}:{y2}；可关闭或重新拖动"
            )
        else:
            state["roi"] = None
            ax.set_title("区域太小（至少 8×8 像素），请重新拖动")
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == "enter" and state["roi"] is not None:
            plt.close(fig)
        elif event.key == "escape":
            plt.close(fig)

    def on_close(_event):
        global reference_selection_fig
        if reference_selection_fig is fig:
            reference_selection_fig = None

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("close_event", on_close)
    plt.show(block=False)


def gaussian_mask(shape, center_rc, sigma: float) -> np.ndarray:
    rr, cc = np.mgrid[: shape[0], : shape[1]]
    r0, c0 = np.asarray(center_rc, dtype=float)
    return np.exp(-((rr - r0) ** 2 + (cc - c0) ** 2) / (2.0 * sigma**2))


def bragg_complex_field(F: np.ndarray, peak_rc, sigma: float):
    """Return a fractionally demodulated complex field and its FFT mask."""
    if sigma <= 0:
        raise ValueError("FFT 掩膜 sigma 必须大于零。")
    mask = gaussian_mask(F.shape, peak_rc, sigma)
    complex_carrier = np.fft.ifft2(np.fft.ifftshift(F * mask))

    g = reciprocal_vector_from_peak(F.shape, peak_rc)
    yy, xx = np.mgrid[: F.shape[0], : F.shape[1]]
    demodulated = complex_carrier * np.exp(-1j * (g[0] * xx + g[1] * yy))
    return demodulated, mask


def local_phase_from_peak(F: np.ndarray, peak_rc, sigma: float):
    """Backward-compatible unmasked phase extraction helper."""
    field, _mask = bragg_complex_field(F, peak_rc, sigma)
    phase = unwrap_phase(np.angle(field))
    return phase, np.abs(field)


def build_confidence_map(
    amplitude1, amplitude2, sigma: float, threshold=CONFIDENCE_THRESHOLD, edge_factor=1.0
):
    """Build a conservative confidence score from both Bragg-field amplitudes."""
    a1 = np.asarray(amplitude1, dtype=float)
    a2 = np.asarray(amplitude2, dtype=float)
    h, w = a1.shape
    margin = max(
        2,
        int(np.ceil(float(edge_factor) * max(h, w) / (np.pi * max(sigma, 1.0)))),
    )
    margin = min(margin, max(2, min(h, w) // 4))
    edge_valid = np.ones((h, w), dtype=bool)
    edge_valid[:margin] = False
    edge_valid[-margin:] = False
    edge_valid[:, :margin] = False
    edge_valid[:, -margin:] = False

    calibration = edge_valid.copy()
    if np.count_nonzero(calibration) < 16:
        calibration[:] = True
    scale1 = max(float(np.percentile(a1[calibration], 90.0)), 1e-15)
    scale2 = max(float(np.percentile(a2[calibration], 90.0)), 1e-15)
    score1 = np.clip(a1 / scale1, 0.0, 1.0)
    score2 = np.clip(a2 / scale2, 0.0, 1.0)
    confidence = np.sqrt(score1 * score2)
    confidence_mask = (confidence >= float(threshold)) & edge_valid
    return {
        "confidence": confidence,
        "confidence_component1": score1,
        "confidence_component2": score2,
        "confidence_mask": confidence_mask,
        "edge_valid_mask": edge_valid,
        "amplitude_scale1": scale1,
        "amplitude_scale2": scale2,
        "edge_margin_pixel": margin,
        "threshold": float(threshold),
    }


def unwrap_phase_masked(wrapped_phase, valid_mask) -> np.ndarray:
    """Unwrap only connected, confidence-qualified pixels."""
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(wrapped_phase)
    if np.count_nonzero(valid) < 16:
        raise ValueError("Too few confident pixels remain for phase unwrapping.")
    masked = np.ma.array(np.asarray(wrapped_phase, dtype=float), mask=~valid)
    unwrapped = unwrap_phase(masked)
    data = unwrapped.filled(np.nan) if np.ma.isMaskedArray(unwrapped) else np.asarray(unwrapped)
    return np.where(valid, data, np.nan)


def detect_phase_discontinuities(phase, valid_mask, dilation=PHASE_JUMP_DILATION):
    """Detect robustly abnormal nearest-neighbour unwrapped-phase gradients."""
    phase = np.asarray(phase, dtype=float)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(phase)
    jump = np.zeros(phase.shape, dtype=bool)
    thresholds = []

    dx = phase[:, 1:] - phase[:, :-1]
    dx_valid = valid[:, 1:] & valid[:, :-1] & np.isfinite(dx)
    dy = phase[1:, :] - phase[:-1, :]
    dy_valid = valid[1:, :] & valid[:-1, :] & np.isfinite(dy)

    for differences, pairs, axis in ((dx, dx_valid, "x"), (dy, dy_valid, "y")):
        values = differences[pairs]
        if values.size < 16:
            threshold = np.pi
            center = 0.0
        else:
            center = float(np.median(values))
            mad = float(np.median(np.abs(values - center)))
            robust_sigma = max(1.4826 * mad, 1e-6)
            threshold = max(0.8 * np.pi, 8.0 * robust_sigma)
        bad = pairs & (np.abs(differences - center) > threshold)
        if axis == "x":
            jump[:, 1:] |= bad
            jump[:, :-1] |= bad
        else:
            jump[1:, :] |= bad
            jump[:-1, :] |= bad
        thresholds.append(threshold)

    if dilation > 0 and np.any(jump):
        jump = binary_dilation(jump, iterations=int(dilation))
    jump &= valid
    return jump, np.asarray(thresholds, dtype=float)


def fit_phase_plane_in_roi(
    phase, roi, amplitude=None, valid_mask=None, return_metrics=False
):
    """Subtract a weighted phase plane fitted only inside the reference ROI."""
    y1, y2, x1, x2 = roi
    if x2 <= x1 or y2 <= y1:
        raise ValueError("参考区为空。")
    yy, xx = np.mgrid[: phase.shape[0], : phase.shape[1]]
    design = np.column_stack(
        (xx[y1:y2, x1:x2].ravel(), yy[y1:y2, x1:x2].ravel(),
         np.ones((y2 - y1) * (x2 - x1)))
    )
    roi_valid = np.isfinite(phase[y1:y2, x1:x2])
    if valid_mask is not None:
        roi_valid &= np.asarray(valid_mask[y1:y2, x1:x2], dtype=bool)
    values = phase[y1:y2, x1:x2].ravel()
    keep = roi_valid.ravel()
    design = design[keep]
    values = values[keep]
    if values.size < 12:
        raise ValueError("Reference ROI has too few valid phase pixels.")
    if amplitude is not None:
        weights = np.asarray(amplitude[y1:y2, x1:x2], dtype=float).ravel()[keep]
        floor = np.percentile(weights, 10.0)
        weights = np.sqrt(np.maximum(weights, floor) / max(np.max(weights), 1e-15))
        coef, *_ = np.linalg.lstsq(design * weights[:, None], values * weights, rcond=None)
    else:
        coef, *_ = np.linalg.lstsq(design, values, rcond=None)
        weights = np.ones_like(values)
    plane = coef[0] * xx + coef[1] * yy + coef[2]
    corrected = np.where(np.isfinite(phase), phase - plane, np.nan)
    residual = values - design @ coef
    residual_rms = float(
        np.sqrt(np.sum((weights * residual) ** 2) / max(np.sum(weights**2), 1e-15))
    )
    metrics = {
        "valid_fraction": float(np.mean(roi_valid)),
        "residual_rms_rad": residual_rms,
        "n_valid": int(values.size),
    }
    if return_metrics:
        return corrected, coef, metrics
    return corrected, coef


def displacement_from_phases(phi1, phi2, g1, g2):
    """Solve phi_i = -g_i dot u for displacement (ux, uy), in pixels."""
    G = np.vstack((g1, g2)).astype(float)
    det = float(np.linalg.det(G))
    if abs(det) < 1e-10:
        raise ValueError("所选 Bragg 峰近乎共线，无法反演二维位移。")
    phases = np.stack((phi1, phi2), axis=0).reshape(2, -1)
    displacement = -np.linalg.solve(G, phases)
    ux = displacement[0].reshape(phi1.shape)
    uy = displacement[1].reshape(phi1.shape)
    return ux, uy


def _finite_median(values):
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else np.nan


def nan_median_filter(data, size: int, valid_mask) -> np.ndarray:
    """Median-filter finite valid samples without filling masked pixels."""
    data = np.where(valid_mask, np.asarray(data, dtype=float), np.nan)
    if int(size) < 3:
        return data
    size = int(size)
    if size % 2 == 0:
        size += 1
    filtered = generic_filter(data, _finite_median, size=size, mode="constant", cval=np.nan)
    return np.where(valid_mask, filtered, np.nan)


def nan_gaussian_filter(data, sigma: float, valid_mask) -> np.ndarray:
    """Normalized Gaussian convolution that ignores invalid samples."""
    data = np.asarray(data, dtype=float)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(data)
    if sigma <= 0:
        return np.where(valid, data, np.nan)
    numerator = gaussian_filter(np.where(valid, data, 0.0), sigma, mode="nearest")
    denominator = gaussian_filter(valid.astype(float), sigma, mode="nearest")
    filtered = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator > 1e-6,
    )
    return np.where(valid_mask, filtered, np.nan)


def filter_displacement_masked(
    ux_px, uy_px, valid_mask, median_size: int, smooth_sigma: float, spacing_xy=(1.0, 1.0)
):
    """Convert displacement to physical units, then median/Gaussian filter it."""
    dx, dy = map(float, spacing_xy)
    if dx <= 0 or dy <= 0:
        raise ValueError("像素尺寸必须大于零。")
    valid = np.asarray(valid_mask, dtype=bool)
    ux_raw = np.where(valid, np.asarray(ux_px, dtype=float) * dx, np.nan)
    uy_raw = np.where(valid, np.asarray(uy_px, dtype=float) * dy, np.nan)
    ux_median = nan_median_filter(ux_raw, median_size, valid)
    uy_median = nan_median_filter(uy_raw, median_size, valid)
    ux_filtered = nan_gaussian_filter(ux_median, smooth_sigma, valid)
    uy_filtered = nan_gaussian_filter(uy_median, smooth_sigma, valid)
    return {
        "ux_raw": ux_raw,
        "uy_raw": uy_raw,
        "ux_median": ux_median,
        "uy_median": uy_median,
        "ux": ux_filtered,
        "uy": uy_filtered,
    }


def masked_local_derivatives(field, valid_mask, spacing_xy=(1.0, 1.0), radius=2):
    """Estimate local derivatives by a masked weighted plane fit at each pixel."""
    z = np.asarray(field, dtype=float)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(z)
    dx, dy = map(float, spacing_xy)
    radius = max(1, int(radius))
    ky, kx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    # Fit in pixel coordinates to avoid ill-scaled normal matrices when dx/dy
    # are around 1e-10 m, then convert the slopes to physical derivatives.
    xoff = kx.astype(float)
    yoff = ky.astype(float)
    sigma = max(radius / 1.5, 0.75)
    weight = np.exp(-(kx**2 + ky**2) / (2.0 * sigma**2))
    m = valid.astype(float)
    zm = np.where(valid, z, 0.0)

    def corr(values, kernel):
        return correlate(values, kernel, mode="constant", cval=0.0)

    s0 = corr(m, weight)
    sample_count = corr(m, np.ones_like(weight))
    sx = corr(m, weight * xoff)
    sy = corr(m, weight * yoff)
    sxx = corr(m, weight * xoff * xoff)
    syy = corr(m, weight * yoff * yoff)
    sxy = corr(m, weight * xoff * yoff)
    sz = corr(zm, weight)
    sxz = corr(zm, weight * xoff)
    syz = corr(zm, weight * yoff)

    safe_s0 = np.where(s0 > 0, s0, 1.0)
    cxx = sxx - sx * sx / safe_s0
    cyy = syy - sy * sy / safe_s0
    cxy = sxy - sx * sy / safe_s0
    cxz = sxz - sx * sz / safe_s0
    cyz = syz - sy * sz / safe_s0
    determinant = cxx * cyy - cxy**2
    determinant_scale = np.abs(cxx * cyy) + cxy**2
    min_points = max(6, 2 * radius + 3)
    fit_valid = (
        valid
        & (sample_count >= min_points)
        & (determinant > 1e-10 * np.maximum(determinant_scale, np.finfo(float).tiny))
    )
    dz_dx_pixel = np.divide(
        cxz * cyy - cyz * cxy,
        determinant,
        out=np.full_like(z, np.nan),
        where=fit_valid,
    )
    dz_dy_pixel = np.divide(
        cyz * cxx - cxz * cxy,
        determinant,
        out=np.full_like(z, np.nan),
        where=fit_valid,
    )
    return dz_dy_pixel / dy, dz_dx_pixel / dx, fit_valid


def strain_components(dux_dy, dux_dx, duy_dy, duy_dx) -> dict:
    """Build small-strain invariants from displacement derivatives."""
    exx = dux_dx
    eyy = duy_dy
    exy = 0.5 * (dux_dy + duy_dx)
    rotation = 0.5 * (duy_dx - dux_dy)

    mean_normal = 0.5 * (exx + eyy)
    mohr_radius = np.sqrt((0.5 * (exx - eyy)) ** 2 + exy**2)
    e1 = mean_normal + mohr_radius
    e2 = mean_normal - mohr_radius
    theta = 0.5 * np.arctan2(2.0 * exy, exx - eyy)

    # max(|principal strain|) is an intuitive scalar "strain size" in percent.
    magnitude = np.maximum(np.abs(e1), np.abs(e2))
    deviatoric = np.abs(e1 - e2)  # this was called "magnitude" in the old code
    tensor_norm = np.sqrt(exx**2 + eyy**2 + 2.0 * exy**2)

    return {
        "exx": exx,
        "eyy": eyy,
        "exy": exy,
        "rotation": rotation,
        "dilatation": exx + eyy,
        "mean_normal": mean_normal,
        "principal_max": e1,
        "principal_min": e2,
        "principal_angle": theta,
        "magnitude": magnitude,
        "deviatoric": deviatoric,
        "tensor_norm": tensor_norm,
    }


def correct_displacement_gradient(
    dux_dy, dux_dx, duy_dy, duy_dx, coordinate_correction
):
    """Transform H=du/dx into scanner-corrected coordinates: Hc=C H C^-1."""
    if coordinate_correction is None:
        return dux_dy, dux_dx, duy_dy, duy_dx
    correction = np.asarray(coordinate_correction, dtype=float)
    if correction.shape != (2, 2) or not np.all(np.isfinite(correction)):
        raise ValueError("Invalid 2D scanner-distortion correction matrix.")
    if np.linalg.cond(correction) > 1e4:
        raise ValueError("Scanner-distortion correction is numerically ill-conditioned.")
    inverse = np.linalg.inv(correction)
    gradient = np.stack(
        (
            np.stack((dux_dx, dux_dy), axis=0),
            np.stack((duy_dx, duy_dy), axis=0),
        ),
        axis=0,
    )
    corrected = np.einsum("ia,ab...,bj->ij...", correction, gradient, inverse)
    return corrected[0, 1], corrected[0, 0], corrected[1, 1], corrected[1, 0]


def calculate_strain_validated(
    ux_px,
    uy_px,
    phase_valid_mask,
    smooth_sigma: float,
    median_size: int = 0,
    spacing_xy=(1.0, 1.0),
    erosion_iterations=DERIVATIVE_MASK_EROSION,
    derivative_radius=DERIVATIVE_RADIUS,
    outlier_limit=MAX_ABS_STRAIN,
    coordinate_correction=None,
):
    """Filter displacement, erode validity, differentiate locally, and flag outliers."""
    phase_valid = np.asarray(phase_valid_mask, dtype=bool)
    filtered = filter_displacement_masked(
        ux_px, uy_px, phase_valid, median_size, smooth_sigma, spacing_xy
    )
    if int(erosion_iterations) > 0:
        effective_erosion = max(
            int(erosion_iterations),
            int(median_size) // 2,
            int(np.ceil(3.0 * max(float(smooth_sigma), 0.0))),
            int(derivative_radius),
        )
        derivative_mask = binary_erosion(
            phase_valid, iterations=effective_erosion, border_value=0
        )
    else:
        effective_erosion = 0
        derivative_mask = phase_valid.copy()
    dux_dy, dux_dx, fit_x = masked_local_derivatives(
        filtered["ux"], derivative_mask, spacing_xy, derivative_radius
    )
    duy_dy, duy_dx, fit_y = masked_local_derivatives(
        filtered["uy"], derivative_mask, spacing_xy, derivative_radius
    )
    dux_dy, dux_dx, duy_dy, duy_dx = correct_displacement_gradient(
        dux_dy, dux_dx, duy_dy, duy_dx, coordinate_correction
    )
    derivative_mask &= fit_x & fit_y
    raw = strain_components(dux_dy, dux_dx, duy_dy, duy_dx)
    for key in tuple(raw):
        raw[key] = np.where(derivative_mask, raw[key], np.nan)

    outlier_mask = derivative_mask & (
        (np.abs(raw["exx"]) > outlier_limit)
        | (np.abs(raw["eyy"]) > outlier_limit)
        | (np.abs(raw["exy"]) > outlier_limit)
        | (raw["magnitude"] > outlier_limit)
    )
    validated_mask = derivative_mask & ~outlier_mask
    results = dict(filtered)
    if coordinate_correction is not None:
        correction = np.asarray(coordinate_correction, dtype=float)
        ux_corrected = correction[0, 0] * filtered["ux"] + correction[0, 1] * filtered["uy"]
        uy_corrected = correction[1, 0] * filtered["ux"] + correction[1, 1] * filtered["uy"]
        results["ux"], results["uy"] = ux_corrected, uy_corrected
    results["derivative_erosion_iterations"] = np.asarray(effective_erosion)
    results["derivative_valid_mask"] = derivative_mask
    results["outlier_mask"] = outlier_mask
    results["valid_mask"] = validated_mask
    for key, value in raw.items():
        results[f"raw_{key}"] = value
        results[key] = np.where(validated_mask, value, np.nan)
    return results


def calculate_strain(ux_px, uy_px, smooth_sigma: float, spacing_xy=(1.0, 1.0)):
    """Compatibility wrapper using the masked local derivative implementation."""
    valid = np.isfinite(ux_px) & np.isfinite(uy_px)
    return calculate_strain_validated(
        ux_px,
        uy_px,
        valid,
        smooth_sigma,
        median_size=0,
        spacing_xy=spacing_xy,
        erosion_iterations=0,
        outlier_limit=np.inf,
    )


def strain_invariants_from_components(exx, eyy, exy) -> dict:
    mean_normal = 0.5 * (exx + eyy)
    radius = np.sqrt((0.5 * (exx - eyy)) ** 2 + exy**2)
    e1 = mean_normal + radius
    e2 = mean_normal - radius
    return {
        "dilatation": exx + eyy,
        "mean_normal": mean_normal,
        "principal_max": e1,
        "principal_min": e2,
        "principal_angle": 0.5 * np.arctan2(2.0 * exy, exx - eyy),
        "magnitude": np.maximum(np.abs(e1), np.abs(e2)),
        "deviatoric": np.abs(e1 - e2),
        "tensor_norm": np.sqrt(exx**2 + eyy**2 + 2.0 * exy**2),
    }


def strain_reference_mask(shape) -> np.ndarray:
    """Return the union of the user-selected physical reference patches."""
    height, width = map(int, shape)
    dx_nm, dy_nm = np.asarray(pixel_size_xy, dtype=float) * 1e9
    yy, xx = np.mgrid[:height, :width]
    xx = xx * dx_nm
    yy = yy * dy_nm
    mask = np.zeros((height, width), dtype=bool)
    radius2 = float(strain_reference_radius_nm) ** 2
    for point in strain_reference_points_xy_nm:
        px, py = np.asarray(point, dtype=float)
        mask |= (xx - px) ** 2 + (yy - py) ** 2 <= radius2
    return mask


def _robust_quadratic_displacement_fit(ux, uy, sample_mask):
    """Fit a compatible low-frequency displacement surface on reference patches."""
    ux = np.asarray(ux, dtype=float)
    uy = np.asarray(uy, dtype=float)
    mask = np.asarray(sample_mask, dtype=bool) & np.isfinite(ux) & np.isfinite(uy)
    if np.count_nonzero(mask) < 30:
        raise RuntimeError("Too few valid displacement pixels lie inside the reference patches.")
    height, width = ux.shape
    dx, dy = map(float, pixel_size_xy)
    yy, xx = np.mgrid[:height, :width]
    x_mid = 0.5 * (width - 1) * dx
    y_mid = 0.5 * (height - 1) * dy
    x_scale = max(0.5 * width * dx, np.finfo(float).eps)
    y_scale = max(0.5 * height * dy, np.finfo(float).eps)
    xn = (xx * dx - x_mid) / x_scale
    yn = (yy * dy - y_mid) / y_scale
    design_full = np.stack(
        (np.ones_like(xn), xn, yn, xn**2, xn * yn, yn**2), axis=-1
    )
    design = design_full[mask]
    values = np.column_stack((ux[mask], uy[mask]))
    keep = np.ones(len(values), dtype=bool)
    coefficients = None
    for _ in range(5):
        if np.count_nonzero(keep) < 30:
            break
        coefficients, _residuals, rank, _singular = np.linalg.lstsq(
            design[keep], values[keep], rcond=None
        )
        if rank < 6:
            raise RuntimeError(
                "Reference patches do not span the image sufficiently for a drift surface. "
                "Choose patches on the left, right, top, and bottom of the defect."
            )
        residual = np.linalg.norm(design @ coefficients - values, axis=1)
        median = float(np.median(residual))
        mad = 1.4826 * float(np.median(np.abs(residual - median))) + 1e-15
        new_keep = residual <= median + 3.5 * mad
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep
    if coefficients is None or np.linalg.cond(design[keep]) > 1e4:
        raise RuntimeError(
            "Reference-patch geometry is ill-conditioned. Distribute at least six patches "
            "around the image instead of placing them along one line."
        )
    background = np.einsum("...k,kj->...j", design_full, coefficients)
    cux, cuy = coefficients[:, 0], coefficients[:, 1]
    dux_dx = (cux[1] + 2.0 * cux[3] * xn + cux[4] * yn) / x_scale
    dux_dy = (cux[2] + cux[4] * xn + 2.0 * cux[5] * yn) / y_scale
    duy_dx = (cuy[1] + 2.0 * cuy[3] * xn + cuy[4] * yn) / x_scale
    duy_dy = (cuy[2] + cuy[4] * xn + 2.0 * cuy[5] * yn) / y_scale
    return background[..., 0], background[..., 1], dux_dx, dux_dy, duy_dx, duy_dy, coefficients


def _fit_compatible_reference_strain_background(exx, eyy, exy, valid_mask):
    """Fit equal-weight strain planes to patch medians and integrate them to displacement."""
    exx, eyy, exy = (np.asarray(value, dtype=float) for value in (exx, eyy, exy))
    valid = np.asarray(valid_mask, dtype=bool)
    height, width = valid.shape
    dx, dy = map(float, pixel_size_xy)
    dx_nm, dy_nm = dx * 1e9, dy * 1e9
    yy, xx = np.mgrid[:height, :width]
    xx_nm, yy_nm = xx * dx_nm, yy * dy_nm
    x_mid = 0.5 * (width - 1) * dx
    y_mid = 0.5 * (height - 1) * dy
    x_scale = max(0.5 * width * dx, np.finfo(float).eps)
    y_scale = max(0.5 * height * dy, np.finfo(float).eps)
    patch_positions, patch_values, patch_counts, patch_indices = [], [], [], []
    radius2 = float(strain_reference_radius_nm) ** 2
    for patch_index, point in enumerate(strain_reference_points_xy_nm):
        px, py = np.asarray(point, dtype=float)
        patch = ((xx_nm - px) ** 2 + (yy_nm - py) ** 2 <= radius2) & valid
        patch &= np.isfinite(exx) & np.isfinite(eyy) & np.isfinite(exy)
        count = int(np.count_nonzero(patch))
        if count < 5:
            continue
        # Use the actual valid-pixel centroid rather than the clicked centre;
        # clipped/irregular masks otherwise bias an exactly linear background.
        patch_x = float(np.median(xx_nm[patch])) * 1e-9
        patch_y = float(np.median(yy_nm[patch])) * 1e-9
        patch_positions.append(((patch_x - x_mid) / x_scale,
                                (patch_y - y_mid) / y_scale))
        patch_values.append(tuple(np.nanmedian(value[patch]) for value in (exx, eyy, exy)))
        patch_counts.append(count)
        patch_indices.append(patch_index)
    if len(patch_values) < 6:
        raise RuntimeError(
            f"Only {len(patch_values)} reference patches contain enough valid strain data; "
            "at least 6 distributed patches are required for compatible background fitting."
        )
    patch_positions = np.asarray(patch_positions, dtype=float)
    patch_values = np.asarray(patch_values, dtype=float)
    design = np.column_stack((np.ones(len(patch_positions)), patch_positions))
    keep = np.ones(len(patch_values), dtype=bool)
    coefficients = None
    for _ in range(5):
        if np.count_nonzero(keep) < 6:
            break
        coefficients, _residuals, rank, _singular = np.linalg.lstsq(
            design[keep], patch_values[keep], rcond=None
        )
        if rank < 3:
            raise RuntimeError(
                "Reference patches are nearly collinear. Place patches across the top, "
                "bottom, left, and right sides of the flat region."
            )
        residual = np.linalg.norm(design @ coefficients - patch_values, axis=1)
        median = float(np.median(residual))
        mad = 1.4826 * float(np.median(np.abs(residual - median))) + 1e-12
        new_keep = residual <= median + 3.5 * mad
        if np.count_nonzero(new_keep) < 6 or np.array_equal(new_keep, keep):
            break
        keep = new_keep
    if coefficients is None or np.linalg.cond(design[keep]) > 1e3:
        raise RuntimeError(
            "Reference-patch layout cannot determine a stable strain background. "
            "Spread the patches over a wider two-dimensional area."
        )
    # Each strain component is a plane in normalized coordinates. Any such
    # linear 2-D strain field satisfies Saint-Venant compatibility. Integrate
    # the three planes into one explicit quadratic displacement field so ux/uy
    # and the displayed strain correction remain mutually consistent.
    x_centered = xx * dx - x_mid
    y_centered = yy * dy - y_mid
    xn, yn = x_centered / x_scale, y_centered / y_scale
    design_grid = np.stack((np.ones_like(xn), xn, yn), axis=-1)
    background = np.einsum("...k,kj->...j", design_grid, coefficients)
    bg_exx, bg_eyy, bg_exy = background[..., 0], background[..., 1], background[..., 2]
    a0, ax, ay = coefficients[:, 0]
    b0, bx, by = coefficients[:, 1]
    c0, cx, cy = coefficients[:, 2]
    ux_background = (
        a0 * x_centered + ax * x_centered**2 / (2.0 * x_scale)
        + ay * x_centered * y_centered / y_scale
        + 0.5 * (2.0 * cy / y_scale - bx / x_scale) * y_centered**2
    )
    uy_background = (
        b0 * y_centered + bx * x_centered * y_centered / x_scale
        + by * y_centered**2 / (2.0 * y_scale) + 2.0 * c0 * x_centered
        + 0.5 * (2.0 * cx / x_scale - ay / y_scale) * x_centered**2
    )
    fitted_patch_values = design @ coefficients
    return {
        "exx": bg_exx,
        "eyy": bg_eyy,
        "exy": bg_exy,
        "ux": ux_background,
        "uy": uy_background,
        "coefficients": coefficients,
        "patch_values": patch_values,
        "patch_fitted_values": fitted_patch_values,
        "patch_positions_normalized": patch_positions,
        "patch_valid_counts": np.asarray(patch_counts, dtype=int),
        "patch_indices": np.asarray(patch_indices, dtype=int),
        "patch_fit_inlier": keep,
    }


def apply_strain_reference_correction(results: dict, mode: str) -> dict:
    """Set selected strain-free patches to a compatible relative-strain baseline."""
    mode = str(mode).lower()
    if mode == "none":
        results["reference_correction_mode"] = np.asarray("none")
        return results
    minimum = 6 if mode == "drift" else 1
    if len(strain_reference_points_xy_nm) < minimum:
        raise RuntimeError(
            f"{mode} reference correction requires at least {minimum} saved reference patches."
        )
    reference_mask = strain_reference_mask(np.asarray(results["exx"]).shape)
    valid_before = np.asarray(results["valid_mask"], dtype=bool)
    sample_mask = reference_mask & valid_before
    height, width = valid_before.shape
    dx_nm, dy_nm = np.asarray(pixel_size_xy, dtype=float) * 1e9
    yy_nm, xx_nm = np.mgrid[:height, :width]
    xx_nm, yy_nm = xx_nm * dx_nm, yy_nm * dy_nm
    valid_patch_count = 0
    for point in strain_reference_points_xy_nm:
        px, py = np.asarray(point, dtype=float)
        patch = ((xx_nm - px) ** 2 + (yy_nm - py) ** 2
                 <= float(strain_reference_radius_nm) ** 2)
        if np.count_nonzero(patch & valid_before) >= 5:
            valid_patch_count += 1
    if valid_patch_count < minimum:
        raise RuntimeError(
            f"Only {valid_patch_count} reference patches contain enough valid data; "
            f"{minimum} are required for {mode} correction. Move patches away from grey edges."
        )
    exx = np.asarray(results["exx"], dtype=float).copy()
    eyy = np.asarray(results["eyy"], dtype=float).copy()
    exy = np.asarray(results["exy"], dtype=float).copy()
    fit_info = None
    if mode == "offset":
        offsets = np.array(
            [np.nanmedian(value[sample_mask]) for value in (exx, eyy, exy)], dtype=float
        )
        if not np.all(np.isfinite(offsets)):
            raise RuntimeError("The selected reference patches contain no valid strain pixels.")
        exx -= offsets[0]
        eyy -= offsets[1]
        exy -= offsets[2]
        coefficients = np.full((3, 3), np.nan)
    elif mode == "drift":
        fit_info = _fit_compatible_reference_strain_background(
            exx, eyy, exy, valid_before
        )
        exx -= fit_info["exx"]
        eyy -= fit_info["eyy"]
        exy -= fit_info["exy"]
        results["ux"] = np.asarray(results["ux"], float) - fit_info["ux"]
        results["uy"] = np.asarray(results["uy"], float) - fit_info["uy"]
        coefficients = fit_info["coefficients"]
        offsets = np.array(
            [np.nanmedian(fit_info[key][sample_mask]) for key in ("exx", "eyy", "exy")],
            dtype=float,
        )
    else:
        raise ValueError(f"Unknown strain-reference correction mode: {mode}")
    # The smooth displacement fit removes spatial drift, but local strain and
    # interpolated displacement are not numerically identical (especially for
    # atomic-site fitting). Anchor the remaining compatible affine strain so
    # the selected patches define the exact relative-strain zero.
    zero_anchor = np.asarray(
        [np.nanmedian(value[sample_mask]) for value in (exx, eyy, exy)], dtype=float
    )
    if not np.all(np.isfinite(zero_anchor)):
        raise RuntimeError("The reference-patch residual strain is not finite.")
    exx -= zero_anchor[0]
    eyy -= zero_anchor[1]
    exy -= zero_anchor[2]
    if "ux" in results and "uy" in results:
        coordinate_anchor = zero_anchor + (offsets if mode == "offset" else 0.0)
        x_centered = xx_nm * 1e-9 - 0.5 * (width - 1) * float(pixel_size_xy[0])
        y_centered = yy_nm * 1e-9 - 0.5 * (height - 1) * float(pixel_size_xy[1])
        results["ux"] = np.asarray(results["ux"], float) - (
            coordinate_anchor[0] * x_centered + coordinate_anchor[2] * y_centered
        )
        results["uy"] = np.asarray(results["uy"], float) - (
            coordinate_anchor[2] * x_centered + coordinate_anchor[1] * y_centered
        )
    invariants = strain_invariants_from_components(exx, eyy, exy)
    new_outliers = valid_before & (
        (np.abs(exx) > MAX_ABS_STRAIN) | (np.abs(eyy) > MAX_ABS_STRAIN)
        | (np.abs(exy) > MAX_ABS_STRAIN) | (invariants["magnitude"] > MAX_ABS_STRAIN)
    )
    valid = valid_before & ~new_outliers
    results["outlier_mask"] = np.asarray(results["outlier_mask"], bool) | new_outliers
    results["valid_mask"] = valid
    for key, value in (("exx", exx), ("eyy", eyy), ("exy", exy)):
        corrected = np.where(valid, value, np.nan)
        results[key] = corrected
        results["raw_" + key] = np.where(valid_before, value, np.nan)
    corrected_invariants = strain_invariants_from_components(
        results["exx"], results["eyy"], results["exy"]
    )
    results.update(corrected_invariants)
    for key, value in corrected_invariants.items():
        results["raw_" + key] = value
    results["reference_correction_mode"] = np.asarray(mode)
    results["reference_patch_mask"] = reference_mask
    results["reference_points_xy_nm"] = np.asarray(strain_reference_points_xy_nm, dtype=float)
    results["reference_patch_radius_nm"] = np.asarray(strain_reference_radius_nm)
    results["reference_background_coefficients"] = np.asarray(coefficients)
    results["reference_background_median_strain"] = offsets
    results["reference_zero_anchor_strain"] = zero_anchor
    results["reference_valid_pixel_count"] = np.asarray(np.count_nonzero(sample_mask))
    results["reference_valid_patch_count"] = np.asarray(valid_patch_count)
    results["reference_corrected_median_strain"] = np.asarray(
        [np.nanmedian(results[key][sample_mask]) for key in ("exx", "eyy", "exy")]
    )
    patch_residuals = np.full((len(strain_reference_points_xy_nm), 3), np.nan)
    patch_counts = np.zeros(len(strain_reference_points_xy_nm), dtype=int)
    for index, point in enumerate(strain_reference_points_xy_nm):
        px, py = np.asarray(point, dtype=float)
        patch = ((xx_nm - px) ** 2 + (yy_nm - py) ** 2
                 <= float(strain_reference_radius_nm) ** 2) & valid
        patch_counts[index] = np.count_nonzero(patch)
        if patch_counts[index] >= 5:
            patch_residuals[index] = [
                np.nanmedian(results[key][patch]) for key in ("exx", "eyy", "exy")
            ]
    results["reference_patch_residual_strain"] = patch_residuals
    results["reference_patch_valid_counts"] = patch_counts
    finite_residuals = np.abs(patch_residuals[np.isfinite(patch_residuals)])
    results["reference_patch_max_abs_residual"] = np.asarray(
        np.max(finite_residuals) if finite_residuals.size else np.nan
    )
    if fit_info is not None:
        results["reference_patch_measured_strain"] = fit_info["patch_values"]
        results["reference_patch_fitted_background_strain"] = fit_info["patch_fitted_values"]
        results["reference_patch_fit_indices"] = fit_info["patch_indices"]
        results["reference_patch_fit_inlier"] = fit_info["patch_fit_inlier"]
    return results


def apply_theoretical_lattice_reference(results, lattice_match) -> dict:
    """Add theoretical global strain to relative GPA maps and revalidate outliers."""
    global_strain = np.asarray(lattice_match["spatial_strain"], dtype=float)
    raw_exx = np.asarray(results["raw_exx"], dtype=float) + global_strain[0, 0]
    raw_eyy = np.asarray(results["raw_eyy"], dtype=float) + global_strain[1, 1]
    raw_exy = np.asarray(results["raw_exy"], dtype=float) + global_strain[0, 1]
    raw_invariants = strain_invariants_from_components(raw_exx, raw_eyy, raw_exy)
    derivative_valid = np.asarray(results["derivative_valid_mask"], dtype=bool)
    outliers = derivative_valid & (
        (np.abs(raw_exx) > MAX_ABS_STRAIN)
        | (np.abs(raw_eyy) > MAX_ABS_STRAIN)
        | (np.abs(raw_exy) > MAX_ABS_STRAIN)
        | (raw_invariants["magnitude"] > MAX_ABS_STRAIN)
    )
    valid = derivative_valid & ~outliers

    results["raw_exx"] = np.where(derivative_valid, raw_exx, np.nan)
    results["raw_eyy"] = np.where(derivative_valid, raw_eyy, np.nan)
    results["raw_exy"] = np.where(derivative_valid, raw_exy, np.nan)
    for key, value in raw_invariants.items():
        results[f"raw_{key}"] = np.where(derivative_valid, value, np.nan)
    results["outlier_mask"] = outliers
    results["valid_mask"] = valid
    results["exx"] = np.where(valid, raw_exx, np.nan)
    results["eyy"] = np.where(valid, raw_eyy, np.nan)
    results["exy"] = np.where(valid, raw_exy, np.nan)
    validated = strain_invariants_from_components(results["exx"], results["eyy"], results["exy"])
    for key, value in validated.items():
        results[key] = value

    results["lattice_global_strain"] = global_strain
    results["lattice_deformation_gradient"] = lattice_match["deformation_gradient"]
    results["lattice_principal_stretches"] = lattice_match["principal_stretches"]
    results["lattice_match_order"] = lattice_match["order"]
    results["lattice_match_signs"] = lattice_match["signs"]
    results["lattice_match_score"] = np.asarray(lattice_match["match_score"])
    warnings = np.asarray(results.get("quality_warnings", []), dtype=str).tolist()
    if lattice_match["match_score"] > 0.10:
        warnings.append(
            "Measured average lattice differs from theory by more than about 10%; "
            "check units, reconstruction, peak order, or scanner calibration."
        )
    if np.any(outliers):
        warnings.append("Some total-strain pixels exceed the 20% small-strain limit.")
    results["quality_warnings"] = np.asarray(list(dict.fromkeys(warnings)), dtype="U")
    return results


def amplitude_valid_mask(amplitude1, amplitude2, sigma: float) -> np.ndarray:
    """Compatibility wrapper returning the confidence-qualified mask."""
    return build_confidence_map(amplitude1, amplitude2, sigma)["confidence_mask"]


def _fraction(numerator_mask, denominator_mask=None) -> float:
    numerator = np.asarray(numerator_mask, dtype=bool)
    if denominator_mask is None:
        return float(np.mean(numerator))
    denominator = np.asarray(denominator_mask, dtype=bool)
    count = np.count_nonzero(denominator)
    return float(np.count_nonzero(numerator & denominator) / count) if count else np.nan


def _roi_values(data, roi, mask=None) -> np.ndarray:
    y1, y2, x1, x2 = roi
    values = np.asarray(data, dtype=float)[y1:y2, x1:x2]
    good = np.isfinite(values)
    if mask is not None:
        good &= np.asarray(mask, dtype=bool)[y1:y2, x1:x2]
    return values[good]


def build_quality_report(results, roi, fit1, fit2, condition_number, sigma) -> tuple[dict, list[str]]:
    """Calculate NaN-aware global and reference-ROI reliability metrics."""
    y1, y2, x1, x2 = roi
    roi_slice = np.s_[y1:y2, x1:x2]
    edge_valid = results["edge_valid_mask"]
    confidence_mask = results["confidence_mask"]
    phase_valid = results["phase_valid_mask"]
    derivative_valid = results["derivative_valid_mask"]
    final_valid = results["valid_mask"]

    confidence_values = _roi_values(results["confidence"], roi, edge_valid)
    ref_confidence = float(np.median(confidence_values)) if confidence_values.size else np.nan
    ref_amp1 = _roi_values(results["confidence_component1"], roi, edge_valid)
    ref_amp2 = _roi_values(results["confidence_component2"], roi, edge_valid)
    strain_std = {}
    for key in ("raw_exx", "raw_eyy", "raw_exy"):
        values = _roi_values(results[key], roi, derivative_valid)
        strain_std[key] = float(np.std(values)) if values.size else np.nan

    metrics = {
        "low_confidence_fraction": _fraction(edge_valid & ~confidence_mask, edge_valid),
        "phase_jump_fraction": _fraction(results["phase_jump_mask"], confidence_mask),
        "edge_invalid_fraction": 1.0 - float(np.mean(edge_valid)),
        "erosion_loss_fraction": 1.0 - _fraction(derivative_valid, phase_valid),
        "outlier_fraction": _fraction(results["outlier_mask"], derivative_valid),
        "final_valid_fraction": float(np.mean(final_valid)),
        "reference_confidence_median": ref_confidence,
        "reference_amplitude1_median_normalized": (
            float(np.median(ref_amp1)) if ref_amp1.size else np.nan
        ),
        "reference_amplitude2_median_normalized": (
            float(np.median(ref_amp2)) if ref_amp2.size else np.nan
        ),
        "reference_phase_valid_fraction": float(np.mean(phase_valid[roi_slice])),
        "reference_derivative_valid_fraction": float(np.mean(derivative_valid[roi_slice])),
        "reference_phase1_fit_rms_rad": float(fit1["residual_rms_rad"]),
        "reference_phase2_fit_rms_rad": float(fit2["residual_rms_rad"]),
        "reference_exx_std": strain_std["raw_exx"],
        "reference_eyy_std": strain_std["raw_eyy"],
        "reference_exy_std": strain_std["raw_exy"],
        "reciprocal_condition_number": float(condition_number),
        "sigma_fft_pixel": float(sigma),
    }

    finite_stds = [v for v in strain_std.values() if np.isfinite(v)]
    metrics["reference_amplitude_pass"] = bool(
        np.isfinite(ref_confidence) and ref_confidence >= max(0.30, CONFIDENCE_THRESHOLD)
    )
    metrics["reference_valid_fraction_pass"] = bool(
        metrics["reference_phase_valid_fraction"] >= 0.70
        and metrics["reference_derivative_valid_fraction"] >= 0.50
    )
    metrics["reference_phase_fit_pass"] = bool(
        max(fit1["residual_rms_rad"], fit2["residual_rms_rad"]) <= 0.50
    )
    metrics["reference_strain_std_pass"] = bool(
        finite_stds and max(finite_stds) <= 0.01
    )
    metrics["reference_quality_pass"] = bool(
        metrics["reference_amplitude_pass"]
        and metrics["reference_valid_fraction_pass"]
        and metrics["reference_phase_fit_pass"]
        and metrics["reference_strain_std_pass"]
    )

    warnings = []
    if metrics["low_confidence_fraction"] > 0.35:
        warnings.append("Result is dominated by low-confidence Bragg amplitude regions.")
    if metrics["phase_jump_fraction"] > 0.10:
        warnings.append("Result is dominated by detected phase-unwrapping jumps.")
    if metrics["edge_invalid_fraction"] > 0.40 or metrics["erosion_loss_fraction"] > 0.40:
        warnings.append("Result is dominated by FFT-mask edges or validity-mask erosion.")
    if metrics["final_valid_fraction"] < 0.30:
        warnings.append("Less than 30% of image pixels remain scientifically valid.")
    if condition_number > 10.0:
        warnings.append(
            f"Reciprocal-vector inversion is poorly conditioned (condition {condition_number:.1f})."
        )
    if not metrics["reference_amplitude_pass"]:
        warnings.append("Reference ROI has low Bragg-field confidence.")
    if not metrics["reference_valid_fraction_pass"]:
        warnings.append("Reference ROI contains too many invalid or phase-jump pixels.")
    if not metrics["reference_phase_fit_pass"]:
        warnings.append("Reference phase-plane fit residual exceeds 0.50 rad.")
    if not metrics["reference_strain_std_pass"]:
        warnings.append("Reference ROI strain standard deviation exceeds 1%.")
    recommended = float(results.get("recommended_sigma_fft_pixel", np.nan))
    if np.isfinite(recommended) and (sigma < 0.5 * recommended or sigma > 2.0 * recommended):
        warnings.append(
            f"FFT sigma {sigma:.2f} is far from the peak-width recommendation {recommended:.2f}."
        )
    if metrics["outlier_fraction"] > 0:
        warnings.append(
            f"{100.0 * metrics['outlier_fraction']:.2f}% of derivative-valid pixels exceed "
            f"the {100.0 * MAX_ABS_STRAIN:.0f}% small-strain reliability limit; they are flagged."
        )
    return metrics, warnings


def compute_strain(sigma: float, smooth_sigma: float = 1.5, median_size: int = 3):
    if img is None or fft_shifted is None:
        raise RuntimeError("请先载入图像。")
    condition_number = _validate_peaks(img.shape, peaks_selected)
    if reference_roi is None:
        raise RuntimeError("请先选择名义无应变参考区。")

    field1, fft_mask1 = bragg_complex_field(fft_shifted, peaks_selected[0], sigma)
    field2, fft_mask2 = bragg_complex_field(fft_shifted, peaks_selected[1], sigma)
    a1, a2 = np.abs(field1), np.abs(field2)
    wrapped1, wrapped2 = np.angle(field1), np.angle(field2)
    confidence_info = build_confidence_map(a1, a2, sigma)
    confidence_mask = confidence_info["confidence_mask"]

    unwrapped1 = unwrap_phase_masked(wrapped1, confidence_mask)
    unwrapped2 = unwrap_phase_masked(wrapped2, confidence_mask)
    jump1, jump_threshold1 = detect_phase_discontinuities(unwrapped1, confidence_mask, 0)
    jump2, jump_threshold2 = detect_phase_discontinuities(unwrapped2, confidence_mask, 0)
    phase_jump_mask = jump1 | jump2
    if PHASE_JUMP_DILATION > 0 and np.any(phase_jump_mask):
        phase_jump_mask = binary_dilation(phase_jump_mask, iterations=PHASE_JUMP_DILATION)
    phase_jump_mask &= confidence_mask
    phase_valid_mask = confidence_mask & ~phase_jump_mask

    y1, y2, x1, x2 = reference_roi
    valid_count = int(np.count_nonzero(phase_valid_mask[y1:y2, x1:x2]))
    if valid_count < 12:
        raise ValueError(
            "Reference ROI has too few confident, phase-continuous pixels; select another region."
        )
    p1, coef1, fit1 = fit_phase_plane_in_roi(
        unwrapped1, reference_roi, a1, phase_valid_mask, return_metrics=True
    )
    p2, coef2, fit2 = fit_phase_plane_in_roi(
        unwrapped2, reference_roi, a2, phase_valid_mask, return_metrics=True
    )
    p1 = np.where(phase_valid_mask, p1, np.nan)
    p2 = np.where(phase_valid_mask, p2, np.nan)

    g1 = reciprocal_vector_from_peak(img.shape, peaks_selected[0])
    g2 = reciprocal_vector_from_peak(img.shape, peaks_selected[1])
    ux_px, uy_px = displacement_from_phases(p1, p2, g1, g2)
    results = calculate_strain_validated(
        ux_px,
        uy_px,
        phase_valid_mask,
        smooth_sigma,
        median_size=median_size,
        spacing_xy=pixel_size_xy,
    )
    results.update(
        {
            "wrapped_phase1": wrapped1,
            "wrapped_phase2": wrapped2,
            "unwrapped_phase1_confidence_masked": unwrapped1,
            "unwrapped_phase2_confidence_masked": unwrapped2,
            "phase1": p1,
            "phase2": p2,
            "amplitude1": a1,
            "amplitude2": a2,
            "fft_mask1": fft_mask1,
            "fft_mask2": fft_mask2,
            "g1_rad_per_pixel": g1,
            "g2_rad_per_pixel": g2,
            "reference_roi": np.asarray(reference_roi, dtype=int),
            "reference_phase_plane_1": coef1,
            "reference_phase_plane_2": coef2,
            "confidence": confidence_info["confidence"],
            "confidence_component1": confidence_info["confidence_component1"],
            "confidence_component2": confidence_info["confidence_component2"],
            "confidence_mask": confidence_mask,
            "low_confidence_mask": confidence_info["edge_valid_mask"] & ~confidence_mask,
            "edge_valid_mask": confidence_info["edge_valid_mask"],
            "phase_jump_mask1": jump1,
            "phase_jump_mask2": jump2,
            "phase_jump_mask": phase_jump_mask,
            "phase_valid_mask": phase_valid_mask,
            "phase_jump_threshold1_xy": jump_threshold1,
            "phase_jump_threshold2_xy": jump_threshold2,
            "sigma_fft_pixel": np.asarray(sigma),
            "smooth_sigma_pixel": np.asarray(smooth_sigma),
            "median_filter_size_pixel": np.asarray(int(median_size)),
            "pixel_size_xy": np.asarray(pixel_size_xy),
            "confidence_threshold": np.asarray(CONFIDENCE_THRESHOLD),
            "strain_outlier_limit": np.asarray(MAX_ABS_STRAIN),
            "reciprocal_condition_number": np.asarray(condition_number),
            "recommended_sigma_fft_pixel": np.asarray(
                recommended_sigma if recommended_sigma is not None else np.nan
            ),
            "selected_peak_rc": np.asarray(peaks_selected, dtype=float),
            "selected_peak_fwhm_axes_pixel": np.asarray(
                [item["fwhm_axes_pixel"] for item in peak_widths], dtype=float
            ) if peak_widths is not None else np.full((2, 2), np.nan),
        }
    )
    metrics, warnings = build_quality_report(
        results, reference_roi, fit1, fit2, condition_number, sigma
    )
    for key, value in metrics.items():
        results[f"quality_{key}"] = np.asarray(value)
    results["quality_warnings"] = np.asarray(warnings, dtype="U")
    return results


def _fit_atom_center(data, predicted_xy, radius, max_shift):
    """Fit one STM atom with independent amplitude and a local planar background."""
    height, width = data.shape
    px, py = map(float, predicted_xy)
    cx, cy = int(round(px)), int(round(py))
    r = int(radius)
    if cx - r < 0 or cx + r >= width or cy - r < 0 or cy + r >= height:
        return None
    patch = np.asarray(data[cy-r:cy+r+1, cx-r:cx+r+1], dtype=float)
    yy, xx = np.mgrid[cy-r:cy+r+1, cx-r:cx+r+1]
    scale = max(float(np.percentile(patch, 95) - np.percentile(patch, 5)), 1e-15)
    z = (patch - np.median(patch)) / scale
    strongest = np.unravel_index(np.argmax(np.abs(z)), z.shape)
    x_init, y_init = float(xx[strongest]), float(yy[strongest])
    amp_init = float(z[strongest])
    sigma_init = max(0.8, min(0.3 * radius, 2.0))
    initial = np.array([x_init, y_init, sigma_init, amp_init, 0.0, 0.0, 0.0])
    lower = np.array([px-max_shift, py-max_shift, 0.45, -4.0, -2.0, -0.5, -0.5])
    upper = np.array([px+max_shift, py+max_shift, max(1.0, 0.8*radius), 4.0, 2.0, 0.5, 0.5])
    initial = np.clip(initial, lower + 1e-6, upper - 1e-6)

    def model(p):
        x0, y0, sigma, amplitude, background, bx, by = p
        return (background + bx*(xx-px) + by*(yy-py)
                + amplitude*np.exp(-((xx-x0)**2 + (yy-y0)**2)/(2*sigma**2)))

    fit = least_squares(
        lambda p: (model(p)-z).ravel(), initial, bounds=(lower, upper),
        loss="soft_l1", f_scale=0.08, max_nfev=500,
    )
    residual = z - model(fit.x)
    variance = float(np.sum((z-np.mean(z))**2))
    r_squared = 1.0 - float(np.sum(residual**2))/max(variance, 1e-15)
    shift = np.linalg.norm(fit.x[:2] - np.array([px, py]))
    if not fit.success or r_squared < 0.18 or shift >= 0.98*max_shift or abs(fit.x[3]) < 0.05:
        return None
    return np.array(fit.x[:2]), float(r_squared), float(abs(fit.x[3]))


def detect_atomic_spots(data, lattice_spacing_px, expected_count) -> tuple[np.ndarray, np.ndarray]:
    """Detect atomic extrema first, independently of the theoretical lattice origin."""
    z = np.asarray(data, dtype=float)
    spacing = max(float(lattice_spacing_px), 4.0)
    # Remove the broad electronic/topographic envelope while retaining atomic
    # features. Local RMS normalization makes dim and bright areas comparable.
    fine = gaussian_filter(z, 0.55, mode="nearest")
    background = gaussian_filter(z, max(2.0, 0.48*spacing), mode="nearest")
    residual = fine-background
    local_rms = np.sqrt(
        gaussian_filter(residual**2, max(2.0, 0.65*spacing), mode="nearest")
    )
    normalized = np.divide(
        residual, np.maximum(local_rms, np.finfo(float).eps),
        out=np.zeros_like(residual), where=local_rms > 0,
    )
    # The requested lattice markers are the bright atomic sites. Restricting
    # detection to positive extrema avoids matching dark interstitial minima as
    # if they were a second, shifted lattice.
    response = normalized
    nms_size = max(3, int(round(0.48*spacing)))
    if nms_size % 2 == 0:
        nms_size += 1
    local_max = response >= maximum_filter(response, size=nms_size, mode="nearest")-1e-12
    margin = max(3, int(np.ceil(0.24*spacing)))
    local_max[:margin] = local_max[-margin:] = False
    local_max[:, :margin] = local_max[:, -margin:] = False
    rr, cc = np.nonzero(local_max & (response >= 1.05))
    if rr.size == 0:
        raise RuntimeError("No atomic spots were detected after local background removal.")
    scores = response[rr, cc]
    max_candidates = max(24, int(np.ceil(2.2*max(expected_count, 1))))
    order = np.argsort(scores)[::-1][:max_candidates]
    rr, cc, scores = rr[order], cc[order], scores[order]

    centers, qualities = [], []
    refine_radius = max(2, min(4, int(round(0.18*spacing))))
    height, width = z.shape
    for row, col, score in zip(rr, cc, scores):
        y1, y2 = max(0, row-refine_radius), min(height, row+refine_radius+1)
        x1, x2 = max(0, col-refine_radius), min(width, col+refine_radius+1)
        patch = normalized[y1:y2, x1:x2]
        yy, xx = np.mgrid[y1:y2, x1:x2]
        weights = np.maximum(patch, 0.0)**2
        weights *= np.exp(-((yy-row)**2+(xx-col)**2)/(2.0*max(refine_radius/1.5, 0.8)**2))
        total = float(np.sum(weights))
        if total <= 0:
            continue
        center = np.array([np.sum(xx*weights)/total, np.sum(yy*weights)/total])
        if np.linalg.norm(center-np.array([col, row])) > refine_radius:
            continue
        centers.append(center)
        qualities.append(float(score))
    if len(centers) < 8:
        raise RuntimeError(f"Only {len(centers)} atomic candidates were detected.")
    return np.asarray(centers), np.asarray(qualities)


def match_spots_to_lattice(predicted_px, detected_px, detected_quality, spacing_px):
    """One-to-one assignment, followed by a robust affine update and reassignment."""
    predicted = np.asarray(predicted_px, dtype=float)
    detected = np.asarray(detected_px, dtype=float)
    gate = 0.46*float(spacing_px)

    def assign(targets):
        distances = np.linalg.norm(targets[:, None, :]-detected[None, :, :], axis=2)
        rows, cols = linear_sum_assignment(distances)
        keep = distances[rows, cols] <= gate
        return rows[keep], cols[keep], distances[rows[keep], cols[keep]]

    rows, cols, residuals = assign(predicted)
    if len(rows) < 8:
        raise RuntimeError(
            "Too few detected spots match the theoretical grid; realign the unit-cell frame."
        )
    # Robustly estimate only the global registration used for assignment. The
    # original theoretical grid is retained later when calculating displacement.
    design = np.column_stack((predicted[rows], np.ones(len(rows))))
    keep = np.ones(len(rows), dtype=bool)
    coef = None
    for _ in range(3):
        coef, *_ = np.linalg.lstsq(design[keep], detected[cols][keep], rcond=None)
        error = np.linalg.norm(design@coef-detected[cols], axis=1)
        median = np.median(error)
        mad = 1.4826*np.median(np.abs(error-median))+1e-9
        keep = error <= max(0.22*spacing_px, median+3.0*mad)
        if np.count_nonzero(keep) < 8:
            break
    warped = np.column_stack((predicted, np.ones(len(predicted))))@coef
    rows, cols, residuals = assign(warped)
    quality = np.clip(detected_quality[cols]/4.0, 0.05, 1.0)
    return rows, cols, residuals, quality, warped


def compute_atomic_registry_strain(lattice_match=None) -> dict:
    """Calculate strain from fitted atomic positions seeded by the aligned FFT cell."""
    if img is None or unit_cell_origin_xy is None or unit_cell_basis_xy is None:
        raise RuntimeError("Align and confirm the FFT-derived unit cell first.")
    if spatial_unit != "m":
        raise RuntimeError("Physical SXM calibration is required for atomic-position strain.")
    dx_nm, dy_nm = np.asarray(pixel_size_xy, dtype=float) * 1e9
    origin_nm = np.asarray(unit_cell_origin_xy, dtype=float)
    basis_nm = np.asarray(unit_cell_basis_xy, dtype=float)
    origin_px = origin_nm / np.array([dx_nm, dy_nm])
    basis_px = np.diag([1.0/dx_nm, 1.0/dy_nm]) @ basis_nm
    corners_px = np.array([[0, 0], [img.shape[1]-1, 0],
                           [0, img.shape[0]-1], [img.shape[1]-1, img.shape[0]-1]], float)
    uv = (np.linalg.inv(basis_px) @ (corners_px-origin_px).T).T
    imin, jmin = np.floor(np.min(uv, axis=0)).astype(int) - 1
    imax, jmax = np.ceil(np.max(uv, axis=0)).astype(int) + 1
    lattice_indices = np.array([(i, j) for j in range(jmin, jmax+1)
                                for i in range(imin, imax+1)], dtype=float)
    predicted_px = origin_px + lattice_indices @ basis_px.T
    spacing_px = float(min(np.linalg.norm(basis_px[:, 0]), np.linalg.norm(basis_px[:, 1])))
    inside = ((predicted_px[:, 0] >= 1) & (predicted_px[:, 0] < img.shape[1]-1)
              & (predicted_px[:, 1] >= 1) & (predicted_px[:, 1] < img.shape[0]-1))
    predicted_px = predicted_px[inside]
    lattice_indices = lattice_indices[inside]
    detected_px, detected_scores = detect_atomic_spots(
        img, spacing_px, expected_count=len(predicted_px)
    )
    matched_rows, matched_cols, match_residuals, qualities, warped_px = (
        match_spots_to_lattice(predicted_px, detected_px, detected_scores, spacing_px)
    )
    # Refine the already matched candidates on a background-suppressed image.
    # Detection establishes correspondence first; this fit only improves the
    # subpixel center and cannot jump to another theoretical lattice site.
    localization_image = (
        gaussian_filter(img, 0.55, mode="nearest")
        - gaussian_filter(img, max(2.0, 0.48*spacing_px), mode="nearest")
    )
    refined_detected = detected_px[matched_cols].copy()
    refine_radius = int(np.clip(round(0.30*spacing_px), 3, 8))
    refine_shift = float(np.clip(0.16*spacing_px, 1.2, max(1.5, refine_radius-0.5)))
    for k, center in enumerate(refined_detected):
        item = _fit_atom_center(localization_image, center, refine_radius, refine_shift)
        if item is not None:
            refined_detected[k] = item[0]
            qualities[k] = max(qualities[k], min(float(item[1]), 1.0))
    scale_xy = np.array([dx_nm, dy_nm])
    predicted = predicted_px[matched_rows] * scale_xy
    fitted = refined_detected * scale_xy
    match_residuals = np.linalg.norm(refined_detected-warped_px[matched_rows], axis=1)
    if fitted.shape[0] < 16:
        raise RuntimeError(
            f"Only {fitted.shape[0]} detected atoms match the theoretical lattice; "
            "realign the cell or use a larger analysis region."
        )

    correction = (np.eye(2) if lattice_match is None
                  else np.asarray(lattice_match["coordinate_correction"], dtype=float))
    correction_inv = np.linalg.inv(correction)
    neighbour_radius = 2.25 * max(np.linalg.norm(basis_nm[:, 0]),
                                  np.linalg.norm(basis_nm[:, 1]))
    site_strain = np.full((len(fitted), 3), np.nan)
    site_displacement = fitted - predicted
    for k, center in enumerate(predicted):
        distances = np.linalg.norm(predicted-center, axis=1)
        neighbours = distances <= neighbour_radius
        if np.count_nonzero(neighbours) < 6:
            continue
        design = np.column_stack((predicted[neighbours], np.ones(np.count_nonzero(neighbours))))
        weights = np.sqrt(np.maximum(qualities[neighbours], 0.05))
        coef, *_ = np.linalg.lstsq(design*weights[:, None],
                                   fitted[neighbours]*weights[:, None], rcond=None)
        deformation = coef[:2, :].T
        gradient = correction @ (deformation-np.eye(2)) @ correction_inv
        site_strain[k] = (gradient[0, 0], gradient[1, 1],
                          0.5*(gradient[0, 1]+gradient[1, 0]))
    site_valid = np.all(np.isfinite(site_strain), axis=1)
    if np.count_nonzero(site_valid) < 10:
        raise RuntimeError("Too few atomic neighbourhoods remain for a stable strain map.")
    site_strain[site_valid] -= np.nanmedian(site_strain[site_valid], axis=0)

    lx_nm, ly_nm = np.asarray(scan_size_xy, dtype=float) * 1e9
    yy_nm, xx_nm = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    xx_nm = xx_nm * dx_nm
    yy_nm = yy_nm * dy_nm
    maps = []
    for component in range(3):
        dense = griddata(predicted[site_valid], site_strain[site_valid, component],
                         (xx_nm, yy_nm), method="linear")
        maps.append(dense)
    exx, eyy, exy = maps
    valid = np.isfinite(exx) & np.isfinite(eyy) & np.isfinite(exy)
    exx = nan_gaussian_filter(exx, 1.0, valid)
    eyy = nan_gaussian_filter(eyy, 1.0, valid)
    exy = nan_gaussian_filter(exy, 1.0, valid)
    invariants = strain_invariants_from_components(exx, eyy, exy)
    outliers = valid & (invariants["magnitude"] > MAX_ABS_STRAIN)
    valid &= ~outliers
    exx, eyy, exy = [np.where(valid, value, np.nan) for value in (exx, eyy, exy)]
    invariants = strain_invariants_from_components(exx, eyy, exy)

    displacement_m = site_displacement * 1e-9
    ux = griddata(predicted, displacement_m[:, 0], (xx_nm, yy_nm), method="linear")
    uy = griddata(predicted, displacement_m[:, 1], (xx_nm, yy_nm), method="linear")
    results = {"exx": exx, "eyy": eyy, "exy": exy, **invariants,
               "ux": ux, "uy": uy, "valid_mask": valid, "outlier_mask": outliers,
               "phase_jump_mask": np.zeros(img.shape, bool),
               "auto_mask_enabled": np.asarray(False),
               "analysis_mode": np.asarray("atomic_registry"),
               "atomic_predicted_xy_nm": predicted,
               "atomic_fitted_xy_nm": fitted,
               "atomic_detected_xy_nm": detected_px*scale_xy,
               "atomic_detection_score": detected_scores,
               "atomic_fit_r_squared": qualities,
               "atomic_match_residual_pixel": match_residuals,
               "atomic_warped_grid_xy_nm": warped_px*scale_xy,
               "atomic_site_strain": site_strain,
               "coordinate_correction": correction}
    for key in ("exx", "eyy", "exy", *invariants.keys()):
        results["raw_"+key] = np.asarray(results[key])
    return results


def compute_v1_style_relative_strain(
    sigma: float, smooth_sigma: float = 1.0, lattice_match=None
) -> dict:
    """V1-like relative maps with targeted phase-singularity and edge rejection."""
    if img is None or fft_shifted is None or peaks_selected is None:
        raise RuntimeError("Load an image and select two Bragg peaks first.")
    _validate_peaks(img.shape, peaks_selected)
    field1, fft_mask1 = bragg_complex_field(fft_shifted, peaks_selected[0], sigma)
    field2, fft_mask2 = bragg_complex_field(fft_shifted, peaks_selected[1], sigma)
    amplitude1, amplitude2 = np.abs(field1), np.abs(field2)
    use_auto_mask = bool(auto_mask_var.get()) if auto_mask_var is not None else False
    full_roi = (0, img.shape[0], 0, img.shape[1])
    coordinate_correction = (
        None if lattice_match is None else lattice_match["coordinate_correction"]
    )

    if not use_auto_mask:
        phase1 = unwrap_phase(np.angle(field1))
        phase2 = unwrap_phase(np.angle(field2))
        phase1, coef1 = fit_phase_plane_in_roi(phase1, full_roi)
        phase2, coef2 = fit_phase_plane_in_roi(phase2, full_roi)
        g1 = reciprocal_vector_from_peak(img.shape, peaks_selected[0])
        g2 = reciprocal_vector_from_peak(img.shape, peaks_selected[1])
        ux_px, uy_px = displacement_from_phases(phase1, phase2, g1, g2)
        dx, dy = map(float, pixel_size_xy)
        ux = gaussian_filter(ux_px * dx, smooth_sigma)
        uy = gaussian_filter(uy_px * dy, smooth_sigma)
        dux_dy, dux_dx = np.gradient(ux, dy, dx, edge_order=2)
        duy_dy, duy_dx = np.gradient(uy, dy, dx, edge_order=2)
        dux_dy, dux_dx, duy_dy, duy_dx = correct_displacement_gradient(
            dux_dy, dux_dx, duy_dy, duy_dx, coordinate_correction
        )
        if coordinate_correction is not None:
            correction = np.asarray(coordinate_correction, dtype=float)
            ux, uy = (
                correction[0, 0] * ux + correction[0, 1] * uy,
                correction[1, 0] * ux + correction[1, 1] * uy,
            )
        results = strain_components(dux_dy, dux_dx, duy_dy, duy_dx)
        exx = results["exx"] - np.mean(results["exx"])
        eyy = results["eyy"] - np.mean(results["eyy"])
        exy = results["exy"] - np.mean(results["exy"])
        results["exx"], results["eyy"], results["exy"] = exx, eyy, exy
        results.update(strain_invariants_from_components(exx, eyy, exy))
        results.update(
            {
                "ux": ux,
                "uy": uy,
                "phase1": phase1,
                "phase2": phase2,
                "amplitude1": amplitude1,
                "amplitude2": amplitude2,
                "reference_phase_plane_1": coef1,
                "reference_phase_plane_2": coef2,
                "g1_rad_per_pixel": g1,
                "g2_rad_per_pixel": g2,
                "valid_mask": np.ones(img.shape, dtype=bool),
                "outlier_mask": np.zeros(img.shape, dtype=bool),
                "phase_jump_mask": np.zeros(img.shape, dtype=bool),
                "sigma_fft_pixel": np.asarray(sigma),
                "smooth_sigma_pixel": np.asarray(smooth_sigma),
                "auto_mask_enabled": np.asarray(False),
                "coordinate_correction": np.asarray(
                    np.eye(2) if coordinate_correction is None else coordinate_correction
                ),
            }
        )
        return results

    confidence_info = build_confidence_map(
        amplitude1, amplitude2, sigma, threshold=0.06, edge_factor=0.5
    )
    confidence_mask = confidence_info["confidence_mask"]
    phase1 = unwrap_phase_masked(np.angle(field1), confidence_mask)
    phase2 = unwrap_phase_masked(np.angle(field2), confidence_mask)
    jump1, threshold1 = detect_phase_discontinuities(phase1, confidence_mask, dilation=0)
    jump2, threshold2 = detect_phase_discontinuities(phase2, confidence_mask, dilation=0)
    phase_jump_mask = jump1 | jump2
    if np.any(phase_jump_mask):
        phase_jump_mask = binary_dilation(phase_jump_mask, iterations=2)
    phase_valid_mask = confidence_mask & ~phase_jump_mask
    phase1, coef1 = fit_phase_plane_in_roi(phase1, full_roi, amplitude1, phase_valid_mask)
    phase2, coef2 = fit_phase_plane_in_roi(phase2, full_roi, amplitude2, phase_valid_mask)
    phase1 = np.where(phase_valid_mask, phase1, np.nan)
    phase2 = np.where(phase_valid_mask, phase2, np.nan)
    g1 = reciprocal_vector_from_peak(img.shape, peaks_selected[0])
    g2 = reciprocal_vector_from_peak(img.shape, peaks_selected[1])
    ux_px, uy_px = displacement_from_phases(phase1, phase2, g1, g2)
    results = calculate_strain_validated(
        ux_px,
        uy_px,
        phase_valid_mask,
        smooth_sigma,
        median_size=0,
        spacing_xy=pixel_size_xy,
        erosion_iterations=2,
        derivative_radius=2,
        outlier_limit=MAX_ABS_STRAIN,
        coordinate_correction=coordinate_correction,
    )

    exx = results["exx"] - np.nanmean(results["exx"])
    eyy = results["eyy"] - np.nanmean(results["eyy"])
    exy = results["exy"] - np.nanmean(results["exy"])
    invariants = strain_invariants_from_components(exx, eyy, exy)
    post_outliers = results["valid_mask"] & (
        (np.abs(exx) > MAX_ABS_STRAIN)
        | (np.abs(eyy) > MAX_ABS_STRAIN)
        | (np.abs(exy) > MAX_ABS_STRAIN)
        | (invariants["magnitude"] > MAX_ABS_STRAIN)
    )
    results["outlier_mask"] |= post_outliers
    results["valid_mask"] &= ~post_outliers
    exx = np.where(results["valid_mask"], exx, np.nan)
    eyy = np.where(results["valid_mask"], eyy, np.nan)
    exy = np.where(results["valid_mask"], exy, np.nan)
    results["exx"], results["eyy"], results["exy"] = exx, eyy, exy
    results.update(strain_invariants_from_components(exx, eyy, exy))
    results.update(
        {
            "phase1": phase1,
            "phase2": phase2,
            "amplitude1": amplitude1,
            "amplitude2": amplitude2,
            "confidence": confidence_info["confidence"],
            "confidence_mask": confidence_mask,
            "phase_jump_mask": phase_jump_mask,
            "phase_valid_mask": phase_valid_mask,
            "edge_valid_mask": confidence_info["edge_valid_mask"],
            "phase_jump_threshold1_xy": threshold1,
            "phase_jump_threshold2_xy": threshold2,
            "fft_mask1": fft_mask1,
            "fft_mask2": fft_mask2,
            "reference_phase_plane_1": coef1,
            "reference_phase_plane_2": coef2,
            "g1_rad_per_pixel": g1,
            "g2_rad_per_pixel": g2,
            "sigma_fft_pixel": np.asarray(sigma),
            "smooth_sigma_pixel": np.asarray(smooth_sigma),
            "auto_mask_enabled": np.asarray(True),
            "coordinate_correction": np.asarray(
                np.eye(2) if coordinate_correction is None else coordinate_correction
            ),
        }
    )
    return results


# ---------------------------------------------------------------------------
# Plotting and export
# ---------------------------------------------------------------------------
def robust_limit(data, percentile=98.0, symmetric=True):
    numeric = np.asarray(data, dtype=float)
    values = numeric[np.isfinite(numeric)]
    if values.size == 0:
        return (-1.0, 1.0) if symmetric else (0.0, 1.0)
    if symmetric:
        lim = max(float(np.percentile(np.abs(values), percentile)), 1e-12)
        return -lim, lim
    lo, hi = np.percentile(values, (100.0 - percentile, percentile))
    if np.min(values) >= 0:
        lo = 0.0
    if hi <= lo:
        hi = lo + 1e-12
    return float(lo), float(hi)


def rotated_zoom_corners_xy(bounds, angle_deg):
    """Return the closed 4-corner loop of a zoom box rotated about its center."""
    x0, x1, y0, y1 = bounds
    center = np.array([0.5 * (x0 + x1), 0.5 * (y0 + y1)])
    hw, hh = 0.5 * (x1 - x0), 0.5 * (y1 - y0)
    local = np.array([(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh), (-hw, -hh)])
    theta = np.deg2rad(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    rotation = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    return local @ rotation.T + center


def rotated_crop_sample(data, center_xy_nm, half_extent_xy_nm, angle_deg, pixel_xy_nm, out_shape):
    """Resample a rotated rectangular ROI so its own axes become axis-aligned.

    Rotation is done in physical (x, y) units rather than pixel indices so
    that non-square STM pixel spacing still yields a metrically correct crop.
    """
    data = np.asarray(data, dtype=float)
    finite = np.isfinite(data)
    filled = np.where(finite, data, 0.0)
    out_h, out_w = out_shape
    dx_nm, dy_nm = map(float, pixel_xy_nm)
    hw_nm, hh_nm = map(float, half_extent_xy_nm)
    u = (np.arange(out_w) + 0.5 - out_w / 2.0) * (2.0 * hw_nm / out_w)
    v = (np.arange(out_h) + 0.5 - out_h / 2.0) * (2.0 * hh_nm / out_h)
    uu, vv = np.meshgrid(u, v)
    theta = np.deg2rad(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    offset_x = uu * cos_t - vv * sin_t
    offset_y = uu * sin_t + vv * cos_t
    cx_nm, cy_nm = map(float, center_xy_nm)
    src_col = (cx_nm + offset_x) / dx_nm
    src_row = (cy_nm + offset_y) / dy_nm
    coords = np.stack([src_row.ravel(), src_col.ravel()])
    values = map_coordinates(filled, coords, order=1, mode="constant", cval=0.0)
    valid = map_coordinates(finite.astype(float), coords, order=1, mode="constant", cval=0.0)
    values = values.reshape(out_h, out_w)
    valid = valid.reshape(out_h, out_w)
    values[valid < 0.5] = np.nan
    return values


def masked_percent(results, key):
    data = 100.0 * np.asarray(results[key], dtype=float)
    return np.where(results["valid_mask"], data, np.nan)


def strain_to_rgb(e1, e2, theta, valid_mask=None):
    magnitude = np.maximum(np.abs(e1), np.abs(e2))
    finite = magnitude[np.isfinite(magnitude)]
    scale = np.percentile(finite, 98.0) if finite.size else 1.0
    value = np.clip(magnitude / max(float(scale), 1e-12), 0.0, 1.0)
    hue = np.mod(theta, np.pi) / np.pi
    hsv = np.stack((hue, np.ones_like(hue), value), axis=-1)
    rgb = hsv_to_rgb(hsv)
    if valid_mask is not None:
        rgb = np.where(valid_mask[..., None], rgb, 0.75)
    return rgb


def annular_profile_statistics(
    data, valid_mask, center_xy, spacing_xy=(1.0, 1.0), bin_width=None, min_count=5
):
    """Return annular median, IQR, and counts around a manually chosen center."""
    values = np.asarray(data, dtype=float)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(values)
    dx, dy = map(float, spacing_xy)
    if bin_width is None:
        bin_width = min(dx, dy)
    if bin_width <= 0:
        raise ValueError("Annular bin width must be positive.")
    cx, cy = map(float, center_xy)
    yy, xx = np.indices(values.shape)
    radius = np.hypot((xx - cx) * dx, (yy - cy) * dy)
    bins = np.floor(radius / bin_width).astype(int)
    max_bin = int(np.max(bins[valid])) if np.any(valid) else -1
    radii = (np.arange(max_bin + 1, dtype=float) + 0.5) * bin_width
    median = np.full(max_bin + 1, np.nan)
    q25 = np.full(max_bin + 1, np.nan)
    q75 = np.full(max_bin + 1, np.nan)
    counts = np.zeros(max_bin + 1, dtype=int)
    for index in range(max_bin + 1):
        samples = values[valid & (bins == index)]
        counts[index] = samples.size
        if samples.size >= int(min_count):
            q25[index], median[index], q75[index] = np.percentile(samples, (25, 50, 75))
    return {
        "radius": radii,
        "median": median,
        "q25": q25,
        "q75": q75,
        "count": counts,
        "center_xy_pixel": np.asarray((cx, cy), dtype=float),
        "bin_width": np.asarray(bin_width),
    }


def _set_map(artist, data, symmetric=True, limits=None):
    artist.set_data(data)
    artist.set_clim(*(limits or robust_limit(data, symmetric=symmetric)))


def common_symmetric_limit(*arrays, percentile=98.0):
    finite = [np.asarray(a)[np.isfinite(a)] for a in arrays]
    finite = [a for a in finite if a.size]
    if not finite:
        return -1.0, 1.0
    joined = np.concatenate(finite)
    limit = max(float(np.percentile(np.abs(joined), percentile)), 1e-12)
    return -limit, limit


def quality_status_text(results, full=False) -> str:
    warnings = np.asarray(results.get("quality_warnings", []), dtype=str).tolist()
    valid = 100.0 * float(np.mean(results["valid_mask"]))
    condition = float(results["reciprocal_condition_number"])
    sigma = float(results["sigma_fft_pixel"])
    recommended = float(results.get("recommended_sigma_fft_pixel", np.nan))
    sigma_text = f"FFT sigma: {sigma:.2f}"
    if np.isfinite(recommended):
        sigma_text += f" (recommended {recommended:.2f})"
    prefix = (
        f"Valid pixels: {valid:.1f}% | reciprocal condition: {condition:.2f} | {sigma_text}"
    )
    if warnings:
        if full:
            return prefix + "\nWARNINGS:\n- " + "\n- ".join(warnings)
        extra = f" (+{len(warnings) - 1} more; see Advanced diagnostics)" if len(warnings) > 1 else ""
        return prefix + "\nWARNING: " + warnings[0] + extra
    return prefix + "\nQuality checks passed at the configured thresholds."


def update_plot(_value=None) -> None:
    global last_results
    if fig_res is None:
        return
    try:
        results = compute_strain(
            float(sigma_slider.val),
            float(smooth_slider.val),
            int(median_slider.val),
        )
    except Exception as exc:
        messagebox.showerror("计算失败", str(exc))
        return
    last_results = results

    exx = masked_percent(results, "exx")
    eyy = masked_percent(results, "eyy")
    exy = masked_percent(results, "exy")
    e1 = masked_percent(results, "principal_max")
    e2 = masked_percent(results, "principal_min")
    magnitude = masked_percent(results, "magnitude")
    tensor_limits = common_symmetric_limit(exx, eyy, exy)
    principal_limits = common_symmetric_limit(e1, e2)
    _set_map(result_artists["exx"], exx, limits=tensor_limits)
    _set_map(result_artists["eyy"], eyy, limits=tensor_limits)
    _set_map(result_artists["exy"], exy, limits=tensor_limits)
    _set_map(result_artists["e1"], e1, limits=principal_limits)
    _set_map(result_artists["e2"], e2, limits=principal_limits)
    _set_map(result_artists["magnitude"], magnitude, symmetric=False)
    status = quality_status_text(results)
    result_artists["status"].set_text(status)
    result_artists["status"].set_color("darkred" if len(results["quality_warnings"]) else "darkgreen")
    fig_res.canvas.draw_idle()


def read_lattice_inputs() -> tuple[float, float, float]:
    if lattice_a_var is None or lattice_b_var is None or lattice_angle_var is None:
        raise RuntimeError("Lattice input controls are not initialized.")
    try:
        a_nm = float(lattice_a_var.get())
        b_nm = float(lattice_b_var.get())
        angle_deg = float(lattice_angle_var.get())
    except ValueError as exc:
        raise ValueError("Enter numeric values for a, b, and angle.") from exc
    theoretical_direct_basis(a_nm, b_nm, angle_deg)
    return a_nm, b_nm, angle_deg


def mask_sigma_bounds() -> tuple[float, float]:
    """Return a safe FFT-mask range that avoids DC and the other selected peak."""
    if img is None or peaks_selected is None:
        return 3.0, 30.0
    center = np.array([img.shape[0] // 2, img.shape[1] // 2], dtype=float)
    offsets = [np.asarray(point, dtype=float) - center for point in peaks_selected]
    distances = [np.linalg.norm(vector) for vector in offsets]
    distances.extend(
        (np.linalg.norm(offsets[0] - offsets[1]), np.linalg.norm(offsets[0] + offsets[1]))
    )
    return 3.0, max(3.5, min(30.0, 0.35 * min(distances)))


def refresh_auto_mask() -> None:
    """Recalculate an open result after toggling automatic reliability masking."""
    if fig_res is not None and sigma_slider is not None:
        sigma_slider.set_val(float(sigma_slider.val))


def show_strain_distribution(results, a_nm, b_nm, angle_deg) -> None:
    """Show only the final scalar strain-distribution map."""
    global fig_res
    if fig_res is not None:
        plt.close(fig_res)
    magnitude = 100.0 * np.asarray(results["magnitude"], dtype=float)
    finite = magnitude[np.isfinite(magnitude)]
    if finite.size == 0:
        raise ValueError("No valid strain pixels remain after reliability masking.")
    upper = max(float(np.percentile(finite, 98.0)), 1e-9)
    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad("#b0b0b0")
    fig_res, ax = plt.subplots(figsize=(9, 7.5), num="STM strain distribution")
    lx_nm, ly_nm = np.asarray(results["scan_size_xy_m"], dtype=float) * 1e9
    artist = ax.imshow(
        magnitude,
        cmap=cmap,
        origin="upper",
        vmin=0.0,
        vmax=upper,
        extent=(0.0, lx_nm, ly_nm, 0.0),
    )
    fig_res.colorbar(artist, ax=ax, label="Strain magnitude (%)")
    ax.set_title(
        r"Strain distribution: $\max(|\varepsilon_1|,|\varepsilon_2|)$"
        + f"\nTheory: a={a_nm:g} nm, b={b_nm:g} nm, angle={angle_deg:g} deg"
    )
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    if np.any(results["outlier_mask"]):
        ax.contour(
            results["outlier_mask"],
            levels=[0.5],
            colors="cyan",
            linewidths=0.8,
            extent=(0.0, lx_nm, ly_nm, 0.0),
        )
    valid_fraction = 100.0 * float(np.mean(results["valid_mask"]))
    note = f"Valid pixels: {valid_fraction:.1f}%. Cyan contours mark values beyond the reliable range."
    warnings = np.asarray(results.get("quality_warnings", []), dtype=str).tolist()
    if warnings:
        note += "\nWARNING: " + warnings[0]
    fig_res.text(0.02, 0.015, note, fontsize=8, color="darkred" if warnings else "#333333")
    fig_res.tight_layout(rect=(0, 0.055, 1, 1))
    plt.show(block=False)
    fig_res.canvas.draw_idle()


def sample_strain_along_line(results, start_nm, end_nm) -> dict:
    """Sample current strain maps along a physical line without bridging NaNs."""
    h, w = np.asarray(results["exx"]).shape
    lx_nm, ly_nm = np.asarray(scan_size_xy, dtype=float) * 1e9
    x0, y0 = map(float, start_nm)
    x1, y1 = map(float, end_nm)
    col0 = np.clip(x0 / lx_nm * w - 0.5, 0, w - 1)
    col1 = np.clip(x1 / lx_nm * w - 0.5, 0, w - 1)
    row0 = np.clip(y0 / ly_nm * h - 0.5, 0, h - 1)
    row1 = np.clip(y1 / ly_nm * h - 0.5, 0, h - 1)
    pixel_length = float(np.hypot(col1 - col0, row1 - row0))
    count = max(2, int(np.ceil(2.0 * pixel_length)) + 1)
    rows = np.linspace(row0, row1, count)
    cols = np.linspace(col0, col1, count)
    coordinates = np.vstack((rows, cols))
    distance = np.linspace(0.0, np.hypot(x1 - x0, y1 - y0), count)

    output = {
        "distance_nm": distance,
        "x_nm": np.linspace(x0, x1, count),
        "y_nm": np.linspace(y0, y1, count),
        "start_nm": np.asarray((x0, y0)),
        "end_nm": np.asarray((x1, y1)),
    }
    for key in (
        "exx", "eyy", "exy", "mean_normal", "dilatation",
        "principal_max", "principal_min", "magnitude",
    ):
        data = 100.0 * np.asarray(results[key], dtype=float)
        finite = np.isfinite(data)
        numerator = map_coordinates(
            np.where(finite, data, 0.0), coordinates, order=1, mode="nearest"
        )
        support = map_coordinates(
            finite.astype(float), coordinates, order=1, mode="nearest"
        )
        output[key] = np.where(support > 0.999, numerator / np.maximum(support, 1e-12), np.nan)
    return output


def show_line_profile(results, start_nm, end_nm) -> None:
    """Open a separate window with signed and scalar strain profiles."""
    profile = sample_strain_along_line(results, start_nm, end_nm)
    if not np.any(np.isfinite(profile["magnitude"])):
        messagebox.showwarning("Line profile", "The selected line contains no valid strain pixels.")
        return
    results["line_profile_distance_nm"] = profile["distance_nm"]
    results["line_profile_x_nm"] = profile["x_nm"]
    results["line_profile_y_nm"] = profile["y_nm"]
    results["line_profile_start_nm"] = profile["start_nm"]
    results["line_profile_end_nm"] = profile["end_nm"]
    for key in (
        "exx", "eyy", "exy", "mean_normal", "dilatation",
        "principal_max", "principal_min", "magnitude",
    ):
        results[f"line_profile_{key}_percent"] = profile[key]

    figure, axis = plt.subplots(figsize=(9, 5.5), num="Strain along selected line")
    distance = profile["distance_nm"]
    axis.plot(distance, profile["exx"], label=r"$\varepsilon_{xx}$")
    axis.plot(distance, profile["eyy"], label=r"$\varepsilon_{yy}$")
    axis.plot(distance, profile["mean_normal"], color="#7b3294", linewidth=1.8,
              label=r"$\varepsilon_{mean}=(\varepsilon_{xx}+\varepsilon_{yy})/2$")
    axis.plot(distance, profile["principal_max"], label=r"$\varepsilon_1$")
    axis.plot(distance, profile["magnitude"], color="black", linewidth=1.8,
              label=r"$\max(|\varepsilon_1|,|\varepsilon_2|)$")
    axis.axhline(0.0, color="#777777", linewidth=0.7)
    axis.set_xlabel("Distance along line (nm)")
    axis.set_ylabel("Strain (%)")
    axis.set_title(
        f"Line: ({start_nm[0]:.2f}, {start_nm[1]:.2f}) to "
        f"({end_nm[0]:.2f}, {end_nm[1]:.2f}) nm"
    )
    axis.grid(alpha=0.25)
    axis.legend(ncol=2)
    figure.subplots_adjust(left=0.10, right=0.97, top=0.88, bottom=0.22)

    clipboard_columns = (
        ("distance_nm", "distance_nm"),
        ("mean_strain_percent", "mean_normal"),
    )

    def copy_profile_to_clipboard(_event):
        header = "\t".join(name for name, _key in clipboard_columns)
        arrays = [np.asarray(profile[key], dtype=float) for _name, key in clipboard_columns]
        lines = [header]
        for values in zip(*arrays):
            lines.append("\t".join("nan" if not np.isfinite(value) else f"{value:.10g}"
                                   for value in values))
        clipboard_text = "\n".join(lines)
        root = tk._default_root
        if root is None:
            messagebox.showerror("Clipboard", "The Tk clipboard is unavailable.")
            return
        try:
            root.clipboard_clear()
            root.clipboard_append(clipboard_text)
            root.update()
            messagebox.showinfo(
                "Line profile",
                f"Copied {len(distance)} rows to the clipboard. Paste directly into Origin.",
            )
        except tk.TclError as exc:
            messagebox.showerror("Clipboard", str(exc))

    copy_ax = figure.add_axes([0.36, 0.055, 0.28, 0.075])
    copy_button = Button(copy_ax, "Copy data to clipboard")
    copy_button.on_clicked(copy_profile_to_clipboard)
    figure._copy_profile_button = copy_button
    figure._line_profile_data = profile
    plt.show(block=False)
    figure.canvas.draw_idle()


def show_bidirectional_comparison(results) -> None:
    """Display forward, backward, and half-difference normal-strain maps."""
    if str(np.asarray(results.get("scan_direction", ""))) != "both":
        return
    lx_nm, ly_nm = np.asarray(scan_size_xy) * 1e9
    extent = (0.0, lx_nm, ly_nm, 0.0)
    figure, axes = plt.subplots(
        2, 3, figsize=(13.5, 8), num="Forward/backward strain comparison",
        constrained_layout=True,
    )
    panels = (
        ("forward_exx", r"Forward $\varepsilon_{xx}$"),
        ("backward_exx", r"Backward $\varepsilon_{xx}$"),
        ("directional_exx", r"$(F-B)/2\;\varepsilon_{xx}$"),
        ("forward_eyy", r"Forward $\varepsilon_{yy}$"),
        ("backward_eyy", r"Backward $\varepsilon_{yy}$"),
        ("directional_eyy", r"$(F-B)/2\;\varepsilon_{yy}$"),
    )
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#bdbdbd")
    for axis, (key, title) in zip(axes.ravel(), panels):
        values = 100.0 * np.asarray(results[key], dtype=float)
        limits = robust_limit(values, percentile=99.0, symmetric=True)
        artist = axis.imshow(
            values, cmap=cmap, origin="upper", extent=extent,
            vmin=limits[0], vmax=limits[1],
        )
        axis.set_title(title + " (%)")
        axis.set_xlabel("x (nm)")
        axis.set_ylabel("y (nm)")
        figure.colorbar(artist, ax=axis, fraction=0.046, pad=0.03)
    figure.suptitle(
        "Common structure is retained by (F+B)/2; scan-direction artefacts appear in (F-B)/2"
    )
    plt.show(block=False)
    figure.canvas.draw_idle()


def show_v1_style_results(
    results, initial_sigma, sigma_max, validation_text, lattice_match=None
) -> None:
    """Show the four compact relative-strain panels used by strain_v1."""
    global fig_res, sigma_slider, last_results
    analysis_mode = str(np.asarray(results.get("analysis_mode", "gpa")))
    is_atomic = "atomic" in analysis_mode
    is_bidirectional = analysis_mode.startswith("bidirectional_")
    if fig_res is not None:
        plt.close(fig_res)
    lx_nm, ly_nm = np.asarray(scan_size_xy) * 1e9
    extent = (0.0, lx_nm, ly_nm, 0.0)
    fig_res = plt.figure(figsize=(15.5, 8.8), num="STM GPA relative strain")
    grid = fig_res.add_gridspec(
        2, 3, width_ratios=(1.0, 1.0, 0.82),
        left=0.055, right=0.975, bottom=0.16, top=0.89,
        hspace=0.30, wspace=0.22,
    )
    original_ax = fig_res.add_subplot(grid[0, 2])
    info_ax = fig_res.add_subplot(grid[1, 2])
    result_axes, colorbar_axes = [], []
    for row, col in ((0, 0), (0, 1), (1, 0), (1, 1)):
        # A fixed colorbar column keeps all four image panels exactly aligned.
        cell = grid[row, col].subgridspec(
            1, 2, width_ratios=(1.0, 0.038), wspace=0.025
        )
        result_axes.append(fig_res.add_subplot(cell[0, 0]))
        colorbar_axes.append(fig_res.add_subplot(cell[0, 1]))
    shown_image = np.asarray(results.get("display_image", img), dtype=float)
    original_image_artist = original_ax.imshow(
        shown_image, cmap=nanox_cmap, origin="upper", extent=extent
    )
    direction_label = "forward/backward mean" if is_bidirectional else active_scan_direction
    original_ax.set_title(f"STM {source_channel} {direction_label}\nDrag a line here")
    if str(np.asarray(results.get("reference_correction_mode", "none"))) != "none":
        points = np.asarray(results.get("reference_points_xy_nm", []), dtype=float)
        residuals = np.asarray(
            results.get("reference_patch_residual_strain", np.full((len(points), 3), np.nan)),
            dtype=float,
        )
        for index, point in enumerate(points):
            finite = np.abs(residuals[index][np.isfinite(residuals[index])])
            residual_percent = 100.0 * np.max(finite) if finite.size else np.inf
            color = ("lime" if residual_percent <= 0.15 else
                     "yellow" if residual_percent <= 0.40 else "red")
            original_ax.add_patch(
                Circle(point, float(results.get("reference_patch_radius_nm", 0.0)),
                       facecolor=color, edgecolor="cyan", linewidth=0.8, alpha=0.20)
            )
            original_ax.text(
                point[0], point[1], str(index + 1), color="white", fontsize=7,
                ha="center", va="center", fontweight="bold",
            )
    if analysis_mode == "atomic_registry":
        detected_xy = np.asarray(results["atomic_detected_xy_nm"], dtype=float)
        fitted_xy = np.asarray(results["atomic_fitted_xy_nm"], dtype=float)
        original_ax.scatter(
            detected_xy[:, 0], detected_xy[:, 1], s=5, marker=".",
            color="yellow", alpha=0.50,
        )
        original_ax.scatter(
            fitted_xy[:, 0], fitted_xy[:, 1], s=7, facecolors="none",
            edgecolors="cyan", linewidths=0.55, alpha=0.8,
        )
        original_ax.set_title(
            f"STM {source_channel}: yellow detected, cyan lattice-matched\nDrag a line here"
        )
    info_ax.set_axis_off()
    info_lines = [part.strip() for part in validation_text.split(";") if part.strip()]
    info_ax.text(
        0.04, 0.94, "Result information", transform=info_ax.transAxes,
        fontsize=12, fontweight="bold", va="top",
    )
    info_ax.text(
        0.04, 0.84, "\n".join(info_lines), transform=info_ax.transAxes,
        fontsize=10, va="top", ha="left", linespacing=1.15, wrap=True,
        bbox=dict(boxstyle="round,pad=0.7", facecolor="#f5f5f5", edgecolor="#b0b0b0"),
    )

    def percent(key, current):
        return 100.0 * np.asarray(current[key], dtype=float)

    displays = [
        ("exx", r"$\varepsilon_{xx}$ (%)"),
        ("eyy", r"$\varepsilon_{yy}$ (%)"),
        ("principal_max", r"Maximum principal $\varepsilon_1$ (%)"),
        ("mean_normal", r"Mean normal $\varepsilon_{mean}=(\varepsilon_{xx}+\varepsilon_{yy})/2$ (%)"),
    ]
    artists = {}
    strain_cmap = plt.get_cmap("RdBu_r").copy()
    strain_cmap.set_bad("#bdbdbd")
    for ax, cax, (key, title) in zip(result_axes, colorbar_axes, displays):
        data = percent(key, results)
        limits = robust_limit(data, percentile=99.0, symmetric=True)
        artist = ax.imshow(data, cmap=strain_cmap, origin="upper", extent=extent,
                           vmin=limits[0], vmax=limits[1])
        ax.set_title(title)
        fig_res.colorbar(artist, cax=cax)
        artists[key] = artist

    for ax in [original_ax, *result_axes]:
        ax.set_xlabel("x (nm)")
        ax.set_ylabel("y (nm)")

    title_method = ("Atomic-position strain" if is_atomic else "Local relative GPA strain")
    if is_bidirectional:
        title_method += " - forward/backward tensor average"
    fig_res.suptitle(title_method + " (scanner distortion corrected)", fontsize=14)
    slider_ax = fig_res.add_axes([0.08, 0.065, 0.20 if is_bidirectional else 0.30, 0.028])
    sigma_slider = Slider(
        slider_ax, "FFT sigma", 3.0, max(3.0, sigma_max),
        valinit=float(initial_sigma), valstep=0.5
    )
    if is_atomic or is_bidirectional or lattice_match is None:
        sigma_slider.set_active(False)
    comparison_button = None
    if is_bidirectional:
        comparison_ax = fig_res.add_axes([0.29, 0.045, 0.12, 0.060])
        comparison_button = Button(comparison_ax, "F/B comparison")
        comparison_button.on_clicked(lambda _event: show_bidirectional_comparison(results))
    view_axes = [original_ax, *result_axes]

    def apply_strain_display_limits(current, bounds=None):
        """Rescale strain colors/colorbars from the current visible physical ROI."""
        height, width = img.shape
        if bounds is None:
            y_slice, x_slice = slice(0, height), slice(0, width)
        else:
            x0, x1, y0, y1 = bounds
            ix0 = int(np.clip(np.floor(x0 / lx_nm * width), 0, width - 1))
            ix1 = int(np.clip(np.ceil(x1 / lx_nm * width), ix0 + 1, width))
            iy0 = int(np.clip(np.floor(y0 / ly_nm * height), 0, height - 1))
            iy1 = int(np.clip(np.ceil(y1 / ly_nm * height), iy0 + 1, height))
            y_slice, x_slice = slice(iy0, iy1), slice(ix0, ix1)
        for key, _title in displays:
            visible_data = percent(key, current)[y_slice, x_slice]
            artists[key].set_clim(
                *robust_limit(visible_data, percentile=99.0, symmetric=True)
            )

    def render_rotated_zoom(current, bounds, angle_deg):
        """Rotate the STM image and every strain panel about the box center, then crop to it."""
        x0, x1, y0, y1 = bounds
        center_xy_nm = (0.5 * (x0 + x1), 0.5 * (y0 + y1))
        half_extent_xy_nm = (0.5 * (x1 - x0), 0.5 * (y1 - y0))
        pixel_nm = (
            np.asarray(pixel_size_xy, dtype=float) * 1e9 if spatial_unit == "m"
            else np.array([1.0, 1.0])
        )
        out_w = int(np.clip(round(2.0 * half_extent_xy_nm[0] / pixel_nm[0]), 16, 2000))
        out_h = int(np.clip(round(2.0 * half_extent_xy_nm[1] / pixel_nm[1]), 16, 2000))
        new_extent = (0.0, 2.0 * half_extent_xy_nm[0], 2.0 * half_extent_xy_nm[1], 0.0)

        cropped_image = rotated_crop_sample(
            shown_image, center_xy_nm, half_extent_xy_nm, angle_deg, pixel_nm, (out_h, out_w)
        )
        original_image_artist.set_data(cropped_image)
        original_image_artist.set_extent(new_extent)

        for ax, (key, _title) in zip(result_axes, displays):
            data = percent(key, current)
            cropped = rotated_crop_sample(
                data, center_xy_nm, half_extent_xy_nm, angle_deg, pixel_nm, (out_h, out_w)
            )
            artists[key].set_data(cropped)
            artists[key].set_extent(new_extent)
            artists[key].set_clim(*robust_limit(cropped, percentile=99.0, symmetric=True))

        for axis in view_axes:
            axis.set_xlim(new_extent[0], new_extent[1])
            axis.set_ylim(new_extent[2], new_extent[3])

    def update(value):
        global last_results
        if is_atomic or is_bidirectional or lattice_match is None:
            return
        try:
            current = compute_v1_style_relative_strain(
                float(value), smooth_sigma=1.0, lattice_match=lattice_match
            )
            current = apply_strain_reference_correction(
                current, str(np.asarray(results.get("reference_correction_mode", "none")))
            )
        except Exception:
            return
        last_results = current
        if zoom_state.get("applied"):
            try:
                render_rotated_zoom(current, zoom_state["bounds"], zoom_state["angle_deg"])
            except Exception:
                pass
        else:
            for key, _title in displays:
                data = percent(key, current)
                artists[key].set_data(data)
            active_bounds = zoom_state["bounds"] if not zoom_state["active"] else None
            apply_strain_display_limits(current, active_bounds)
        fig_res.canvas.draw_idle()

    sigma_slider.on_changed(update)
    line_state = {"start": None, "end": None, "artist": None, "dragging": False}
    zoom_state = {"active": False, "bounds": None, "angle_deg": 0.0, "applied": False}
    zoom_preview_line, = original_ax.plot([], [], color="orange", linewidth=1.6, linestyle="--")

    def redraw_zoom_preview():
        if zoom_state["bounds"] is None or abs(zoom_state["angle_deg"]) < 1e-9:
            zoom_preview_line.set_data([], [])
        else:
            corners = rotated_zoom_corners_xy(zoom_state["bounds"], zoom_state["angle_deg"])
            zoom_preview_line.set_data(corners[:, 0], corners[:, 1])
        fig_res.canvas.draw_idle()

    def on_zoom_rectangle(eclick, erelease):
        if None in (eclick.xdata, eclick.ydata, erelease.xdata, erelease.ydata):
            return
        x0, x1 = sorted((float(eclick.xdata), float(erelease.xdata)))
        y0, y1 = sorted((float(eclick.ydata), float(erelease.ydata)))
        if (x1 - x0) < 0.01 * lx_nm or (y1 - y0) < 0.01 * ly_nm:
            zoom_state["bounds"] = None
            original_ax.set_title("Zoom box is too small; drag again")
        else:
            zoom_state["bounds"] = (x0, x1, y0, y1)
            original_ax.set_title("Zoom area ready - adjust angle, then click Confirm zoom")
        redraw_zoom_preview()

    zoom_selector = RectangleSelector(
        original_ax, on_zoom_rectangle, useblit=True, button=[1],
        minspanx=5, minspany=5, spancoords="pixels", interactive=True,
        props=dict(facecolor="yellow", edgecolor="yellow", alpha=0.22, fill=True),
    )
    zoom_selector.set_active(False)

    angle_ax = fig_res.add_axes([0.42, 0.010, 0.30, 0.022])
    zoom_angle_slider = Slider(
        angle_ax, "Zoom angle (deg)", -90.0, 90.0, valinit=0.0, valstep=1.0
    )

    def on_angle_change(value):
        zoom_state["angle_deg"] = float(value)
        redraw_zoom_preview()

    zoom_angle_slider.on_changed(on_angle_change)

    def start_zoom_selection(_event):
        zoom_state["active"] = True
        zoom_state["bounds"] = None
        line_state["dragging"] = False
        zoom_selector.set_visible(True)
        zoom_selector.set_active(True)
        original_ax.set_title("STM image\nDrag a rectangular zoom area")
        redraw_zoom_preview()

    def confirm_zoom(_event):
        bounds = zoom_state["bounds"]
        if bounds is None:
            messagebox.showwarning("Zoom area", "Select a rectangular area on the STM image first.")
            return
        angle_deg = float(zoom_state["angle_deg"])
        zoom_selector.set_active(False)
        zoom_selector.set_visible(False)
        zoom_state["active"] = False
        current = last_results if last_results is not None else results
        if abs(angle_deg) < 1e-9:
            x0, x1, y0, y1 = bounds
            zoom_state["applied"] = False
            for axis in view_axes:
                axis.set_xlim(x0, x1)
                axis.set_ylim(y1, y0)
            apply_strain_display_limits(current, bounds)
            original_ax.set_title("STM image\nZoomed view - drag a line for profile")
        else:
            try:
                render_rotated_zoom(current, bounds, angle_deg)
            except Exception as exc:
                messagebox.showerror("Zoom area", str(exc))
                return
            zoom_state["applied"] = True
            original_ax.set_title("STM image (rotated view)\nZoomed view - drag a line for profile")
        fig_res.canvas.draw_idle()

    def reset_full_view(_event):
        zoom_selector.set_active(False)
        zoom_selector.set_visible(False)
        zoom_state["active"] = False
        zoom_state["bounds"] = None
        zoom_state["applied"] = False
        zoom_preview_line.set_data([], [])
        current = last_results if last_results is not None else results
        original_image_artist.set_data(shown_image)
        original_image_artist.set_extent(extent)
        for key, _title in displays:
            data = percent(key, current)
            artists[key].set_data(data)
            artists[key].set_extent(extent)
        for axis in view_axes:
            axis.set_xlim(0.0, lx_nm)
            axis.set_ylim(ly_nm, 0.0)
        apply_strain_display_limits(current, None)
        original_ax.set_title("STM image\nDrag a line here")
        fig_res.canvas.draw_idle()

    def on_press(event):
        if zoom_state["active"] or event.inaxes is not original_ax or event.button != 1:
            return
        line_state["start"] = (float(event.xdata), float(event.ydata))
        line_state["end"] = line_state["start"]
        line_state["dragging"] = True
        if line_state["artist"] is not None:
            line_state["artist"].remove()
        line_state["artist"], = original_ax.plot(
            [event.xdata, event.xdata], [event.ydata, event.ydata],
            color="cyan", linewidth=2.0, marker="o", markersize=4
        )
        fig_res.canvas.draw_idle()

    def on_motion(event):
        if zoom_state["active"] or not line_state["dragging"] or event.inaxes is not original_ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        line_state["end"] = (float(event.xdata), float(event.ydata))
        x0, y0 = line_state["start"]
        line_state["artist"].set_data([x0, event.xdata], [y0, event.ydata])
        fig_res.canvas.draw_idle()

    def on_release(event):
        if zoom_state["active"] or line_state["start"] is None or event.button not in (None, 1):
            return
        line_state["dragging"] = False
        if event.inaxes is original_ax and event.xdata is not None and event.ydata is not None:
            line_state["end"] = (float(event.xdata), float(event.ydata))
        start = line_state["start"]
        end = line_state["end"]
        if np.hypot(end[0] - start[0], end[1] - start[1]) < 1e-6:
            original_ax.set_title("Line is too short; drag again")
        else:
            original_ax.set_title("STM image\nLine ready - click Show line profile")
        fig_res.canvas.draw_idle()

    def on_profile_button(_event):
        start, end = line_state["start"], line_state["end"]
        if start is None or end is None or np.hypot(end[0] - start[0], end[1] - start[1]) < 1e-6:
            messagebox.showwarning("Line profile", "Drag a line on the STM image first.")
            return
        current = last_results if last_results is not None else results
        show_line_profile(current, start, end)

    fig_res.canvas.mpl_connect("button_press_event", on_press)
    fig_res.canvas.mpl_connect("motion_notify_event", on_motion)
    fig_res.canvas.mpl_connect("button_release_event", on_release)
    select_zoom_ax = fig_res.add_axes([0.42, 0.045, 0.12, 0.060])
    confirm_zoom_ax = fig_res.add_axes([0.55, 0.045, 0.12, 0.060])
    full_view_ax = fig_res.add_axes([0.68, 0.045, 0.10, 0.060])
    button_ax = fig_res.add_axes([0.80, 0.045, 0.17, 0.060])
    select_zoom_button = Button(select_zoom_ax, "Select zoom area")
    confirm_zoom_button = Button(confirm_zoom_ax, "Confirm zoom")
    full_view_button = Button(full_view_ax, "Full view")
    profile_button = Button(button_ax, "Show line profile")
    select_zoom_button.on_clicked(start_zoom_selection)
    confirm_zoom_button.on_clicked(confirm_zoom)
    full_view_button.on_clicked(reset_full_view)
    profile_button.on_clicked(on_profile_button)
    # Keep widget/callback state alive for the lifetime of the result figure.
    fig_res._line_profile_button = profile_button
    fig_res._bidirectional_comparison_button = comparison_button
    fig_res._zoom_buttons = (select_zoom_button, confirm_zoom_button, full_view_button)
    fig_res._zoom_selector = zoom_selector
    fig_res._zoom_state = zoom_state
    fig_res._zoom_angle_slider = zoom_angle_slider
    fig_res._zoom_preview_line = zoom_preview_line
    fig_res._line_profile_state = line_state
    plt.show(block=False)
    # The scientific panels need screen space; maximize on the Tk/Windows
    # backend while retaining normal resize controls on other backends.
    try:
        fig_res.canvas.manager.window.state("zoomed")
    except Exception:
        pass
    fig_res.canvas.draw_idle()


def _calculate_active_direction(method: str) -> tuple[dict, dict, float, float, str]:
    """Calculate one independently calibrated direction using the selected method."""
    if img is None or fft_shifted is None or peaks_selected is None:
        raise RuntimeError(f"Complete the {active_scan_direction} FFT peak selection first.")
    if unit_cell_basis_xy is None:
        raise RuntimeError(f"Confirm the {active_scan_direction} FFT-derived unit cell first.")
    a_nm, b_nm, angle_deg = read_lattice_inputs()
    measured_G = measured_reciprocal_vectors_physical(img.shape, peaks_selected, pixel_size_xy)
    lattice_match = match_theoretical_lattice(measured_G, a_nm, b_nm, angle_deg)
    sigma_min, sigma_max = mask_sigma_bounds()
    initial_sigma = float(
        np.clip(recommended_sigma if recommended_sigma is not None else 4.0,
                sigma_min, sigma_max)
    )
    if method == "atomic":
        results = compute_atomic_registry_strain(lattice_match=lattice_match)
        method_text = "fitted atomic positions"
        mask_mode = "atomic-fit validity"
    else:
        results = compute_v1_style_relative_strain(
            initial_sigma, smooth_sigma=1.0, lattice_match=lattice_match
        )
        results["analysis_mode"] = np.asarray("gpa")
        method_text = "GPA phase"
        mask_mode = "auto" if bool(results["auto_mask_enabled"]) else "off"
    reference_mode = "none" if reference_correction_var is None else reference_correction_var.get()
    results = apply_strain_reference_correction(results, reference_mode)
    measured_lengths = lattice_match["measured_direct_lengths"] * 1e9
    correction = lattice_match["coordinate_correction"]
    text = (
        f"{active_scan_direction}: a={measured_lengths[0]:.4g}, "
        f"b={measured_lengths[1]:.4g} nm, "
        f"angle={lattice_match['measured_direct_angle_deg']:.1f} deg; "
        f"correction={correction[0, 0]:.4f}/{correction[1, 1]:.4f}, "
        f"shear={correction[0, 1]:+.4f}; method: {method_text}; mask: {mask_mode}; "
        f"valid pixels: {100*np.mean(results['valid_mask']):.1f}%; "
        f"reference: {reference_mode}"
    )
    if reference_mode != "none":
        residual_percent = 100.0 * np.asarray(
            results["reference_corrected_median_strain"], dtype=float
        )
        text += (
            f"; reference residual xx/yy/xy="
            f"{residual_percent[0]:+.3g}/{residual_percent[1]:+.3g}/"
            f"{residual_percent[2]:+.3g}%; max patch residual="
            f"{100.0 * float(results['reference_patch_max_abs_residual']):.3g}%"
        )
    if method == "atomic":
        text += (
            f"; detected/matched atoms: {len(results['atomic_detected_xy_nm'])}/"
            f"{len(results['atomic_fitted_xy_nm'])}; median match error: "
            f"{np.median(results['atomic_match_residual_pixel']):.2f} px"
        )
    results["scan_direction"] = np.asarray(active_scan_direction)
    results["lattice_validation_mismatch"] = np.asarray(lattice_match["match_score"])
    results["selected_fft_periods_m"] = lattice_match["measured_fft_periods"]
    results["measured_direct_lattice_m"] = lattice_match["measured_direct_lengths"]
    results["measured_direct_angle_deg"] = np.asarray(lattice_match["measured_direct_angle_deg"])
    results["unit_cell_origin_xy"] = np.asarray(unit_cell_origin_xy)
    results["unit_cell_basis_xy"] = np.asarray(unit_cell_basis_xy)
    results["unit_cell_vertices_xy"] = np.asarray(unit_cell_vertices_xy)
    results["unit_cell_snap_fraction"] = np.asarray(unit_cell_snap_fraction)
    if analysis_roi_full is not None:
        results["analysis_roi_full_y1_y2_x1_x2"] = np.asarray(analysis_roi_full, dtype=int)
    return results, lattice_match, initial_sigma, sigma_max, text


def combine_bidirectional_results(forward: dict, backward: dict, method: str) -> dict:
    """Average tensor components and retain half-difference scan diagnostics."""
    if analysis_roi_full is None:
        y1, y2, x1, x2 = 0, scan_images_full["forward"].shape[0], 0, scan_images_full["forward"].shape[1]
    else:
        y1, y2, x1, x2 = map(int, analysis_roi_full)
    valid = np.asarray(forward["valid_mask"], bool) & np.asarray(backward["valid_mask"], bool)
    components = {}
    directional = {}
    for key in ("exx", "eyy", "exy"):
        f_value = np.asarray(forward[key], float)
        b_value = np.asarray(backward[key], float)
        valid &= np.isfinite(f_value) & np.isfinite(b_value)
        components[key] = 0.5 * (f_value + b_value)
        directional[key] = 0.5 * (f_value - b_value)
    components = {key: np.where(valid, value, np.nan) for key, value in components.items()}
    directional = {key: np.where(valid, value, np.nan) for key, value in directional.items()}
    invariants = strain_invariants_from_components(
        components["exx"], components["eyy"], components["exy"]
    )
    direction_invariants = strain_invariants_from_components(
        directional["exx"], directional["eyy"], directional["exy"]
    )
    combined = {
        **components,
        **invariants,
        "valid_mask": valid,
        "outlier_mask": np.asarray(forward["outlier_mask"], bool)
                        | np.asarray(backward["outlier_mask"], bool),
        "phase_jump_mask": np.asarray(forward["phase_jump_mask"], bool)
                           | np.asarray(backward["phase_jump_mask"], bool),
        "analysis_mode": np.asarray(f"bidirectional_{method}"),
        "scan_direction": np.asarray("both"),
        "auto_mask_enabled": np.asarray(
            bool(forward.get("auto_mask_enabled", False))
            or bool(backward.get("auto_mask_enabled", False))
        ),
        "reference_correction_mode": np.asarray(
            forward.get("reference_correction_mode", "none")
        ),
        "reference_patch_mask": np.asarray(
            forward.get("reference_patch_mask", np.zeros(valid.shape, bool)), bool
        ),
        "reference_points_xy_nm": np.asarray(strain_reference_points_xy_nm, dtype=float),
        "reference_patch_radius_nm": np.asarray(strain_reference_radius_nm),
        "display_image": 0.5 * (
            np.asarray(scan_images_full["forward"])[y1:y2, x1:x2]
            + np.asarray(scan_images_full["backward"])[y1:y2, x1:x2]
        ),
    }
    for key, value in components.items():
        combined["raw_" + key] = value
    for key, value in directional.items():
        combined["directional_" + key] = value
    for key, value in direction_invariants.items():
        combined["directional_" + key] = value
    for prefix, source in (("forward_", forward), ("backward_", backward)):
        for key, value in source.items():
            combined[prefix + key] = value
    if "reference_patch_residual_strain" in forward and "reference_patch_residual_strain" in backward:
        combined_residuals = 0.5 * (
            np.asarray(forward["reference_patch_residual_strain"], dtype=float)
            + np.asarray(backward["reference_patch_residual_strain"], dtype=float)
        )
        combined["reference_patch_residual_strain"] = combined_residuals
        finite = np.abs(combined_residuals[np.isfinite(combined_residuals)])
        combined["reference_patch_max_abs_residual"] = np.asarray(
            np.max(finite) if finite.size else np.nan
        )
    return combined


def run_gpa() -> None:
    """Run the explicitly selected method for one or both calibrated traces."""
    global last_results
    try:
        if spatial_unit != "m":
            raise RuntimeError("The SXM scan-range calibration is required.")
        save_active_direction_state()
        directions = selected_scan_directions()
        method = "gpa" if analysis_method_var is None else str(analysis_method_var.get()).lower()
        calculations = {}
        matches = {}
        texts = []
        sigmas, maxima = [], []
        for direction in directions:
            if not direction_states.get(direction, {}).get("calibrated", False):
                raise RuntimeError(
                    f"Complete manual FFT peak and unit-cell calibration for {direction} first."
                )
            activate_scan_direction(direction)
            result, match, sigma, sigma_max, text = _calculate_active_direction(method)
            calculations[direction] = result
            matches[direction] = match
            sigmas.append(sigma)
            maxima.append(sigma_max)
            texts.append(text)
        if len(directions) == 2:
            results = combine_bidirectional_results(
                calculations["forward"], calculations["backward"], method
            )
            validation_text = (
                f"Analysis channel: {source_channel}; bidirectional tensor average; "
                f"common valid pixels: {100*np.mean(results['valid_mask']):.1f}%; "
                + "; ".join(texts)
            )
            lattice_match = None
        else:
            results = calculations[directions[0]]
            validation_text = f"Analysis channel: {source_channel}; " + texts[0]
            lattice_match = matches[directions[0]]
        last_results = results
        show_v1_style_results(
            results, float(np.mean(sigmas)), float(np.max(maxima)),
            validation_text, lattice_match=lattice_match
        )
    except Exception as exc:
        messagebox.showerror("Strain calculation failed", str(exc))


def select_profile_center() -> None:
    """Select a center manually and plot annular median/IQR strain statistics."""
    if last_results is None:
        messagebox.showwarning("No results", "Run GPA before selecting a profile center.")
        return
    results = last_results
    magnitude = masked_percent(results, "magnitude")
    fig, ax = plt.subplots(figsize=(8, 7), num="Select annular-profile center")
    finite = magnitude[np.isfinite(magnitude)]
    upper = float(np.percentile(finite, 98.0)) if finite.size else 1.0
    image_artist = ax.imshow(magnitude, cmap="inferno", origin="upper", vmin=0, vmax=upper)
    fig.colorbar(image_artist, ax=ax, label="Strain magnitude (%)")
    ax.set_title("Left-click the profile center (manual selection)")

    def on_click(event):
        if event.inaxes is not ax or event.button != 1:
            return
        center = (float(event.xdata), float(event.ydata))
        ix = int(np.clip(round(center[0]), 0, magnitude.shape[1] - 1))
        iy = int(np.clip(round(center[1]), 0, magnitude.shape[0] - 1))
        if not results["valid_mask"][iy, ix]:
            ax.set_title("Selected center is invalid; choose a confidence-qualified pixel")
            fig.canvas.draw_idle()
            return
        stats = annular_profile_statistics(
            magnitude,
            results["valid_mask"],
            center,
            results["pixel_size_xy"],
        )
        results["profile_center_xy_pixel"] = stats["center_xy_pixel"]
        results["profile_radius_spatial_unit"] = stats["radius"]
        results["profile_median_percent"] = stats["median"]
        results["profile_q25_percent"] = stats["q25"]
        results["profile_q75_percent"] = stats["q75"]
        results["profile_count"] = stats["count"]
        plt.close(fig)

        radius = stats["radius"].copy()
        xlabel = "Radius (pixel)"
        if spatial_unit == "m":
            radius *= 1e9
            xlabel = "Radius (nm)"
        profile_fig, profile_ax = plt.subplots(figsize=(8, 5), num="Annular strain statistics")
        profile_ax.fill_between(
            radius, stats["q25"], stats["q75"], color="tab:blue", alpha=0.25, label="IQR"
        )
        profile_ax.plot(radius, stats["median"], color="tab:blue", label="Annular median")
        profile_ax.set_xlabel(xlabel)
        profile_ax.set_ylabel(r"$\max(|\varepsilon_1|, |\varepsilon_2|)$ (%)")
        profile_ax.set_title(
            f"Manual center: x={center[0]:.1f}, y={center[1]:.1f} pixels"
        )
        profile_ax.grid(alpha=0.25)
        profile_ax.legend()
        profile_fig.tight_layout()
        plt.show(block=False)
        profile_fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.tight_layout()
    plt.show(block=False)
    fig.canvas.draw_idle()


def show_diagnostics() -> None:
    """Show advanced, separately inspectable GPA processing stages."""
    if last_results is None or fft_shifted is None:
        messagebox.showwarning("No diagnostics", "Run GPA before opening diagnostics.")
        return

    results = last_results
    fig, axes = plt.subplots(4, 4, figsize=(19, 17), num="STM GPA - advanced diagnostics")

    def show_map(ax, data, title, cmap="viridis", symmetric=False, colorbar=True, interpolation="nearest"):
        data = np.asarray(data)
        limits = robust_limit(data, symmetric=symmetric)
        if not symmetric and np.any(np.isfinite(data)) and np.nanmin(data) >= 0:
            limits = (0.0, limits[1])
        artist = ax.imshow(
            data, cmap=cmap, origin="upper", vmin=limits[0], vmax=limits[1],
            interpolation=interpolation,
        )
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        if colorbar:
            fig.colorbar(artist, ax=ax, fraction=0.046, pad=0.03)
        return artist

    fft_view = fft_display_array(fft_shifted)
    show_map(axes[0, 0], fft_view, "FFT with Gaussian-mask contours", gwyddion_cmap, interpolation="bilinear")
    axes[0, 0].contour(results["fft_mask1"], levels=[np.exp(-0.5)], colors="cyan", linewidths=1)
    axes[0, 0].contour(results["fft_mask2"], levels=[np.exp(-0.5)], colors="lime", linewidths=1)
    show_map(axes[0, 1], results["fft_mask1"], "FFT Gaussian mask 1", "magma")
    show_map(axes[0, 2], results["fft_mask2"], "FFT Gaussian mask 2", "magma")
    show_map(axes[0, 3], results["confidence"], "Dual-amplitude confidence", "viridis")

    show_map(axes[1, 0], results["phase1"], "Validated reference phase 1 (rad)", "RdBu_r", True)
    show_map(axes[1, 1], results["phase2"], "Validated reference phase 2 (rad)", "RdBu_r", True)
    show_map(axes[1, 2], results["amplitude1"], "Bragg-field amplitude 1", "inferno")
    show_map(axes[1, 3], results["amplitude2"], "Bragg-field amplitude 2", "inferno")

    show_map(axes[2, 0], results["phase_jump_mask"], "Dilated phase-jump mask", "gray", colorbar=False)
    show_map(axes[2, 1], results["phase_valid_mask"], "Valid before derivative erosion", "gray", colorbar=False)
    show_map(axes[2, 2], results["valid_mask"], "Final validated mask", "gray", colorbar=False)
    show_map(axes[2, 3], results["outlier_mask"], "Physically implausible outliers", "Reds", colorbar=False)

    displacement_scale = 1e9 if spatial_unit == "m" else 1.0
    displacement_unit = "nm" if spatial_unit == "m" else "pixel"
    show_map(
        axes[3, 0], results["ux"] * displacement_scale,
        f"Filtered ux ({displacement_unit})", "RdBu_r", True
    )
    show_map(
        axes[3, 1], results["uy"] * displacement_scale,
        f"Filtered uy ({displacement_unit})", "RdBu_r", True
    )
    show_map(axes[3, 2], 100.0 * results["raw_magnitude"], "Raw strain magnitude (%)", "inferno")
    show_map(axes[3, 3], 100.0 * results["magnitude"], "Validated strain magnitude (%)", "inferno")

    warning_text = quality_status_text(results, full=True)
    fig.suptitle("Advanced GPA diagnostics", fontsize=15)
    fig.text(0.01, 0.005, warning_text, fontsize=8,
             color="darkred" if len(results["quality_warnings"]) else "darkgreen")
    fig.tight_layout(rect=(0, 0.035, 1, 0.97))
    plt.show(block=False)
    fig.canvas.draw_idle()


def export_results() -> None:
    if last_results is None:
        messagebox.showwarning("没有结果", "请先运行 GPA。")
        return
    initial = "strain_results.npz" if source_path is None else source_path.stem + "_strain.npz"
    path = filedialog.asksaveasfilename(
        title="保存 GPA 数值结果",
        defaultextension=".npz",
        initialfile=initial,
        filetypes=[("NumPy compressed data", "*.npz")],
    )
    if not path:
        return
    np.savez_compressed(
        path,
        image=np.asarray(last_results.get("display_image", img)),
        image_forward=scan_images_full.get("forward", np.asarray([])),
        image_backward_aligned=scan_images_full.get("backward", np.asarray([])),
        strain_values_are_dimensionless_fractions=np.asarray(True),
        source_spatial_unit=np.asarray(spatial_unit),
        source_analysis_channel=np.asarray(source_channel),
        selected_analysis_method=np.asarray(
            "gpa" if analysis_method_var is None else analysis_method_var.get()
        ),
        selected_scan_direction=np.asarray(
            "forward" if scan_direction_var is None else scan_direction_var.get()
        ),
        backward_x_was_flipped=np.asarray(backward_x_flipped),
        **last_results,
    )
    messagebox.showinfo("保存完成", path)


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------
def build_gui() -> tk.Tk:
    global lattice_a_var, lattice_b_var, lattice_angle_var
    global scan_info_var, auto_mask_var, analysis_channel_var
    global analysis_method_var, scan_direction_var, reference_correction_var
    root = tk.Tk()
    root.title("STM GPA Analyzer - simple lattice mode")

    lattice_a_var = tk.StringVar(value="0.8")
    lattice_b_var = tk.StringVar(value="0.693")
    lattice_angle_var = tk.StringVar(value="90")
    scan_info_var = tk.StringVar(value="No SXM loaded")
    auto_mask_var = tk.BooleanVar(value=False)
    analysis_channel_var = tk.StringVar(value="Z")
    analysis_method_var = tk.StringVar(value="atomic")
    scan_direction_var = tk.StringVar(value="forward")
    reference_correction_var = tk.StringVar(value="drift")

    tk.Label(root, text="Analysis channel").grid(row=0, column=0, padx=3, pady=3)
    tk.Radiobutton(
        root, text="Current", variable=analysis_channel_var, value="Current"
    ).grid(row=0, column=1, padx=2)
    tk.Radiobutton(
        root, text="Z", variable=analysis_channel_var, value="Z"
    ).grid(row=0, column=2, padx=2)
    tk.Label(root, text="Analysis method").grid(row=1, column=0, padx=3, pady=3)
    tk.Radiobutton(
        root, text="GPA", variable=analysis_method_var, value="gpa"
    ).grid(row=1, column=1, padx=2)
    tk.Radiobutton(
        root, text="Atomic", variable=analysis_method_var, value="atomic"
    ).grid(row=1, column=2, padx=2)

    def direction_changed() -> None:
        if not scan_images_full:
            return
        requested = selected_scan_directions()
        target = requested[0]
        try:
            activate_scan_direction(target)
            show_image()
        except Exception as exc:
            messagebox.showerror("Scan direction", str(exc))

    tk.Label(root, text="Scan direction").grid(row=2, column=0, padx=3, pady=3)
    tk.Radiobutton(
        root, text="Forward", variable=scan_direction_var, value="forward",
        command=direction_changed,
    ).grid(row=2, column=1, padx=2)
    tk.Radiobutton(
        root, text="Backward", variable=scan_direction_var, value="backward",
        command=direction_changed,
    ).grid(row=2, column=2, padx=2)
    tk.Radiobutton(
        root, text="Both", variable=scan_direction_var, value="both",
        command=direction_changed,
    ).grid(row=2, column=3, padx=2)

    tk.Label(root, text="Reference correction").grid(row=3, column=0, padx=3, pady=3)
    tk.Radiobutton(
        root, text="None", variable=reference_correction_var, value="none"
    ).grid(row=3, column=1, padx=2)
    tk.Radiobutton(
        root, text="Offset", variable=reference_correction_var, value="offset"
    ).grid(row=3, column=2, padx=2)
    tk.Radiobutton(
        root, text="Drift", variable=reference_correction_var, value="drift"
    ).grid(row=3, column=3, padx=2)

    tk.Button(root, text="1. Load SXM", command=load_file, width=24).grid(
        row=4, column=0, columnspan=4, sticky="ew", padx=5, pady=5
    )

    tk.Label(root, text="a (nm)").grid(row=5, column=0, padx=3, pady=3)
    tk.Entry(root, textvariable=lattice_a_var, width=9).grid(row=5, column=1, padx=3)
    tk.Label(root, text="b (nm)").grid(row=5, column=2, padx=3)
    tk.Entry(root, textvariable=lattice_b_var, width=9).grid(row=5, column=3, padx=3)
    tk.Label(root, text="angle (deg)").grid(row=6, column=0, columnspan=2, padx=3, pady=3)
    tk.Entry(root, textvariable=lattice_angle_var, width=9).grid(row=6, column=2, padx=3)

    tk.Label(root, text="FFT zoom").grid(row=7, column=0, padx=3, pady=3)
    tk.Button(root, text="1x", command=lambda: set_zoom(1)).grid(row=7, column=1, padx=2)
    tk.Button(root, text="4x", command=lambda: set_zoom(4)).grid(row=7, column=2, padx=2)
    tk.Button(root, text="16x", command=lambda: set_zoom(16)).grid(row=7, column=3, padx=2)
    tk.Checkbutton(
        root,
        text="Auto mask",
        variable=auto_mask_var,
        command=refresh_auto_mask,
        indicatoron=False,
        width=22,
        selectcolor="#a8ddb5",
        relief=tk.RAISED,
        offrelief=tk.RAISED,
    ).grid(row=8, column=0, columnspan=4, padx=5, pady=4, sticky="ew")
    tk.Button(root, text="2. Select analysis region", command=select_analysis_region).grid(
        row=9, column=0, columnspan=4, sticky="ew", padx=5, pady=4
    )
    tk.Button(root, text="3. Pick two Bragg peaks", command=pick_g_vectors).grid(
        row=10, column=0, columnspan=2, sticky="ew", padx=5, pady=5
    )
    tk.Button(root, text="3'. Pick two Bragg peaks (fitted)", command=pick_g_vectors_fitted).grid(
        row=10, column=2, columnspan=2, sticky="ew", padx=5, pady=5
    )
    tk.Button(root, text="4. Align FFT unit cell", command=select_unit_cell_frame).grid(
        row=11, column=0, columnspan=4, sticky="ew", padx=5, pady=5
    )
    tk.Button(root, text="5. Select strain-free patches", command=select_strain_reference_patches).grid(
        row=12, column=0, columnspan=4, sticky="ew", padx=5, pady=5
    )
    tk.Button(root, text="6. Calculate selected strain", command=run_gpa).grid(
        row=13, column=0, columnspan=4, sticky="ew", padx=5, pady=5
    )
    tk.Label(root, textvariable=scan_info_var, fg="#204060").grid(
        row=14, column=0, columnspan=4, padx=5, pady=3
    )
    tk.Label(
        root,
        text=(
            "For Both, calibrate Forward first; the program then switches to Backward. "
            "Both directions use the same ROI and reference patches. Drift mode fits a "
            "compatible quadratic displacement background separately for each direction "
            "before tensor-component averaging."
        ),
        fg="#444444",
        wraplength=390,
    ).grid(row=15, column=0, columnspan=4, padx=6, pady=6)

    def close_application() -> None:
        global app_closing, reference_selection_fig, unit_cell_selection_fig
        global analysis_region_fig, strain_reference_selection_fig, fig_res
        app_closing = True
        reference_selection_fig = None
        unit_cell_selection_fig = None
        analysis_region_fig = None
        strain_reference_selection_fig = None
        fig_res = None
        plt.close("all")
        try:
            root.quit()
            root.destroy()
        except tk.TclError:
            pass

    root.protocol("WM_DELETE_WINDOW", close_application)
    return root


def main() -> None:
    global app_closing
    app_closing = False
    root = build_gui()
    root.mainloop()


if __name__ == "__main__":
    main()
