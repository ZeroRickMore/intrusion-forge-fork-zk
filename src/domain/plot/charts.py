import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from .base import (
    Plot,
    _apply_labels,
    _ensure_ax,
    _fig_to_plot,
    _finalize,
    _format_value,
    _smart_legend_loc,
)
from .style import (
    HIGHLIGHT_COLOR,
    MUTED_COLOR,
    NEUTRAL_COLOR,
    PALETTE,
    extended_palette,
)


def bar_plot(
    labels: np.ndarray | list[str],
    values: np.ndarray | list[float],
    *,
    orientation: str = "h",
    color: str | list[str] | None = None,
    sort: str | None = "asc",
    top_k: int | None = None,
    annotate_values: bool = True,
    value_format: str = "{:.3f}",
    color_gradient: bool = False,
    x_label: str = "",
    y_label: str = "",
    title: str = "",
    ylim: tuple[float, float] | None = None,
    xlim: tuple[float, float] | None = None,
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
    bar_positions: np.ndarray | None = None,
    bar_alpha: float = 1.0,
    axvline: float | None = None,
    axvline_color: str = "black",
    hide_yticks: bool = False,
    hide_left_spine: bool = False,
) -> Plot | None:
    """Bar chart with optional sorting, top-k filtering, and value annotations.

    `color` may be a single color (applied to all bars), a list of per-bar
    colors, or None (defaults to PALETTE[0]). `color_gradient=True` is
    ignored when `color` is a list.
    `bar_positions`: explicit numeric positions for `barh` — use when the
    y-axis is shared with another panel (e.g. `sharey=True`).
    """
    if orientation not in ("h", "v"):
        raise ValueError("`orientation` must be 'h' or 'v'.")
    if sort not in ("asc", "desc", None):
        raise ValueError("`sort` must be 'asc', 'desc', or None.")

    items = list(zip(list(labels), list(values)))
    if sort == "asc":
        items.sort(key=lambda kv: kv[1])
        items = items[-top_k:] if top_k is not None else items
    elif sort == "desc":
        items.sort(key=lambda kv: kv[1], reverse=True)
        items = items[:top_k] if top_k is not None else items
    elif top_k is not None:
        items = items[:top_k]

    plot_labels = [str(name) for name, _ in items]
    plot_values = np.array([v for _, v in items], dtype=float)
    n = len(plot_labels)

    if isinstance(color, list):
        bar_color = list(color)
    else:
        base = color if color is not None else PALETTE[0]
        if color_gradient and n > 0:
            base_rgb = np.array(mcolors.to_rgb(base))
            white = np.ones(3)
            ts = 0.25 + 0.75 * np.arange(n) / max(n - 1, 1)
            bar_color = [mcolors.to_hex(t * base_rgb + (1 - t) * white) for t in ts]
        else:
            bar_color = base

    if figsize is None:
        figsize = (
            (12, max(6, n * 0.4 + 1.5))
            if orientation == "h"
            else (max(8, n * 0.5 + 1.5), 6)
        )

    ax, fig = _ensure_ax(ax, figsize)

    positions = bar_positions if bar_positions is not None else plot_labels
    draw = ax.barh if orientation == "h" else ax.bar
    draw(positions, plot_values, color=bar_color, alpha=bar_alpha, edgecolor="white", linewidth=0.5)

    if orientation == "h":
        ax.set_xlabel(x_label or "Value")
        if y_label:
            ax.set_ylabel(y_label)
        ax.grid(True, axis="x")
        ax.grid(False, axis="y")
    else:
        if x_label:
            ax.set_xlabel(x_label)
        ax.set_ylabel(y_label or "Value")
        ax.grid(True, axis="y")
        ax.grid(False, axis="x")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    if annotate_values and n:
        value_range = plot_values.max() - plot_values.min() or 1.0
        offset = max(value_range * 0.01, 1e-6)
        for i, v in enumerate(plot_values):
            text = value_format.format(v)
            pos = float(bar_positions[i]) if bar_positions is not None else i
            if orientation == "h":
                ax.text(v + offset, pos, text, va="center", ha="left")
            else:
                ax.text(pos, v + offset, text, ha="center", va="bottom")

    if ylim is not None:
        ax.set_ylim(ylim)
    if xlim is not None:
        ax.set_xlim(xlim)
    if title:
        ax.set_title(title)
    if axvline is not None:
        ax.axvline(axvline, color=axvline_color, linewidth=1.0, zorder=5)
    if hide_yticks:
        ax.tick_params(axis="y", left=False, labelleft=False)
    if hide_left_spine:
        ax.spines["left"].set_visible(False)

    return _finalize(fig)


def violin_plot(
    categories: np.ndarray,
    values: np.ndarray,
    *,
    category_order: list | None = None,
    split: bool = False,
    inner: str = "box",
    colors: tuple[str, ...] | None = None,
    violin_alpha: float = 0.55,
    x_label: str = "",
    y_label: str = "",
    title: str = "",
    show_legend: bool = True,
    figsize: tuple[float, float] = (5, 4),
    ax: Axes | None = None,
) -> Plot | None:
    """Violin plot with optional split layout and inner box/quartile indicators."""
    if inner not in ("box", "quartiles", "none"):
        raise ValueError("`inner` must be 'box', 'quartiles', or 'none'.")

    categories = np.asarray(categories)
    values = np.asarray(values)

    unique_cats = (
        list(category_order)
        if category_order is not None
        else list(dict.fromkeys(categories.tolist()))
    )
    n_cats = len(unique_cats)

    if split and n_cats != 2:
        raise ValueError("split=True requires exactly 2 categories.")

    cat_colors = list(colors) if colors is not None else extended_palette(n_cats)
    if len(cat_colors) < n_cats:
        raise ValueError("`colors` must have at least one entry per category.")

    ax, fig = _ensure_ax(ax, figsize)
    legend_handles = []

    for idx, (cat, color) in enumerate(zip(unique_cats, cat_colors)):
        mask = categories == cat
        vals = values[mask]
        if len(vals) < 2 or np.unique(vals).size < 2:
            continue

        position = 0 if split else idx
        parts = ax.violinplot(
            vals,
            positions=[position],
            showmedians=False,
            showextrema=False,
            widths=0.85,
        )
        body = parts["bodies"][0]
        body.set_facecolor(color)
        body.set_alpha(violin_alpha)
        body.set_edgecolor(color)
        body.set_linewidth(1.0)

        if split and body.get_paths():
            verts = body.get_paths()[0].vertices
            clip = np.minimum if idx == 0 else np.maximum
            verts[:, 0] = clip(verts[:, 0], position)

        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        whisker_low = float(np.min(vals))
        whisker_high = float(np.max(vals))

        if inner == "box":
            box_half = 0.05
            ax.add_patch(
                plt.Rectangle(
                    (position - box_half, q1),
                    box_half * 2,
                    q3 - q1,
                    facecolor="white",
                    edgecolor=MUTED_COLOR,
                    linewidth=1.0,
                    zorder=4,
                )
            )
            ax.plot(
                [position - box_half, position + box_half],
                [med, med],
                color=MUTED_COLOR,
                linewidth=1.8,
                zorder=5,
            )
            for y0, y1 in ((whisker_low, q1), (q3, whisker_high)):
                ax.plot(
                    [position, position],
                    [y0, y1],
                    color=MUTED_COLOR,
                    linewidth=0.8,
                    zorder=4,
                )
        elif inner == "quartiles":
            sign = 0 if not split else (-1 if idx == 0 else 1)
            ax.plot(
                [position, position + sign * 0.06],
                [med, med],
                color=color,
                lw=2.0,
                zorder=4,
            )
            ax.vlines(position + sign * 0.04, q1, q3, color=color, lw=1.5, zorder=4)

        legend_handles.append(
            plt.Line2D([], [], color=color, lw=6, alpha=violin_alpha, label=str(cat))
        )

    if split:
        ax.axvline(0, color="#aaaaaa", lw=0.8, zorder=2)
        ax.set_xticks([])
    else:
        ax.set_xticks(range(n_cats))
        ax.set_xticklabels([str(c) for c in unique_cats])

    if show_legend and legend_handles:
        ax.legend(handles=legend_handles, loc="best")

    _apply_labels(ax, x_label, y_label, title)
    return _finalize(fig)


def strip_plot(
    categories: np.ndarray,
    values: np.ndarray,
    fill_categorical_colors: tuple[str, ...] = (),
    *,
    fill_values: np.ndarray | None = None,
    fill_cmap: str | None = None,
    marker_values: np.ndarray | None = None,
    marker_shapes: tuple[str, ...] = ("o", "X"),
    category_order: list | None = None,
    orientation: str = "v",
    show_median: bool = True,
    x_label: str = "",
    y_label: str = "",
    title: str = "",
    marker_size: float = 36.0,
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
) -> Plot | None:
    """Strip plot with categorical fill and optional per-point marker encoding.

    `fill_cmap`: when set, `fill_values` is treated as continuous floats mapped
    through this colormap instead of used as categorical indices.
    """
    if orientation not in ("v", "h"):
        raise ValueError("`orientation` must be 'v' or 'h'.")

    categories = np.asarray(categories)
    values = np.asarray(values, dtype=float)

    if category_order is None:
        category_order = list(dict.fromkeys(categories.tolist()))
    n_cats = len(category_order)
    cat_to_pos = {cat: i for i, cat in enumerate(category_order)}

    fill_arr = (
        np.asarray(fill_values, dtype=float) if fill_values is not None else values
    )
    finite = np.isfinite(fill_arr)
    if fill_cmap is not None:
        cmap_fn = plt.get_cmap(fill_cmap)
        lo = float(np.nanmin(fill_arr)) if finite.any() else 0.0
        hi = float(np.nanmax(fill_arr)) if finite.any() else 1.0
        span = hi - lo if hi > lo else 1.0
        normed = np.where(finite, (fill_arr - lo) / span, 0.5)
        point_colors = np.array([cmap_fn(float(v)) for v in normed], dtype=float)
        point_colors[~finite] = [0.75, 0.75, 0.75, 0.85]
    else:
        fill_idx = np.clip(
            np.where(finite, fill_arr.astype(int), 0),
            0,
            len(fill_categorical_colors) - 1,
        )
        point_colors = np.array(
            [mcolors.to_rgba(fill_categorical_colors[i]) for i in fill_idx], dtype=float
        )
        point_colors[~finite] = [0.75, 0.75, 0.75, 0.85]

    if marker_values is not None:
        marker_arr = np.asarray(marker_values, dtype=float)
        finite_m = np.isfinite(marker_arr)
        marker_idx = np.clip(
            np.where(finite_m, marker_arr, 0.0).astype(int),
            0,
            len(marker_shapes) - 1,
        )
        per_point_marker = np.array([marker_shapes[i] for i in marker_idx])
    else:
        per_point_marker = None

    rng = np.random.default_rng(seed=42)
    base_positions = np.array([cat_to_pos[c] for c in categories], dtype=float)
    positions = base_positions + rng.uniform(-0.25, 0.25, size=len(categories))

    if figsize is None:
        figsize = (
            (11, max(6.0, 0.35 * n_cats + 2.0))
            if orientation == "h"
            else (max(8.0, 0.6 * n_cats + 2.0), 7.0)
        )

    ax, fig = _ensure_ax(ax, figsize)

    def _xy(pos: np.ndarray, val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (pos, val) if orientation == "v" else (val, pos)

    scatter_kwargs = dict(
        s=marker_size,
        edgecolors="white",
        linewidths=0.4,
        zorder=3,
        alpha=0.85,
    )

    if per_point_marker is None:
        x, y = _xy(positions, values)
        ax.scatter(x, y, c=point_colors, **scatter_kwargs)
    else:
        for shape in np.unique(per_point_marker):
            mask = per_point_marker == shape
            if not mask.any():
                continue
            x, y = _xy(positions[mask], values[mask])
            ax.scatter(x, y, c=point_colors[mask], marker=shape, **scatter_kwargs)

    if show_median:
        for cat in category_order:
            mask = categories == cat
            if not mask.any():
                continue
            pos = cat_to_pos[cat]
            med = float(np.median(values[mask]))
            along = (pos - 0.3, pos + 0.3)
            across = (med, med)
            x, y = (along, across) if orientation == "v" else (across, along)
            ax.plot(x, y, color=MUTED_COLOR, linewidth=1.5, zorder=4)

    cat_labels = [str(c) for c in category_order]
    if orientation == "v":
        ax.set_xticks(range(n_cats))
        ax.set_xticklabels(cat_labels)
        plt.setp(ax.get_xticklabels(), rotation=90, ha="center")
        ax.grid(False, axis="x")
        ax.grid(True, axis="y")
    else:
        ax.set_yticks(range(n_cats))
        ax.set_yticklabels(cat_labels)
        ax.invert_yaxis()
        ax.grid(True, axis="x")
        ax.grid(False, axis="y")

    _apply_labels(ax, x_label, y_label, title)
    return _finalize(fig)


def strip_count_panel_plot(
    categories: np.ndarray,
    values: np.ndarray,
    category_order: list[str],
    counts_by_class: dict[str, int],
    fill_values: np.ndarray,
    fill_categorical_colors: tuple[str, ...],
    x_label: str,
    *,
    fill_cmap: str | None = None,
    fill_cmap_label: str = "",
    marker_values: np.ndarray | None = None,
    marker_shapes: tuple[str, ...] = ("o", "X"),
    failed_counts_by_class: dict[str, int] | None = None,
) -> Plot:
    """Strip plot (left) + horizontal count bar (right) sharing the y-axis.

    When `failed_counts_by_class` is provided, a red overlay bar per class
    shows how many clusters have failure_rate > 0, and a vertical reference
    line is drawn at x=0.
    """
    n_cats = len(category_order)
    height = max(3.0, 0.35 * n_cats + 1.5)
    fig, (ax_left, ax_right) = plt.subplots(
        1,
        2,
        gridspec_kw={"width_ratios": [3.5, 1.0], "wspace": 0.04},
        figsize=(11, height),
        sharey=True,
    )

    strip_plot(
        categories=categories,
        values=values,
        fill_values=fill_values,
        fill_categorical_colors=fill_categorical_colors,
        fill_cmap=fill_cmap,
        marker_values=marker_values,
        marker_shapes=marker_shapes,
        category_order=category_order,
        orientation="h",
        show_median=True,
        x_label=x_label,
        y_label="Class",
        ax=ax_left,
    )

    if fill_cmap is not None:
        fill_arr = np.asarray(fill_values, dtype=float)
        finite = np.isfinite(fill_arr)
        lo = float(np.nanmin(fill_arr[finite])) if finite.any() else 0.0
        hi = float(np.nanmax(fill_arr[finite])) if finite.any() else 1.0
        norm = mcolors.Normalize(vmin=lo, vmax=hi)
        sm = plt.cm.ScalarMappable(cmap=fill_cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax_left, fraction=0.025, pad=0.02, shrink=0.8)
        cbar.set_label(fill_cmap_label, fontsize=8)

    counts = [int(counts_by_class.get(cat, 0)) for cat in category_order]
    y_positions = np.arange(n_cats)
    max_count = max(counts) if counts else 1

    bar_plot(
        labels=list(category_order),
        values=counts,
        orientation="h",
        sort=None,
        bar_positions=y_positions,
        bar_alpha=0.55,
        color=MUTED_COLOR,
        annotate_values=True,
        value_format="{:.0f}",
        x_label="n clusters",
        hide_yticks=True,
        hide_left_spine=True,
        xlim=(0, max_count * 1.18),
        axvline=0 if failed_counts_by_class is not None else None,
        ax=ax_right,
    )

    if failed_counts_by_class is not None:
        failed_counts = [int(failed_counts_by_class.get(cat, 0)) for cat in category_order]
        if any(failed_counts):
            bar_plot(
                labels=list(category_order),
                values=failed_counts,
                orientation="h",
                sort=None,
                bar_positions=y_positions,
                bar_alpha=0.85,
                color=HIGHLIGHT_COLOR,
                annotate_values=False,
                x_label="n clusters",
                hide_yticks=True,
                hide_left_spine=True,
                ax=ax_right,
            )

    return _fig_to_plot(fig)


def scatter_plot(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    noise_mask: np.ndarray | None = None,
    highlight_mask: np.ndarray | None = None,
    names: dict | None = None,
    palette: str | list[str] = "extended",
    marker_size: float = 14.0,
    marker_alpha: float | None = None,
    minority_fraction: float = 0.05,
    x_label: str = "Dim 1",
    y_label: str = "Dim 2",
    title: str = "",
    show_legend: bool = True,
    legend_max_items: int = 20,
    legend_on_top: bool = False,
    figsize: tuple[float, float] = (6.5, 5.0),
    ax: Axes | None = None,
) -> Plot | None:
    """Scatter 2D with label-coloring and optional highlight mask.

    `legend_on_top` renders an opaque legend above the points (high z-order),
    trading the occluded points for a cleaner figure.
    """
    X = np.asarray(X)
    labels = np.asarray(labels)
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("`X` must have shape (n, 2).")

    noise = (
        np.asarray(noise_mask, dtype=bool)
        if noise_mask is not None
        else np.zeros(len(X), dtype=bool)
    )
    highlight = (
        np.asarray(highlight_mask, dtype=bool)
        if highlight_mask is not None
        else np.zeros(len(X), dtype=bool)
    )

    unique_labels = sorted(int(l) for l in np.unique(labels[~noise]))
    n_labels = max(len(unique_labels), 1)
    if isinstance(palette, list):
        color_list = list(palette)
        while len(color_list) < n_labels:
            color_list.extend(palette)
        color_list = color_list[:n_labels]
    else:
        color_list = extended_palette(n_labels)
    color_map = {lbl: color_list[i] for i, lbl in enumerate(unique_labels)}

    ax, fig = _ensure_ax(ax, figsize)
    ax.set_aspect("equal", adjustable="datalim")

    if noise.any():
        ax.scatter(
            X[noise, 0],
            X[noise, 1],
            c=MUTED_COLOR,
            marker=".",
            s=marker_size * 0.45,
            alpha=0.4,
            linewidths=0,
            zorder=1,
        )

    counts = {lbl: int(((labels == lbl) & ~noise).sum()) for lbl in unique_labels}
    total_visible = max(sum(counts.values()), 1)
    draw_order = sorted(unique_labels, key=lambda l: counts[l], reverse=True)

    for lbl in draw_order:
        base = (labels == lbl) & ~noise
        if not base.any():
            continue
        color = color_map[lbl]
        n_class = counts[lbl]
        is_minority = n_class / total_visible < minority_fraction

        alpha_val = (
            float(np.clip(2.0 / np.sqrt(max(n_class, 1)), 0.25, 0.85))
            if marker_alpha is None
            else float(marker_alpha)
        )
        size_val = marker_size * (1.4 if is_minority else 1.0)
        if is_minority:
            alpha_val = min(alpha_val + 0.15, 0.85)

        normal = base & ~highlight
        hot = base & highlight
        if normal.any():
            ax.scatter(
                X[normal, 0],
                X[normal, 1],
                c=[color],
                s=size_val,
                alpha=alpha_val,
                marker="o",
                edgecolors="none",
                zorder=3 if is_minority else 2,
                rasterized=True,
            )
        if hot.any():
            ax.scatter(
                X[hot, 0],
                X[hot, 1],
                c=[color],
                s=size_val,
                alpha=min(alpha_val + 0.25, 0.95),
                marker="o",
                edgecolors=HIGHLIGHT_COLOR,
                linewidths=0.9,
                zorder=10,
                rasterized=True,
            )

    if show_legend and names is not None:
        if len(unique_labels) > legend_max_items:
            ax.text(
                0.99,
                0.02,
                f"{len(unique_labels)} groups",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=9,
                color=MUTED_COLOR,
            )
        else:
            handles = [
                plt.Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="",
                    markersize=8,
                    markerfacecolor=color_map[lbl],
                    markeredgecolor="#444444",
                    markeredgewidth=0.5,
                    label=names.get(int(lbl), str(lbl)),
                )
                for lbl in unique_labels
            ]
            if highlight_mask is not None:
                handles.append(
                    plt.Line2D(
                        [],
                        [],
                        marker="o",
                        linestyle="",
                        markersize=8,
                        markerfacecolor=NEUTRAL_COLOR,
                        markeredgecolor=HIGHLIGHT_COLOR,
                        markeredgewidth=0.9,
                        label="misclassified",
                    )
                )
            ncol = 1 if len(handles) <= 6 else 2 if len(handles) <= 12 else 3
            loc = _smart_legend_loc(ax, X[~noise]) if (~noise).any() else "best"
            legend = ax.legend(
                handles=handles,
                loc=loc,
                ncol=ncol,
                framealpha=1.0 if legend_on_top else None,
            )
            if legend_on_top:
                legend.set_zorder(100)

    _apply_labels(ax, x_label, y_label, title)
    return _finalize(fig)


def numeric_scatter_plot(
    x: np.ndarray,
    y: np.ndarray,
    *,
    color_values: np.ndarray | None = None,
    reference_line: bool = False,
    trend_line: bool = False,
    annotations: dict[str, float] | None = None,
    cmap: str = "viridis",
    colorbar_label: str = "",
    x_label: str = "",
    y_label: str = "",
    title: str = "",
    figsize: tuple[float, float] = (5.5, 5.5),
    ax: Axes | None = None,
) -> Plot | None:
    """Generic numeric x–y scatter, with optional continuous coloring, a y=x
    reference line, a linear trend line, and a metrics annotation box."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ax, fig = _ensure_ax(ax, figsize)

    if reference_line and len(x):
        lo = float(min(x.min(), y.min()))
        hi = float(max(x.max(), y.max()))
        ax.plot([lo, hi], [lo, hi], color=NEUTRAL_COLOR, linewidth=1.0, linestyle=":")

    sc = ax.scatter(
        x,
        y,
        c=color_values if color_values is not None else PALETTE[0],
        cmap=cmap if color_values is not None else None,
        s=22,
        alpha=0.7,
        edgecolors="none",
    )
    if color_values is not None and fig is not None:
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(colorbar_label)

    if trend_line:
        finite = np.isfinite(x) & np.isfinite(y)
        if int(finite.sum()) >= 2:
            m, b = np.polyfit(x[finite], y[finite], 1)
            x_range = np.array([float(x[finite].min()), float(x[finite].max())])
            ax.plot(x_range, m * x_range + b, color=NEUTRAL_COLOR, linewidth=1.0, linestyle="--", zorder=2)

    if annotations:
        text = "\n".join(
            f"{name} = {_format_value(value, kind='score')}"
            for name, value in annotations.items()
        )
        ax.text(
            0.05,
            0.95,
            text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox=dict(facecolor="white", edgecolor="#cccccc", boxstyle="round,pad=0.3"),
        )

    ax.grid(True, alpha=0.15, linewidth=0.5)
    _apply_labels(ax, x_label, y_label, title)
    return _finalize(fig)


def line_plot(
    series: dict[str, list[float] | np.ndarray],
    *,
    x_label: str = "step",
    y_label: str = "value",
    title: str = "",
    colors: list[str] | None = None,
    linewidth: float = 1.2,
    show_legend: bool = True,
    figsize: tuple[float, float] = (8.0, 4.0),
    ax: Axes | None = None,
) -> Plot | None:
    """Multi-series line plot (e.g. training history curves).

    `series` is a mapping `{name: values}`. Each series is drawn as a line; the
    x axis is the step index for every series. `colors` defaults to the
    extended palette in series-key order.
    """
    ax, fig = _ensure_ax(ax, figsize)
    keys = list(series.keys())
    palette = colors if colors is not None else extended_palette(max(len(keys), 1))

    for i, (name, values) in enumerate(series.items()):
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            continue
        ax.plot(np.arange(arr.size), arr, color=palette[i], linewidth=linewidth, label=name)

    _apply_labels(ax, x_label=x_label, y_label=y_label, title=title)
    ax.grid(True, axis="both", alpha=0.15, linewidth=0.5)
    if show_legend and len(keys) > 1:
        ax.legend(loc="best", fontsize=8)

    return _finalize(fig)


def selective_accuracy_plot(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    baseline: float,
    annotations: dict[str, float] | None = None,
    x_label: str = "Fraction rejected (riskiest first)",
    y_label: str = "Accuracy on retained set",
    title: str = "Selective accuracy",
    figsize: tuple[float, float] = (6.0, 5.0),
    ax: Axes | None = None,
) -> Plot | None:
    """Selective-prediction curves with an explicit per-series x axis.

    `curves` maps a strategy name to its `(x, y)` arrays, where x is the fraction
    of test traffic rejected (riskiest regions first) and y the retained-set
    metric (accuracy or macro-recall) — so the curve rises as more is excluded.
    Each strategy rejects in a different order, so the x grids differ (unlike
    `line_plot`'s shared index x). A dashed horizontal line marks the Random
    baseline at `baseline`; `annotations` renders a scalar box.
    """
    ax, fig = _ensure_ax(ax, figsize)
    palette = extended_palette(max(len(curves), 1))

    for i, (name, (x, y)) in enumerate(curves.items()):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.size == 0:
            continue
        ax.plot(x, y, color=palette[i], linewidth=1.6, label=name)

    ax.axhline(
        baseline,
        color=NEUTRAL_COLOR,
        linewidth=1.0,
        linestyle="--",
        label="Random",
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    if annotations:
        text = "\n".join(
            f"{name} = {_format_value(value, kind='score')}"
            for name, value in annotations.items()
        )
        ax.text(
            0.95,
            0.05,
            text,
            transform=ax.transAxes,
            va="bottom",
            ha="right",
            fontsize=9,
            bbox=dict(facecolor="white", edgecolor="#cccccc", boxstyle="round,pad=0.3"),
        )

    ax.grid(True, alpha=0.15, linewidth=0.5)
    ax.legend(loc="best", fontsize=8)
    _apply_labels(ax, x_label, y_label, title)
    return _finalize(fig)


def beeswarm_plot(
    shap_values: np.ndarray,
    feature_values: np.ndarray,
    feature_names: list[str],
    *,
    max_display: int = 20,
    title: str = "",
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,  # noqa: ARG001 — ignored: SHAP creates its own figure
) -> Plot | None:
    """Beeswarm of per-sample SHAP values rendered via the native SHAP library.

    Delegates to `shap.plots.beeswarm()` (requires a SHAP `Explanation` object
    constructed from the raw arrays) and wraps the result in a `Plot`.
    """
    import shap as shap_lib

    shap_values = np.asarray(shap_values, dtype=float)
    feature_values = np.asarray(feature_values, dtype=float)
    if shap_values.ndim != 2:
        raise ValueError("`shap_values` must have shape (n_samples, n_features).")

    exp = shap_lib.Explanation(
        values=shap_values,
        data=feature_values,
        feature_names=list(feature_names),
    )
    returned_ax = shap_lib.plots.beeswarm(exp, max_display=max_display, show=False)
    fig = returned_ax.get_figure()
    if title:
        returned_ax.set_title(title, fontsize=10)
    if figsize is not None:
        fig.set_size_inches(figsize)
    return _fig_to_plot(fig)


def _fit_label(d: str, fit: dict) -> str:
    """One-line `α = mean±std, R² = r2` annotation for a distance's cost fit."""
    parts = [f"{d}: α = {fit['alpha_mean']:.3f}"]
    if fit.get("alpha_std"):
        parts.append(f" ± {fit['alpha_std']:.3f}")
    if fit.get("r2_min") is not None:
        parts.append(f", R² = {fit['r2_min']:.4f}")
    return "".join(parts)


def _m_label(m: float) -> str:
    """Compact label for a sample count, e.g. 50000 -> '50k'."""
    return f"{m / 1000:.0f}k" if m >= 1000 else f"{m:.0f}"


def cost_model_plot(
    points_by_distance: dict[str, dict],
    fits_by_distance: dict[str, dict],
    share_by_distance: dict[str, dict],
    *,
    m_prod: float | None = None,
    reference_slope: float = 2.0,
    figsize: tuple[float, float] = (10.0, 4.0),
) -> Plot:
    """k-NN graph build-cost figure for the paper.

    Panel (a) — log-log scaling: per distance, the measured medians (seed mean with
    min–max whiskers) and the fitted power law T = c·m^alpha, plus a slope-
    `reference_slope` Theta(m^2) guide anchored to the lowest series. When `m_prod`
    is given, a vertical marker shows the production operating point (the subsample
    cap the pipeline actually runs at). Panel (b) — how the end-to-end runtime splits
    at that operating point, one 100%-stacked bar per distance (complexity vs rest).

    `points_by_distance[d]` = {m, build_mean, build_min, build_max} (arrays over the
    m-grid); `fits_by_distance[d]` = {alpha_mean, alpha_std, c_mean, r2_min};
    `share_by_distance[d]` = {complexity_s, non_complexity_s, share}. Distances are
    drawn in sorted order so colours are stable; a single distance degrades cleanly.
    """
    distances = sorted(points_by_distance)
    colors = {d: PALETTE[i % len(PALETTE)] for i, d in enumerate(distances)}
    fig, (ax_scale, ax_share) = plt.subplots(1, 2, figsize=figsize)

    # (a) log-log scaling -----------------------------------------------------
    anchor: tuple[float, float] | None = None  # lowest first point, for the guide
    all_m: list[float] = []
    for d in distances:
        pts = points_by_distance[d]
        m = np.asarray(pts["m"], dtype=float)
        mean = np.asarray(pts["build_mean"], dtype=float)
        lo = np.asarray(pts["build_min"], dtype=float)
        hi = np.asarray(pts["build_max"], dtype=float)
        color = colors[d]
        all_m.extend(m.tolist())
        ax_scale.errorbar(
            m, mean, yerr=[mean - lo, hi - mean],
            fmt="o", ms=5, color=color, ecolor=color, elinewidth=0.8,
            capsize=2, zorder=3, label=d,
        )
        fit = fits_by_distance.get(d, {})
        c, alpha = fit.get("c_mean"), fit.get("alpha_mean")
        if c is not None and alpha is not None and m.size >= 2:
            xx = np.geomspace(m.min(), m.max(), 64)
            ax_scale.plot(xx, c * xx**alpha, color=color, linewidth=1.4, zorder=2)
        if m.size and (anchor is None or mean[0] < anchor[1]):
            anchor = (float(m[0]), float(mean[0]))

    if anchor is not None and len(all_m) >= 2 and min(all_m) < max(all_m):
        xx = np.geomspace(min(all_m), max(all_m), 32)
        m0, t0 = anchor
        ax_scale.plot(
            xx, t0 * (xx / m0) ** reference_slope,
            color=NEUTRAL_COLOR, linewidth=1.0, linestyle="--", zorder=1,
            label=f"slope {reference_slope:g} (Θ(m²))",
        )

    ax_scale.set_xscale("log")
    ax_scale.set_yscale("log")

    if m_prod:
        ax_scale.axvline(m_prod, color=MUTED_COLOR, linewidth=1.1, linestyle=":", zorder=1)
        ax_scale.text(
            m_prod, 0.97, f"production cap  m ≈ {_m_label(m_prod)} ",
            transform=ax_scale.get_xaxis_transform(),
            rotation=90, va="top", ha="right", fontsize=7.5, color=MUTED_COLOR,
        )

    text = "\n".join(_fit_label(d, fits_by_distance[d]) for d in distances if d in fits_by_distance)
    if text:
        ax_scale.text(
            0.05, 0.95, text, transform=ax_scale.transAxes, va="top", ha="left",
            fontsize=8, bbox=dict(facecolor="white", edgecolor="#cccccc", boxstyle="round,pad=0.3"),
        )
    _apply_labels(ax_scale, "training subsample m", "k-NN build time (s)", "(a) build-cost scaling")
    ax_scale.grid(True, which="both", alpha=0.15, linewidth=0.5)
    ax_scale.legend(loc="lower right", fontsize=8)

    # (b) runtime breakdown at the production operating point -----------------
    for i, d in enumerate(distances):
        sh = share_by_distance.get(d, {})
        comp = float(sh.get("complexity_s", 0.0))
        rest = float(sh.get("non_complexity_s", 0.0))
        total = comp + rest
        frac = comp / total if total else 0.0
        ax_share.barh(i, frac, color=colors[d], zorder=3, label=f"{d} (complexity)")
        ax_share.barh(
            i, 1.0 - frac, left=frac, color=NEUTRAL_COLOR, zorder=2,
            label="rest of pipeline" if i == 0 else None,
        )
        ax_share.text(min(frac + 0.02, 0.9), i, f"{frac * 100:.0f}%", va="center", ha="left", fontsize=9)

    ax_share.set_yticks(np.arange(len(distances)))
    ax_share.set_yticklabels(distances)
    ax_share.set_xlim(0, 1)
    ax_share.grid(True, axis="x", alpha=0.15, linewidth=0.5)
    cap = f" at production cap (m ≈ {_m_label(m_prod)})" if m_prod else ""
    _apply_labels(
        ax_share, "share of end-to-end runtime", "",
        f"(b) where the time goes{cap}",
    )
    if distances:
        ax_share.legend(loc="lower right", fontsize=8)

    return _fig_to_plot(fig)
