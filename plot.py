import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.colors import Normalize, LogNorm, LinearSegmentedColormap

arr = npt.NDArray[np.generic]


def log_fit(
    raw_x: arr | list[int | float],
    raw_y: arr | list[int | float],
    start: float | None = None,
    end: float | None = None,
    offset: float = 0.0,
) -> tuple[arr, arr, float, float]:
    """
    log-log scale linear fitting of (raw_x, raw_y).

    Return
    fit_x: x-coordinate of two points of fitted line
    fit_y: y-coordinate of two points of fitted line
    slope: slope of the line in log-log scale
    residual: fitting error
    """
    raw_x = (
        np.array(raw_x, dtype=np.float64)
        if isinstance(raw_x, list)
        else raw_x.astype(np.float64)
    )
    raw_y = (
        np.array(raw_y, dtype=np.float64)
        if isinstance(raw_y, list)
        else raw_y.astype(np.float64)
    )
    if start is None:
        start = raw_x.min()
    if end is None:
        end = raw_x.max()

    poly, residual, _, _, _ = np.polyfit(
        np.log10(raw_x), np.log10(raw_y), 1, full=True)
    fit_x = np.array([start, end], dtype=np.float64)
    fit_y = pow(10.0, poly[1] - offset) * np.power(fit_x, poly[0])
    return fit_x, fit_y, poly[0], residual[0]


def lin_log_fit(
    raw_x: arr | list[int | float],
    raw_y: arr | list[int | float],
    start: float | None = None,
    end: float | None = None,
    offset: float = 0.0,
) -> tuple[arr, arr, float, float]:
    """
    lin-log scale linear fitting of (raw_x, raw_y).

    Return
    fit_x: x-coordinate of two points of fitted line
    fit_y: y-coordinate of two points of fitted line
    slope: slope of the line in log-log scale
    residual: fitting error
    """
    raw_x = (
        np.array(raw_x, dtype=np.float64)
        if isinstance(raw_x, list)
        else raw_x.astype(np.float64)
    )
    raw_y = (
        np.array(raw_y, dtype=np.float64)
        if isinstance(raw_y, list)
        else raw_y.astype(np.float64)
    )
    if start is None:
        start = raw_x.min()
    if end is None:
        end = raw_x.max()
    poly, residual, _, _, _ = np.polyfit(raw_x, np.log10(raw_y), 1, full=True)
    fit_x = np.array([start, end], dtype=np.float64)
    fit_y = np.power(10.0, poly[0] * fit_x + poly[1] - offset)
    return fit_x, fit_y, poly[0], residual[0]


def log_lin_fit(
    raw_x: arr | list[int | float],
    raw_y: arr | list[int | float],
    start: float | None = None,
    end: float | None = None,
    offset: float = 0.0,
) -> tuple[arr, arr, float, float]:
    """
    log-lin scale linear fitting of (raw_x, raw_y).

    Return
    fit_x: x-coordinate of two points of fitted line
    fit_y: y-coordinate of two points of fitted line
    slope: slope of the line in log-log scale
    residual: fitting error
    """
    raw_x = (
        np.array(raw_x, dtype=np.float64)
        if isinstance(raw_x, list)
        else raw_x.astype(np.float64)
    )
    raw_y = (
        np.array(raw_y, dtype=np.float64)
        if isinstance(raw_y, list)
        else raw_y.astype(np.float64)
    )
    if start is None:
        start = raw_x.min()
    if end is None:
        end = raw_x.max()
    poly, residual, _, _, _ = np.polyfit(np.log10(raw_x), raw_y, 1, full=True)
    fit_x = np.array([start, end], dtype=np.float64)
    fit_y = poly[0] * np.log10(fit_x) + poly[1] - offset
    return fit_x, fit_y, poly[0], residual[0]


def lin_fit(
    raw_x: arr | list[int | float],
    raw_y: arr | list[int | float],
    start: float | None = None,
    end: float | None = None,
    offset: float = 0.0,
) -> tuple[arr, arr, float, float]:
    """
    lin-lin scale linear fitting of (raw_x, raw_y).

    Return
    fit_x: x-coordinate of two points of fitted line
    fit_y: y-coordinate of two points of fitted line
    slope: slope of the line in log-log scale
    residual: fitting error
    """
    raw_x = (
        np.array(raw_x, dtype=np.float64)
        if isinstance(raw_x, list)
        else raw_x.astype(np.float64)
    )
    raw_y = (
        np.array(raw_y, dtype=np.float64)
        if isinstance(raw_y, list)
        else raw_y.astype(np.float64)
    )
    if start is None:
        start = raw_x.min()
    if end is None:
        end = raw_x.max()
    poly, residual, _, _, _ = np.polyfit(raw_x, raw_y, 1, full=True)
    fit_x = np.array([start, end], dtype=np.float64)
    fit_y = poly[0] * fit_x + poly[1] - offset
    return fit_x, fit_y, poly[0], residual[0]


def log_log_line(
    x0: float, y0: float, slope: float, x1: float, ax: Axes | None = None, **kwargs
) -> float:
    """
    Draw line at log-log plot with passing (x0, y0), with slope.
    Another end point is x1
    """
    y1 = np.power(x1 / x0, slope) * y0
    if ax:
        ax.plot([x0, x1], [y0, y1], **kwargs)

    return x1


def get_colormap(
    cvals: list[float],
    colors: list[str],
    mode: str = "linear",
) -> tuple[LinearSegmentedColormap, Normalize]:

    if mode == "linear":
        tuples = list(
            zip(map(Normalize(min(cvals), max(cvals)), cvals), colors))
        return LinearSegmentedColormap.from_list("", tuples), Normalize(min(cvals), max(cvals))

    elif mode == "log":
        tuples = list(
            zip(map(LogNorm(min(cvals), max(cvals)), cvals), colors))
        return LinearSegmentedColormap.from_list("", tuples), LogNorm(min(cvals), max(cvals))

    raise ValueError("mode must be either 'linear'(default) or 'log'")
