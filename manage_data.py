from __future__ import annotations

import pickle
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Self

import numpy as np
import numpy.typing as npt
import json
import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.axes import Axes

from AOUP import Parameter


def get_conditions(
    N_particle: int | None = None,
    N_ensemble: int | None = None,
    velocity: float | None = None,
    Lambda: float | None = None,
    boundary: float | None = None,
    N_bins: int | None = None,
    gamma: float | None = None,
    slope: float | None = None,
    temperature: float | None = None,
    tau: float | None = None,
    Da: float | None = None,
    delta_t: float | None = None,
    initial: int | None = None,
    sampling: int | None = None,
) -> list[str]:

    conditions: list[str] = []

    if N_particle is not None:
        conditions.append(f"N_particle == {N_particle}")
    if N_ensemble is not None:
        conditions.append(f"N_ensemble == {N_ensemble}")
    if velocity is not None:
        conditions.append(f"velocity == {velocity}")
    if Lambda is not None:
        conditions.append(f"Lambda == {Lambda}")
    if boundary is not None:
        conditions.append(f"boundary == {boundary}")
    if N_bins is not None:
        conditions.append(f"N_bins = {N_bins}")
    if gamma is not None:
        conditions.append(f"gamma == {gamma}")
    if slope is not None:
        conditions.append(f"slope == {slope}")
    if temperature is not None:
        conditions.append(f"temperature == {temperature}")
    if tau is not None:
        conditions.append(f"tau == {tau}")
    if Da is not None:
        conditions.append(f"Da == {Da}")
    if delta_t is not None:
        conditions.append(f"delta_t == {delta_t}")
    if initial is not None:
        conditions.append(f"initial == {initial}")
    if sampling is not None:
        conditions.append(f"sampling == {sampling}")

    return conditions


def filter_file(f: Path, suffix: str) -> bool:
    return f.is_file() and (f.suffix == suffix) and f.stat().st_size > 0


def get_setting(
    conditions: list[str],
    location: Path = Path("."),
) -> Any:

    # * Scan the setting directory and gather result files
    setting_dir = location / f"setting"
    setting_files = [f for f in setting_dir.iterdir()
                     if filter_file(f, ".json")]

    # * Read files
    settings: list[dict[str, Any]] = []
    for file in setting_files:
        with open(file, "rb") as f:
            settings.append(json.load(f))

    df = pd.DataFrame(settings)

    if len(conditions) == 0:
        return df["key"]

    else:
        return df.query(" and ".join(conditions))["key"]


def load_result(
    conditions: list[str],
    location: Path = Path("."),
) -> pd.DataFrame:

    # * Scan the result directory and gather result files
    result_dir = location / f"data"
    # result_keys = get_setting(location=location, conditions=conditions)
    # result_files = [result_dir /
    #                 f"{result_key}.pkl" for result_key in result_keys]
    result_files = [f for f in result_dir.iterdir() if filter_file(f, ".pkl")]

    # * Read files
    results: list[dict[str, Any]] = []
    for file in result_files:
        if file.is_file():
            with open(file, "rb") as f:
                results.append(pickle.load(f))

    # * Concatenate to single dataframe
    df = pd.DataFrame(results)

    if len(conditions) == 0:
        return df.sort_values(by=["velocity", "Lambda", "slope"], ascending=True)

    else:
        return df.query(" and ".join(conditions)).sort_values(by=["velocity", "Lambda", "slope"], ascending=True)

    # return df.sort_values(by=["Lambda", "slope"], ascending=True)


def get_drag_by_velocity(
    df: pd.DataFrame,
    velocity: float,
) -> npt.NDArray:

    available_slope, available_Lambda = sorted(
        set(df["slope"].to_numpy())), sorted(set(df["Lambda"].to_numpy()))

    drags = np.zeros((len(available_slope), len(available_Lambda)))

    for i, slope in enumerate(available_slope):
        for j, Lambda in enumerate(available_Lambda):
            filtered_df = df.query(" and ".join(get_conditions(
                slope=slope, Lambda=Lambda, velocity=velocity)))

            drag = filtered_df["drag"].to_numpy()
            N_ensemble = filtered_df["N_ensemble"].to_numpy()
            sampling = filtered_df["sampling"].to_numpy()

            if len(drag) == 0:
                continue
            else:
                assert np.abs(
                    drag * N_ensemble) >= 1.0, f"drag greater or equal to 1 expected, got: {np.abs(drag * N_ensemble)}"

                drags[i, j] = drag * N_ensemble

    return drags


def get_std_by_velocity(
    df: pd.DataFrame,
    velocity: float,
) -> npt.NDArray:

    available_slope, available_Lambda = sorted(
        set(df["slope"].to_numpy())), sorted(set(df["Lambda"].to_numpy()))

    stds = np.zeros((len(available_slope), len(available_Lambda)))

    for i, slope in enumerate(available_slope):
        for j, Lambda in enumerate(available_Lambda):
            filtered_df = df.query(" and ".join(get_conditions(
                slope=slope, Lambda=Lambda, velocity=velocity)))

            std = filtered_df["std"].to_numpy()
            N_ensemble = filtered_df["N_ensemble"].to_numpy()
            sampling = filtered_df["sampling"].to_numpy()

            if len(std) == 0:
                continue
            else:
                stds[i, j] = std * N_ensemble

    return stds


def get_log_scale(
    df: pd.DataFrame,
    velocity: float,
) -> npt.NDArray:

    drag = get_drag_by_velocity(df, velocity)

    pm_flag = drag/np.abs(drag)
    data = np.log10(np.abs(drag))
    data *= pm_flag
    data[np.isnan(data)] = 0.0

    return data


def delete_result(
    key_names: list[str],
    location: Path = Path("."),
) -> None:

    del_setting, del_data = 0, 0

    for key in key_names:
        target_setting = location / f"setting/{key}.json"
        target_file = location / f"data/{key}.pkl"

        # try:
        #     target_setting.unlink()
        #     del_setting += 1
        # except OSError:
        #     print(f"No setting found for key in setting: {key}")

        try:
            target_file.unlink()
            del_data += 1
        except OSError:
            print(f"No file found for key in data: {key}")

    print(f"setting deleted: {del_setting}, data deleted: {del_data}")


def delete_all(
    location: Path = Path("."),
) -> None:
    setting_dir = location / "setting"
    data_dir = location / "data"

    # settings = [f for f in setting_dir.iterdir()]
    datas = [f for f in data_dir.iterdir()]

    del_setting, del_data = 0, 0
    # for setting in settings:
    #     try:
    #         setting.unlink()
    #         del_setting += 1
    #     except OSError:
    #         print(f"No setting found for key in setting: {setting}")
    for data in datas:
        try:
            data.unlink()
            del_data += 1
        except OSError:
            print(f"No file found for key in data: {data}")

    print(f"setting deleted: {del_setting}, data deleted: {del_data}")
