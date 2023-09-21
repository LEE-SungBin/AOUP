import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from dataclasses import dataclass, field, asdict
from datetime import datetime
import pandas as pd
from pathlib import Path
# * from typing import Self only available from python 3.11
from typing import Self
from typing import Any
import hashlib
import pickle
import argparse
import time
from numba import njit
import itertools


@dataclass
class num_Params:
    N_particle: int
    N_ensemble: int
    velocity: float
    Lambda: float
    boundary: float
    N_bins: int
    gamma: float
    slope: float
    temperature: float
    tau: float
    Da: float
    delta_t: float
    initial: int
    sampling: int
    interval: int
    degree: int

    def __post__init__(self) -> None:
        assert self.degree >= 1, f"order must be greater or equal to 1, order = {self.degree}"

    def to_log(self) -> str:
        return ", ".join(
            f"{key}={np.round(log, 3)}" for key, log in zip(asdict(self).keys(), asdict(self).values())
        )


@dataclass
class non_num_Params:
    activity: bool
    description: str

    # def __post__init__(self) -> None:

    def to_log(self) -> str:
        return ", ".join(
            f"{key}={log}" for key, log in zip(asdict(self).keys(), asdict(self).values())
        )


def get_logspace(
    max_value: float,
    num: int,
) -> npt.NDArray:

    logspace: list[float] = [max_value]

    for i in range(int(np.round((num-1)/2, 0))):
        logspace.append(max_value*0.3/10**i)
        logspace.append(max_value*0.1/10**i)

    return np.array(logspace)[::-1]


def get_linspace(
    max_value: float,
    num: int,
) -> npt.NDArray:

    linspace: list[float] = []

    for i in range(num):
        linspace.append(max_value/num*(i+1))

    return np.array(linspace)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    from run_AOUP import AOUP

    parser.add_argument("-N", "--N_particle", type=int, default=1)
    parser.add_argument("-ens", "--N_ensemble", type=int, default=1000)
    parser.add_argument("-mode", "--mode", type=str,
                        default="manual", choices=["manual", "velocity", "Lambda", "slope"])
    parser.add_argument("-v", "--velocity", type=float, default=1.0)
    parser.add_argument("-d", "--Lambda", type=float, default=1.0)
    parser.add_argument("-f", "--slope", type=float, default=1.0)
    parser.add_argument("-max_v", "--max_velocity", type=float, default=1.0)
    parser.add_argument("-N_v", "--N_velocity", type=int, default=7)
    parser.add_argument("-max_d", "--max_Lambda", type=float, default=1.0)
    parser.add_argument("-N_d", "--N_Lambda", type=int, default=7)
    parser.add_argument("-max_f", "--max_slope", type=float, default=1.0)
    parser.add_argument("-N_f", "--N_slope", type=int, default=7)
    parser.add_argument("-L", "--boundary", type=float, default=5.0)
    parser.add_argument("-bin", "--N_bins", type=int, default=100)
    parser.add_argument("-g", "--gamma", type=float, default=1.0)
    parser.add_argument("-T", "--temperature", type=float, default=0.001)
    parser.add_argument("-tau", "--tau", type=float, default=1.0)
    parser.add_argument("-Da", "--Da", type=float, default=1.0)
    parser.add_argument("-dt", "--delta_t", type=float, default=0.001)
    parser.add_argument("-init", "--initial", type=int, default=10000)
    parser.add_argument("-sam", "--sampling", type=int, default=100)
    parser.add_argument("-unit", "--interval", type=int, default=1000)
    parser.add_argument("-deg", "--degree", type=int, default=4)

    args = parser.parse_args()

    if args.mode == "manual":
        params = num_Params(
            velocity=args.velocity,
            Lambda=args.Lambda,
            slope=args.slope,
            N_particle=args.N_particle,
            N_ensemble=args.N_ensemble,
            boundary=args.boundary,
            N_bins=args.N_bins,
            gamma=args.gamma,
            temperature=args.temperature,
            tau=args.tau,
            Da=args.Da,
            delta_t=args.delta_t,
            initial=args.initial,
            sampling=args.sampling,
            interval=args.interval,
            degree=args.degree,
        )

        aoup = AOUP(params)
        # aoup.average_distribution(frames=1000)
        aoup.animate_histogram(frames=300, interval=100, fps=30)
        aoup.run_AOUP()

    elif args.mode == "velocity":
        # velocities = get_linspace(
        #     max_value=args.max_velocity, num=args.N_velocity)

        velocities = get_logspace(
            max_value=args.max_velocity, num=args.N_velocity)

        for velocity in velocities:
            params = num_Params(
                velocity=velocity,
                Lambda=args.Lambda,
                slope=args.slope,
                N_particle=args.N_particle,
                N_ensemble=args.N_ensemble,
                boundary=args.boundary,
                N_bins=args.N_bins,
                gamma=args.gamma,
                temperature=args.temperature,
                tau=args.tau,
                Da=args.Da,
                delta_t=args.delta_t,
                initial=args.initial,
                sampling=args.sampling,
                interval=args.interval,
                degree=args.degree,
            )

            # print(Params)

            aoup = AOUP(params)
            # aoup.average_distribution(frames=100)
            # aoup.animate_histogram(frames=300, interval=100, fps=30)
            aoup.run_AOUP()

    elif args.mode == "Lambda":
        Lambdas = get_logspace(
            max_value=args.max_Lambda, num=args.N_Lambda)

        for Lambda in Lambdas:
            params = num_Params(
                velocity=args.velocity,
                Lambda=Lambda,
                slope=args.slope,
                N_particle=args.N_particle,
                N_ensemble=args.N_ensemble,
                boundary=args.boundary,
                N_bins=args.N_bins,
                gamma=args.gamma,
                temperature=args.temperature,
                tau=args.tau,
                Da=args.Da,
                delta_t=args.delta_t,
                initial=args.initial,
                sampling=args.sampling,
                interval=args.interval,
                degree=args.degree,
            )

            # print(Params)

            aoup = AOUP(params)
            # aoup.average_distribution(frames=100)
            # aoup.animate_histogram(frames=300, interval=100, fps=10)
            aoup.run_AOUP()

    elif args.mode == "slope":
        slopes = get_logspace(
            max_value=args.max_slope, num=args.N_slope)

        for slope in slopes:
            params = num_Params(
                velocity=args.velocity,
                Lambda=args.Lambda,
                slope=slope,
                N_particle=args.N_particle,
                N_ensemble=args.N_ensemble,
                boundary=args.boundary,
                N_bins=args.N_bins,
                gamma=args.gamma,
                temperature=args.temperature,
                tau=args.tau,
                Da=args.Da,
                delta_t=args.delta_t,
                initial=args.initial,
                sampling=args.sampling,
                interval=args.interval,
                degree=args.degree,
            )

            # print(Params)

            aoup = AOUP(params)
            # aoup.average_distribution(frames=100)
            # aoup.animate_histogram(frames=300, interval=100, fps=10)
            aoup.run_AOUP()

    else:
        raise ValueError(
            "mode should be 'manual', 'velocity', 'Lambda', or 'slope'.")
