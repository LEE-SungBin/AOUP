import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from dataclasses import dataclass, field, asdict
from datetime import datetime
import pandas as pd
from pathlib import Path
from typing import Any, Self
import hashlib
import json
import pickle
import argparse
import time


@dataclass
class Parameter:
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

    def to_log(self) -> str:
        return " ".join(
            f"{log}" for log in asdict(self).values()
        )


class AOUP:
    def __init__(self, parameter: Parameter):
        self.parameter = parameter

        self.set_coeff()
        self.set_zero()

    def set_coeff(self) -> None:
        self.N_particle = self.parameter.N_particle  # * number of AOUPs
        self.N_ensemble = self.parameter.N_ensemble  # * number of ensemble
        self.velocity = self.parameter.velocity  # * velocity of object
        self.Lambda = self.parameter.Lambda      # * size of object
        self.boundary = self.parameter.boundary  # * size of periodic boundary
        self.N_bins = self.parameter.N_bins      # * number of histogram bins
        self.gamma = self.parameter.gamma        # * drag coefficient
        self.slope = self.parameter.slope        # * slope of potential
        self.temperature = self.parameter.temperature    # * temperature
        # * autocorrelation time-scale of colored noise
        self.tau = self.parameter.tau
        self.Da = self.parameter.Da              # * coefficients of colored noise
        self.delta_t = self.parameter.delta_t    # * update time interval
        # * number of iter to remove initial effect
        self.initial = self.parameter.initial
        self.sampling = self.parameter.sampling  # * number of iter to collect samples

    def set_zero(self) -> None:  # * initialize position and noise
        rng = np.random.default_rng()

        self.position = rng.uniform(
            low=-self.boundary/2, high=self.boundary/2, size=(self.N_ensemble, self.N_particle))

        self.colored_noise = rng.normal(
            loc=0.0, scale=1.0, size=(self.N_ensemble, self.N_particle))

    def get_result(self) -> None:  # * get average and std of drag
        now = time.perf_counter()
        for _ in range(self.initial):
            self.time_evolution()  # * update self.position and self.colored_noise

        drag = []  # * [self.sampling, self.N_ensemble]
        for _ in range(self.sampling):
            self.time_evolution()
            drag.append(self.get_drag())  # * append drag
        drag = np.array(drag)

        key = hashlib.sha1(str(self.parameter).encode()).hexdigest()[:6]
        data_dir, setting_dir = Path("data"), Path("setting")
        data_dir.mkdir(parents=True, exist_ok=True)
        setting_dir.mkdir(parents=True, exist_ok=True)
        name = key
        output = {
            "key": key,
        }
        output.update(asdict(self.parameter))

        with open(setting_dir / f"{name}.json", "w") as file:
            json.dump(output, file)  # * save setting

        result = {
            "average": np.mean(drag, axis=0),
            "std": np.std(drag, axis=0),
            "time": time.perf_counter()-now
        }
        output.update(result)

        with open(data_dir / f"{name}.pkl", "wb") as file:
            pickle.dump(output, file)  # * save result

        log = (
            f"{datetime.now().replace(microsecond=0)} {self.parameter.to_log()} {np.mean(drag)} {np.std(drag)} {time.perf_counter()-now}\n"
        )

        with open("log.txt", "a") as file:
            file.write(log)  # * save log

    def time_evolution(self) -> None:  # * time evolution of AOUPs
        rng = np.random.default_rng()

        force = self.get_force()

        self.position += (force / self.gamma - self.velocity) * self.delta_t
        self.position += self.colored_noise * self.delta_t
        self.position += rng.normal(
            loc=0.0,
            scale=np.sqrt(2 * self.temperature / self.gamma * self.delta_t),
            size=(self.N_ensemble, self.N_particle)
        )

        self.position = self.periodic_boundary(self.position)

        self.colored_noise += - self.colored_noise / self.tau * self.delta_t
        self.colored_noise += rng.normal(
            loc=0.0,  # * mean
            scale=np.sqrt(2 * self.Da / self.tau * self.delta_t),  # * std
            size=(self.N_ensemble, self.N_particle)
        )

    def get_drag(self) -> npt.NDArray:  # * calculate drag force
        positive_drag = self.slope * sum(
            1 for i in self.position.reshape(-1) if 0 <
            i < self.Lambda / 2) / self.N_ensemble

        negative_drag = self.slope * sum(
            1 for i in self.position.reshape(-1) if -
            self.Lambda / 2 < i < 0) / self.N_ensemble

        positive = (0 < self.position) & (self.position < self.Lambda / 2)
        positive_drag = positive.astype(np.int64).sum(axis=1) * self.slope

        negative = (- self.Lambda / 2 < self.position) & (self.position < 0)
        negative_drag = negative.astype(np.int64).sum(axis=1) * self.slope

        return positive_drag - negative_drag

    def get_force(self) -> npt.NDArray:  # * get external force from object
        force = np.zeros(shape=(self.N_ensemble, self.N_particle))

        force = np.where(np.array(
            [- self.Lambda / 2 < self.position, self.position < 0.0]).all(0), -self.slope, force)

        force = np.where(np.array(
            [0.0 < self.position, self.position < self.Lambda / 2]).all(0), self.slope, force)

        return force

    def animation(self) -> None:  # * animate histogram
        self.fig, self.ax = plt.subplots(tight_layout=True)
        self.bins = np.linspace(-self.boundary/2,
                                self.boundary/2, self.N_bins+1)

        self.ax.hist(self.position.reshape(-1), bins=self.bins)

        ani = animation.FuncAnimation(
            fig=self.fig, func=self.update, frames=300, interval=0, blit=False)

        ani.save(f"test.mp4", fps=30, extra_args=['-vcodec', 'libx264'])

    def update(self, i: int) -> None:  # * update animation
        print(i, end=" ")
        self.time_evolution()

        self.ax.cla()
        self.ax.hist(self.position[0], bins=self.bins)
        self.ax.axvline(-self.Lambda/2, linestyle="--", color="k")
        self.ax.axvline(0.0, linestyle="--", color="k")
        self.ax.axvline(self.Lambda/2, linestyle="--", color="k")
        self.ax.axhline(self.N_particle / self.N_bins,
                        linestyle="--", color="k")

        lx = np.linspace(-self.Lambda/2, 0, 10)
        rx = np.linspace(0, self.Lambda/2, 10)

        def f(x: npt.NDArray):
            return self.N_particle/self.N_bins * (1 + x / (self.Lambda / 2))

        def g(x: npt.NDArray):
            return self.N_particle/self.N_bins * (1 - x / (self.Lambda / 2))

        self.ax.plot(lx, f(lx), color="b")
        self.ax.plot(rx, g(rx), color="r")

        self.ax.set_xlim(-self.boundary/2, self.boundary/2)
        self.ax.set_ylim(0, self.N_particle/self.N_bins * 1.5)

        self.ax.text(
            0.99, 0.99, f"iter = {i+1}",
            verticalalignment="top", horizontalalignment='right',
            transform=self.ax.transAxes,
            color='black', fontsize=20
        )

        self.ax.text(
            0.99, 0.91, f"drag = {self.get_drag()[0]}",
            verticalalignment="top", horizontalalignment='right',
            transform=self.ax.transAxes,
            color='black', fontsize=20
        )

    # * periodic boundary condition

    def periodic_boundary(self, position: npt.NDArray) -> npt.NDArray:
        position = np.where(position > self.boundary / 2,
                            position - self.boundary * (((position - self.boundary/2) / self.boundary).astype(int) + 1), position)

        position = np.where(position < - self.boundary / 2,
                            position + self.boundary * ((-(position + self.boundary/2) / self.boundary).astype(int) + 1), position)

        return position


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-N", "--N_particle", type=int, default=10000)
    parser.add_argument("-ens", "--N_ensemble", type=int, default=100)
    parser.add_argument("-v", "--velocity", type=float, default=1.0)
    parser.add_argument("-d", "--Lambda", type=float, default=1.0)
    parser.add_argument("-L", "--boundary", type=float, default=5.0)
    parser.add_argument("-bin", "--N_bins", type=int, default=40)
    parser.add_argument("-g", "--gamma", type=float, default=1.0)
    parser.add_argument("-f", "--slope", type=float, default=1.0)
    parser.add_argument("-T", "--temperature", type=float, default=1.0)
    parser.add_argument("-tau", "--tau", type=float, default=5.0)
    parser.add_argument("-Da", "--Da", type=float, default=5.0)
    parser.add_argument("-dt", "--delta_t", type=float, default=0.01)
    parser.add_argument("-init", "--initial", type=int, default=10000)
    parser.add_argument("-sam", "--sampling", type=int, default=10000)

    args = parser.parse_args()

    parameter = Parameter(
        N_particle=args.N_particle,
        N_ensemble=args.N_ensemble,
        velocity=args.velocity,
        Lambda=args.Lambda,
        boundary=args.boundary,
        N_bins=args.N_bins,
        gamma=args.gamma,
        slope=args.slope,
        temperature=args.temperature,
        tau=args.tau,
        Da=args.Da,
        delta_t=args.delta_t,
        initial=args.initial,
        sampling=args.sampling,
    )

    # print(parameter)

    aoup = AOUP(parameter)
    # aoup.animation()
    aoup.get_result()
